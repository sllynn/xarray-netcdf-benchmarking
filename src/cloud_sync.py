#!/usr/bin/env python3
"""
Cloud Sync Module with Token Management.

Implements secure cloud sync using AzCopy with SAS token refresh for
syncing local Zarr stores to Azure Blob Storage (Unity Catalog Volumes).

Key features:
- TokenManager class for SAS token generation and refresh
- Uses databricks.sdk.temporary_path_credentials for token generation
- "Check-Before-Act" pattern: refresh if now > (expiration - 5_mins)
- Wrapper for azcopy sync with retry logic
"""

import os
import subprocess
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, urlencode
import time

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    source_path: str
    destination_path: str
    files_transferred: int
    bytes_transferred: int
    elapsed_seconds: float
    error: Optional[str] = None
    command: Optional[str] = None


@dataclass
class TokenInfo:
    """SAS token information."""
    token: str
    expiration: datetime
    azure_url: str


class TokenManager:
    """Manages SAS token generation and refresh for Azure Blob Storage.
    
    Uses Databricks SDK to generate short-lived SAS tokens for azcopy.
    Implements a "Check-Before-Act" pattern to proactively refresh tokens
    before they expire.
    
    Parameters
    ----------
    volume_path : str
        Unity Catalog Volume path (e.g., '/Volumes/catalog/schema/volume_name').
    refresh_buffer_minutes : int
        Minutes before expiration to trigger refresh (default: 5).
    token_duration_hours : int
        Requested token duration in hours (default: 1).
    
    Examples
    --------
    >>> token_manager = TokenManager('/Volumes/my_catalog/my_schema/silver')
    >>> sas_url = token_manager.get_sas_url()
    >>> subprocess.run(['azcopy', 'sync', source, sas_url])
    """
    
    def __init__(
        self,
        volume_path: str,
        refresh_buffer_minutes: int = 5,
        token_duration_hours: int = 1,
    ):
        self.volume_path = volume_path
        self.refresh_buffer_minutes = refresh_buffer_minutes
        self.token_duration_hours = token_duration_hours
        
        self._token_info: Optional[TokenInfo] = None
        self._workspace_client = None
    
    def _get_workspace_client(self):
        """Get or create Databricks workspace client."""
        if self._workspace_client is None:
            try:
                from databricks.sdk import WorkspaceClient
                self._workspace_client = WorkspaceClient()
            except ImportError:
                raise ImportError(
                    "databricks-sdk is required for token management. "
                    "Install with: pip install databricks-sdk"
                )
        return self._workspace_client
    
    def _needs_refresh(self) -> bool:
        """Check if token needs to be refreshed."""
        if self._token_info is None:
            return True
        
        buffer = timedelta(minutes=self.refresh_buffer_minutes)
        return datetime.utcnow() > (self._token_info.expiration - buffer)
    
    def _refresh_token(self) -> None:
        """Generate a new SAS token using Databricks SDK."""
        logger.info(f"Refreshing SAS token for {self.volume_path}")
        
        try:
            w = self._get_workspace_client()
            
            # Get temporary credentials for the volume path
            # This uses the Unity Catalog temporary_path_credentials API
            from databricks.sdk.service.catalog import (
                GenerateTemporaryPathCredentialRequest,
                PathOperation,
            )
            
            response = w.path_credentials.generate_temporary_path_credential(
                GenerateTemporaryPathCredentialRequest(
                    path=self.volume_path,
                    operation=PathOperation.WRITE,
                )
            )
            
            # Extract Azure SAS token from response
            azure_credentials = response.azure_user_delegation_sas
            if azure_credentials is None:
                raise ValueError("No Azure credentials returned from Databricks")
            
            sas_token = azure_credentials.sas_token
            
            # Parse the destination URL
            # Volume paths map to underlying Azure Blob Storage URLs
            storage_url = self._get_storage_url()
            
            # Calculate expiration
            expiration = datetime.utcnow() + timedelta(hours=self.token_duration_hours)
            
            self._token_info = TokenInfo(
                token=sas_token,
                expiration=expiration,
                azure_url=storage_url,
            )
            
            logger.info(f"Token refreshed, expires at {expiration}")
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            raise
    
    def _get_storage_url(self) -> str:
        """Get the underlying Azure Blob Storage URL for the volume.
        
        In a real implementation, this would query the Unity Catalog to
        get the storage location mapping. For now, we construct it from
        environment or configuration.
        """
        # Try to get from environment (set in Databricks)
        storage_account = os.environ.get('AZURE_STORAGE_ACCOUNT')
        container = os.environ.get('AZURE_STORAGE_CONTAINER')
        
        if storage_account and container:
            # Parse volume path to get subdirectory
            parts = self.volume_path.strip('/').split('/')
            if len(parts) >= 4 and parts[0].lower() == 'volumes':
                subpath = '/'.join(parts[3:])  # Skip Volumes/catalog/schema
            else:
                subpath = parts[-1] if parts else ''
            
            return f"https://{storage_account}.blob.core.windows.net/{container}/{subpath}"
        
        # Fallback: assume DBFS-style URL conversion
        logger.warning(
            "Storage URL not configured. Set AZURE_STORAGE_ACCOUNT and "
            "AZURE_STORAGE_CONTAINER environment variables."
        )
        return f"abfss://container@account.dfs.core.windows.net{self.volume_path}"
    
    def get_sas_url(self) -> str:
        """Get destination URL with valid SAS token.
        
        Returns
        -------
        str
            Full URL with SAS token appended as query parameter.
        """
        if self._needs_refresh():
            self._refresh_token()
        
        return f"{self._token_info.azure_url}?{self._token_info.token}"
    
    @property
    def token_expiration(self) -> Optional[datetime]:
        """Get current token expiration time."""
        return self._token_info.expiration if self._token_info else None


def find_azcopy() -> Optional[str]:
    """Find azcopy executable on the system.
    
    Returns
    -------
    str or None
        Path to azcopy executable, or None if not found.
    """
    # Check if azcopy is in PATH
    azcopy_path = shutil.which('azcopy')
    if azcopy_path:
        return azcopy_path
    
    # Check common locations
    common_paths = [
        '/usr/local/bin/azcopy',
        '/usr/bin/azcopy',
        os.path.expanduser('~/bin/azcopy'),
        '/databricks/driver/bin/azcopy',
        '/local_disk0/bin/azcopy',
    ]
    
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    return None


def sync_with_azcopy(
    source_path: str,
    destination_url: str,
    delete_destination: bool = False,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    log_level: str = 'WARNING',
    dry_run: bool = False,
    retry_count: int = 3,
    retry_delay_seconds: float = 1.0,
) -> SyncResult:
    """Sync local directory to Azure Blob Storage using azcopy.
    
    Parameters
    ----------
    source_path : str
        Local source directory path.
    destination_url : str
        Destination URL with SAS token.
    delete_destination : bool
        Delete files in destination not in source (default: False).
    include_pattern : str, optional
        Only sync files matching this pattern.
    exclude_pattern : str, optional
        Exclude files matching this pattern.
    log_level : str
        AzCopy log level (default: 'WARNING').
    dry_run : bool
        If True, only report what would be synced (default: False).
    retry_count : int
        Number of retry attempts on failure (default: 3).
    retry_delay_seconds : float
        Delay between retries in seconds (default: 1.0).
    
    Returns
    -------
    SyncResult
        Result of the sync operation.
    """
    azcopy_path = find_azcopy()
    if azcopy_path is None:
        return SyncResult(
            success=False,
            source_path=source_path,
            destination_path=destination_url.split('?')[0],
            files_transferred=0,
            bytes_transferred=0,
            elapsed_seconds=0,
            error="azcopy not found. Install from https://aka.ms/downloadazcopy",
        )
    
    # Build command
    cmd = [
        azcopy_path,
        'sync',
        source_path,
        destination_url,
        '--recursive=true',
        f'--log-level={log_level}',
    ]
    
    if delete_destination:
        cmd.append('--delete-destination=true')
    
    if include_pattern:
        cmd.append(f'--include-pattern={include_pattern}')
    
    if exclude_pattern:
        cmd.append(f'--exclude-pattern={exclude_pattern}')
    
    if dry_run:
        cmd.append('--dry-run')
    
    # Redact SAS token from logs
    cmd_display = [c if '?' not in c else c.split('?')[0] + '?[REDACTED]' for c in cmd]
    logger.info(f"Running: {' '.join(cmd_display)}")
    
    start_time = time.perf_counter()
    last_error = None
    
    for attempt in range(retry_count):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            elapsed = time.perf_counter() - start_time
            
            if result.returncode == 0:
                # Parse output for statistics
                files_transferred = 0
                bytes_transferred = 0
                
                for line in result.stdout.split('\n'):
                    if 'Files Transferred:' in line or 'Total files transferred:' in line:
                        try:
                            files_transferred = int(line.split(':')[-1].strip())
                        except ValueError:
                            pass
                    elif 'Bytes Transferred:' in line or 'Total bytes transferred:' in line:
                        try:
                            bytes_transferred = int(line.split(':')[-1].strip())
                        except ValueError:
                            pass
                
                return SyncResult(
                    success=True,
                    source_path=source_path,
                    destination_path=destination_url.split('?')[0],
                    files_transferred=files_transferred,
                    bytes_transferred=bytes_transferred,
                    elapsed_seconds=elapsed,
                    command=' '.join(cmd_display),
                )
            else:
                last_error = result.stderr or result.stdout or f"Exit code {result.returncode}"
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                
        except subprocess.TimeoutExpired:
            last_error = "Sync operation timed out after 1 hour"
            logger.warning(f"Attempt {attempt + 1} timed out")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} error: {e}")
        
        if attempt < retry_count - 1:
            time.sleep(retry_delay_seconds)
    
    elapsed = time.perf_counter() - start_time
    
    return SyncResult(
        success=False,
        source_path=source_path,
        destination_path=destination_url.split('?')[0],
        files_transferred=0,
        bytes_transferred=0,
        elapsed_seconds=elapsed,
        error=last_error,
        command=' '.join(cmd_display),
    )


class CloudSyncer:
    """High-level interface for syncing Zarr stores to cloud storage.
    
    Combines TokenManager and azcopy sync with automatic token refresh.
    
    Parameters
    ----------
    volume_path : str
        Unity Catalog Volume path.
    token_manager : TokenManager, optional
        Custom token manager. Created automatically if not provided.
    
    Examples
    --------
    >>> syncer = CloudSyncer('/Volumes/my_catalog/my_schema/silver')
    >>> result = syncer.sync('/local_disk0/forecast.zarr')
    >>> print(f"Synced {result.files_transferred} files in {result.elapsed_seconds:.1f}s")
    """
    
    def __init__(
        self,
        volume_path: str,
        token_manager: Optional[TokenManager] = None,
    ):
        self.volume_path = volume_path
        self.token_manager = token_manager or TokenManager(volume_path)
    
    def sync(
        self,
        source_path: str,
        subdirectory: Optional[str] = None,
        **kwargs,
    ) -> SyncResult:
        """Sync local path to cloud storage.
        
        Parameters
        ----------
        source_path : str
            Local source directory.
        subdirectory : str, optional
            Subdirectory within the volume.
        **kwargs
            Additional arguments passed to sync_with_azcopy.
        
        Returns
        -------
        SyncResult
            Result of the sync operation.
        """
        # Get destination URL with fresh token
        destination_url = self.token_manager.get_sas_url()
        
        if subdirectory:
            # Append subdirectory to URL (before the SAS token)
            base_url, token = destination_url.split('?', 1)
            destination_url = f"{base_url.rstrip('/')}/{subdirectory}?{token}"
        
        return sync_with_azcopy(source_path, destination_url, **kwargs)
    
    def sync_zarr_chunks(
        self,
        zarr_path: str,
        changed_chunks: Optional[list[str]] = None,
        **kwargs,
    ) -> SyncResult:
        """Sync Zarr store, optionally limiting to changed chunks.
        
        Parameters
        ----------
        zarr_path : str
            Path to local Zarr store.
        changed_chunks : list[str], optional
            List of chunk paths that changed. If None, syncs everything.
        **kwargs
            Additional arguments passed to sync_with_azcopy.
        
        Returns
        -------
        SyncResult
            Result of the sync operation.
        """
        if changed_chunks:
            # For selective sync, we'd need to construct include patterns
            # For now, just sync everything (azcopy handles incremental)
            logger.info(f"Syncing Zarr with {len(changed_chunks)} changed chunks")
        
        return self.sync(zarr_path, **kwargs)


# Fallback sync using dbutils.fs (slower but always available on Databricks)
def sync_with_dbutils(
    source_path: str,
    destination_path: str,
    dbutils=None,
) -> SyncResult:
    """Sync using dbutils.fs.cp (fallback when azcopy not available).
    
    WARNING: This is significantly slower than azcopy and does not support
    differential sync. Use only as a fallback.
    
    Parameters
    ----------
    source_path : str
        Local source directory.
    destination_path : str
        Destination path (e.g., '/Volumes/catalog/schema/volume').
    dbutils : object, optional
        Databricks dbutils object. If None, attempts to get from globals.
    
    Returns
    -------
    SyncResult
        Result of the sync operation.
    """
    start_time = time.perf_counter()
    
    if dbutils is None:
        try:
            # Try to get dbutils from Databricks runtime
            import IPython
            dbutils = IPython.get_ipython().user_ns.get('dbutils')
        except Exception:
            pass
    
    if dbutils is None:
        return SyncResult(
            success=False,
            source_path=source_path,
            destination_path=destination_path,
            files_transferred=0,
            bytes_transferred=0,
            elapsed_seconds=0,
            error="dbutils not available outside Databricks environment",
        )
    
    try:
        # Note: This copies ALL files, not just changed ones
        logger.warning(
            "Using dbutils.fs.cp for sync. This is slower than azcopy "
            "and does not support differential sync."
        )
        
        dbutils.fs.cp(source_path, destination_path, recurse=True)
        
        elapsed = time.perf_counter() - start_time
        
        return SyncResult(
            success=True,
            source_path=source_path,
            destination_path=destination_path,
            files_transferred=-1,  # Unknown with dbutils
            bytes_transferred=-1,  # Unknown with dbutils
            elapsed_seconds=elapsed,
        )
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return SyncResult(
            success=False,
            source_path=source_path,
            destination_path=destination_path,
            files_transferred=0,
            bytes_transferred=0,
            elapsed_seconds=elapsed,
            error=str(e),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check if azcopy is available
    azcopy = find_azcopy()
    if azcopy:
        print(f"azcopy found at: {azcopy}")
    else:
        print("azcopy not found on this system")
    
    # Demo token manager (will fail without Databricks environment)
    print("\nTokenManager demo (requires Databricks environment):")
    print("  token_manager = TokenManager('/Volumes/catalog/schema/silver')")
    print("  sas_url = token_manager.get_sas_url()")

