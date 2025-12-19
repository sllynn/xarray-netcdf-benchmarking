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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse
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
    storage_account: str
    container: str


class TokenManager:
    """Manages SAS token generation and refresh for Azure Blob Storage.
    
    Uses Databricks SDK to generate short-lived SAS tokens for azcopy.
    Implements a "Check-Before-Act" pattern to proactively refresh tokens
    before they expire.
    
    Parameters
    ----------
    volume_name : str
        Unity Catalog Volume full name (e.g., 'catalog.schema.volume_name').
    subpath : str, optional
        Subpath within the volume (e.g., 'forecast.zarr').
    refresh_buffer_minutes : int
        Minutes before expiration to trigger refresh (default: 5).
    
    Examples
    --------
    >>> token_manager = TokenManager('my_catalog.my_schema.silver', subpath='forecast.zarr')
    >>> sas_url = token_manager.get_sas_url()
    >>> subprocess.run(['azcopy', 'sync', source, sas_url])
    """
    
    def __init__(
        self,
        volume_name: str,
        subpath: Optional[str] = None,
        refresh_buffer_minutes: int = 5,
    ):
        self.volume_name = volume_name
        self.subpath = subpath
        self.refresh_buffer_minutes = refresh_buffer_minutes
        
        self._token_info: Optional[TokenInfo] = None
        self._workspace_client = None
        self._volume_storage_location: Optional[str] = None
    
    @classmethod
    def from_volume_path(
        cls,
        volume_path: str,
        refresh_buffer_minutes: int = 5,
    ) -> "TokenManager":
        """Create TokenManager from a Volume path.
        
        Parameters
        ----------
        volume_path : str
            Volume path like '/Volumes/catalog/schema/volume_name/subpath'.
        refresh_buffer_minutes : int
            Minutes before expiration to trigger refresh.
        
        Returns
        -------
        TokenManager
            Configured token manager.
        """
        # Parse /Volumes/catalog/schema/volume_name/subpath
        parts = volume_path.strip('/').split('/')
        if len(parts) < 4 or parts[0].lower() != 'volumes':
            raise ValueError(
                f"Invalid volume path: {volume_path}. "
                "Expected format: /Volumes/catalog/schema/volume_name[/subpath]"
            )
        
        catalog, schema, volume = parts[1], parts[2], parts[3]
        volume_name = f"{catalog}.{schema}.{volume}"
        subpath = '/'.join(parts[4:]) if len(parts) > 4 else None
        
        return cls(
            volume_name=volume_name,
            subpath=subpath,
            refresh_buffer_minutes=refresh_buffer_minutes,
        )
    
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
    
    def _get_volume_storage_location(self) -> str:
        """Get the underlying storage location for the Volume."""
        if self._volume_storage_location is None:
            w = self._get_workspace_client()
            volume_info = w.volumes.read(self.volume_name)
            self._volume_storage_location = volume_info.storage_location
            logger.info(f"Volume {self.volume_name} -> {self._volume_storage_location}")
        return self._volume_storage_location
    
    def _parse_storage_url(self, storage_url: str) -> tuple[str, str, str]:
        """Parse abfss:// URL into storage account, container, and path.
        
        Parameters
        ----------
        storage_url : str
            URL like 'abfss://container@account.dfs.core.windows.net/path'.
        
        Returns
        -------
        tuple[str, str, str]
            (storage_account, container, path)
        """
        # abfss://container@account.dfs.core.windows.net/path
        parsed = urlparse(storage_url)
        
        # Host is container@account.dfs.core.windows.net
        container, host_rest = parsed.netloc.split('@', 1)
        storage_account = host_rest.split('.')[0]
        path = parsed.path.lstrip('/')
        
        return storage_account, container, path
    
    def _needs_refresh(self) -> bool:
        """Check if token needs to be refreshed."""
        if self._token_info is None:
            return True
        
        buffer = timedelta(minutes=self.refresh_buffer_minutes)
        now = datetime.now(timezone.utc)
        
        # Handle both timezone-aware and naive expiration times
        expiration = self._token_info.expiration
        if expiration.tzinfo is None:
            expiration = expiration.replace(tzinfo=timezone.utc)
        
        return now > (expiration - buffer)
    
    def _refresh_token(self) -> None:
        """Generate a new SAS token using Databricks SDK."""
        logger.info(f"Refreshing SAS token for volume {self.volume_name}")
        
        try:
            from databricks.sdk.service.catalog import PathOperation
            
            w = self._get_workspace_client()
            
            # Get the storage location from the Volume metadata
            volume_root_url = self._get_volume_storage_location()
            
            # Generate temporary write credentials for the Volume
            creds = w.temporary_path_credentials.generate_temporary_path_credentials(
                url=volume_root_url,
                operation=PathOperation.PATH_READ_WRITE,
            )
            
            # Extract Azure SAS token from response
            if creds.azure_user_delegation_sas is None:
                raise ValueError("No Azure credentials returned from Databricks")
            
            sas_token = creds.azure_user_delegation_sas.sas_token
            
            # Parse the storage URL
            storage_account, container, base_path = self._parse_storage_url(volume_root_url)
            
            # Build the Azure Blob URL (https:// format for azcopy)
            if self.subpath:
                blob_path = f"{base_path}/{self.subpath}".strip('/')
            else:
                blob_path = base_path
            
            azure_url = f"https://{storage_account}.blob.core.windows.net/{container}/{blob_path}"
            
            # Parse expiration time from response
            # expiration_time can be: None, int (Unix timestamp ms), str (ISO format), or datetime
            expiration_raw = creds.expiration_time
            if expiration_raw is None:
                # Fallback: assume 1 hour from now
                expiration = datetime.now(timezone.utc) + timedelta(hours=1)
            elif isinstance(expiration_raw, int):
                # Unix timestamp in milliseconds
                expiration = datetime.fromtimestamp(expiration_raw / 1000, tz=timezone.utc)
            elif isinstance(expiration_raw, str):
                # Parse ISO format string
                expiration = datetime.fromisoformat(expiration_raw.replace('Z', '+00:00'))
            elif isinstance(expiration_raw, datetime):
                expiration = expiration_raw
            else:
                # Unknown type, fallback
                logger.warning(f"Unknown expiration_time type: {type(expiration_raw)}, using 1 hour default")
                expiration = datetime.now(timezone.utc) + timedelta(hours=1)
            
            self._token_info = TokenInfo(
                token=sas_token,
                expiration=expiration,
                azure_url=azure_url,
                storage_account=storage_account,
                container=container,
            )
            
            logger.info(f"Token refreshed, expires at {expiration}")
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            raise
    
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
    
    def get_azure_url(self) -> str:
        """Get the Azure Blob URL without the SAS token.
        
        Returns
        -------
        str
            Azure Blob URL (useful for logging without exposing token).
        """
        if self._token_info is None:
            self._refresh_token()
        return self._token_info.azure_url
    
    @property
    def token_expiration(self) -> Optional[datetime]:
        """Get current token expiration time."""
        return self._token_info.expiration if self._token_info else None
    
    @property
    def storage_account(self) -> Optional[str]:
        """Get the storage account name."""
        return self._token_info.storage_account if self._token_info else None
    
    @property
    def container(self) -> Optional[str]:
        """Get the container name."""
        return self._token_info.container if self._token_info else None


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
    
    # Check common locations including our install location
    common_paths = [
        '/usr/local/bin/azcopy',
        '/usr/bin/azcopy',
        os.path.expanduser('~/bin/azcopy'),
        '/databricks/driver/bin/azcopy',
        '/local_disk0/bin/azcopy',
        '/tmp/azcopy/azcopy',  # Our install location
    ]
    
    # Also check for extracted azcopy in /tmp/azcopy/*/
    import glob
    common_paths.extend(glob.glob('/tmp/azcopy/*/azcopy'))
    
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    return None


def install_azcopy(install_dir: str = "/tmp/azcopy") -> str:
    """Download and install azcopy to a local directory.
    
    Parameters
    ----------
    install_dir : str
        Directory to install azcopy to (default: /tmp/azcopy).
    
    Returns
    -------
    str
        Path to the installed azcopy executable.
    
    Raises
    ------
    RuntimeError
        If installation fails.
    """
    import glob
    import platform
    
    # Check if already installed
    existing = glob.glob(f"{install_dir}/*/azcopy")
    if existing:
        logger.info(f"azcopy already installed at {existing[0]}")
        return existing[0]
    
    # Determine download URL based on platform
    system = platform.system().lower()
    if system == "linux":
        download_url = "https://aka.ms/downloadazcopy-v10-linux"
    elif system == "darwin":
        download_url = "https://aka.ms/downloadazcopy-v10-mac"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")
    
    logger.info(f"Installing azcopy for {system} to {install_dir}...")
    
    os.makedirs(install_dir, exist_ok=True)
    
    try:
        # Download azcopy
        download_result = subprocess.run(
            [
                "curl", "-sL",
                download_url,
                "-o", f"{install_dir}/azcopy.tar.gz"
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if download_result.returncode != 0:
            raise RuntimeError(f"Failed to download azcopy: {download_result.stderr}")
        
        # Extract
        extract_result = subprocess.run(
            [
                "tar", "-xzf", f"{install_dir}/azcopy.tar.gz",
                "-C", install_dir
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if extract_result.returncode != 0:
            raise RuntimeError(f"Failed to extract azcopy: {extract_result.stderr}")
        
        # Find the extracted binary
        azcopy_paths = glob.glob(f"{install_dir}/*/azcopy")
        if not azcopy_paths:
            raise RuntimeError(f"azcopy binary not found after extraction")
        
        azcopy_path = azcopy_paths[0]
        
        # Verify it works
        verify_result = subprocess.run(
            [azcopy_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if verify_result.returncode != 0:
            raise RuntimeError(f"azcopy verification failed: {verify_result.stderr}")
        
        logger.info(f"azcopy installed: {verify_result.stdout.strip()}")
        return azcopy_path
        
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"azcopy installation timed out: {e}")
    except Exception as e:
        raise RuntimeError(f"azcopy installation failed: {e}")


def ensure_azcopy() -> str:
    """Find azcopy or install it if not available.
    
    Returns
    -------
    str
        Path to azcopy executable.
    
    Raises
    ------
    RuntimeError
        If azcopy cannot be found or installed.
    """
    azcopy_path = find_azcopy()
    if azcopy_path:
        return azcopy_path
    
    # Try to install
    return install_azcopy()


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
    # Try to find or auto-install azcopy
    try:
        azcopy_path = ensure_azcopy()
    except RuntimeError as e:
        return SyncResult(
            success=False,
            source_path=source_path,
            destination_path=destination_url.split('?')[0],
            files_transferred=0,
            bytes_transferred=0,
            elapsed_seconds=0,
            error=str(e),
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
    volume_name : str
        Unity Catalog Volume full name (e.g., 'catalog.schema.volume_name').
    subpath : str, optional
        Subpath within the volume (e.g., 'forecast.zarr').
    token_manager : TokenManager, optional
        Custom token manager. Created automatically if not provided.
    
    Examples
    --------
    >>> syncer = CloudSyncer('my_catalog.my_schema.silver', subpath='forecast.zarr')
    >>> result = syncer.sync('/local_disk0/forecast.zarr')
    >>> print(f"Synced {result.files_transferred} files in {result.elapsed_seconds:.1f}s")
    
    Or create from a Volume path:
    
    >>> syncer = CloudSyncer.from_volume_path('/Volumes/catalog/schema/volume/forecast.zarr')
    >>> result = syncer.sync('/local_disk0/forecast.zarr')
    """
    
    def __init__(
        self,
        volume_name: str,
        subpath: Optional[str] = None,
        token_manager: Optional[TokenManager] = None,
    ):
        self.volume_name = volume_name
        self.subpath = subpath
        self.token_manager = token_manager or TokenManager(volume_name, subpath=subpath)
    
    @classmethod
    def from_volume_path(
        cls,
        volume_path: str,
        token_manager: Optional[TokenManager] = None,
    ) -> "CloudSyncer":
        """Create CloudSyncer from a Volume path.
        
        Parameters
        ----------
        volume_path : str
            Volume path like '/Volumes/catalog/schema/volume_name/subpath'.
        token_manager : TokenManager, optional
            Custom token manager.
        
        Returns
        -------
        CloudSyncer
            Configured syncer.
        """
        # Parse /Volumes/catalog/schema/volume_name/subpath
        parts = volume_path.strip('/').split('/')
        if len(parts) < 4 or parts[0].lower() != 'volumes':
            raise ValueError(
                f"Invalid volume path: {volume_path}. "
                "Expected format: /Volumes/catalog/schema/volume_name[/subpath]"
            )
        
        catalog, schema, volume = parts[1], parts[2], parts[3]
        volume_name = f"{catalog}.{schema}.{volume}"
        subpath = '/'.join(parts[4:]) if len(parts) > 4 else None
        
        if token_manager is None:
            token_manager = TokenManager(volume_name, subpath=subpath)
        
        return cls(
            volume_name=volume_name,
            subpath=subpath,
            token_manager=token_manager,
        )
    
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
    print("  # Option 1: Using volume name directly")
    print("  token_manager = TokenManager('catalog.schema.volume', subpath='forecast.zarr')")
    print("  sas_url = token_manager.get_sas_url()")
    print()
    print("  # Option 2: Using volume path")
    print("  token_manager = TokenManager.from_volume_path('/Volumes/catalog/schema/volume/forecast.zarr')")
    print("  sas_url = token_manager.get_sas_url()")
    print()
    print("CloudSyncer demo:")
    print("  syncer = CloudSyncer.from_volume_path('/Volumes/catalog/schema/volume/forecast.zarr')")
    print("  result = syncer.sync('/local_disk0/forecast.zarr')")
    print("  print(f'Synced {result.files_transferred} files in {result.elapsed_seconds:.1f}s')")

