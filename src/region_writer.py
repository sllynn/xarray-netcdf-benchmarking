#!/usr/bin/env python3
"""
GRIB to Zarr Region Write Module.

Implements the core processing logic that reads a GRIB file and writes it to a
specific "slot" in the pre-allocated Zarr store using region writes.

Key features:
- Parse GRIB file to extract forecast hour and variable
- Map forecast hour to step index using the non-uniform step lookup
- Perform atomic region write using xarray.to_zarr(region=...)
- Handle concurrent writes safely (one file = one chunk)
- Stage files to local SSD for faster reads from cloud storage
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr
import shutil
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import logging

try:
    import eccodes
    HAS_ECCODES = True
except ImportError:
    HAS_ECCODES = False

try:
    import cfgrib
    HAS_CFGRIB = True
except ImportError:
    HAS_CFGRIB = False

from .zarr_init import generate_forecast_steps, build_hour_to_index_map

logger = logging.getLogger(__name__)

# Default staging directory on Databricks (local SSD)
DEFAULT_STAGING_DIR = "/local_disk0/grib_staging"

# Fallback mapping from GRIB shortName to CF variable names
# Used only when cfVarName is not available in the GRIB file
# GRIB uses names like "2t", "10u", "10v" while CF conventions use "t2m", "u10", "v10"
# Reference: ECMWF Parameter Database https://confluence.ecmwf.int/display/ECC/GRIB+Parameters+Database
GRIB_TO_CF_VARIABLE_MAP = {
    "2t": "t2m",      # 2-metre temperature
    "10u": "u10",     # 10-metre U wind component  
    "10v": "v10",     # 10-metre V wind component
    "sp": "sp",       # Surface pressure (same in both)
    "msl": "msl",     # Mean sea level pressure
    "tp": "tp",       # Total precipitation
    "tcc": "tcc",     # Total cloud cover
    "skt": "skt",     # Skin temperature
    "stl1": "stl1",   # Soil temperature level 1
    # Add more mappings as needed
}


def get_cf_varname_from_grib(gid: int) -> str:
    """Extract the CF variable name from a GRIB message handle.
    
    First tries to read cfVarName directly from the GRIB file (set by eccodes
    based on ECMWF parameter definitions). Falls back to a hardcoded mapping
    if cfVarName is not available.
    
    Parameters
    ----------
    gid : int
        eccodes GRIB message handle.
    
    Returns
    -------
    str
        CF/Zarr variable name (e.g., "t2m", "u10", "v10").
    """
    # First try to get cfVarName directly from eccodes
    try:
        cf_var_name = eccodes.codes_get(gid, "cfVarName")
        if cf_var_name and cf_var_name != "unknown":
            return cf_var_name
    except eccodes.CodesInternalError:
        pass
    
    # Fall back to shortName with manual mapping
    short_name = eccodes.codes_get(gid, "shortName")
    return GRIB_TO_CF_VARIABLE_MAP.get(short_name, short_name)


def stage_file_locally(
    remote_path: str,
    staging_dir: str = DEFAULT_STAGING_DIR,
) -> str:
    """Copy a file from remote/cloud storage to local SSD for faster reading.
    
    Parameters
    ----------
    remote_path : str
        Path to the file (can be DBFS, Volumes, or cloud storage path).
    staging_dir : str
        Local directory to copy files to (default: /local_disk0/grib_staging).
    
    Returns
    -------
    str
        Path to the local copy of the file.
    """
    import time
    remote_path = str(remote_path)
    
    # Volumes are accessed directly at /Volumes/... (FUSE mount)
    # DBFS is accessed at /dbfs/... (legacy FUSE mount)
    # Both can be slow for random I/O, so we copy to local SSD
    fuse_path = remote_path
    
    # Create staging directory
    os.makedirs(staging_dir, exist_ok=True)
    
    # Generate local path
    filename = Path(remote_path).name
    local_path = os.path.join(staging_dir, filename)
    
    # Copy to local with timing
    t0 = time.perf_counter()
    shutil.copy2(fuse_path, local_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    # Log with file size for throughput analysis
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    throughput_mbps = file_size_mb / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    logger.debug(
        f"Staged {filename}: {file_size_mb:.1f}MB in {elapsed_ms:.0f}ms "
        f"({throughput_mbps:.1f}MB/s)"
    )
    
    return local_path


def cleanup_staged_file(local_path: str, staging_dir: str = DEFAULT_STAGING_DIR) -> None:
    """Remove a staged file if it's in the staging directory.
    
    Parameters
    ----------
    local_path : str
        Path to the local file.
    staging_dir : str
        The staging directory (only files here will be deleted).
    """
    if local_path.startswith(staging_dir):
        try:
            os.remove(local_path)
        except OSError:
            pass  # File may already be deleted


@dataclass
class GribMetadata:
    """Metadata extracted from a GRIB file."""
    variable: str
    forecast_hour: int
    reference_time: datetime
    valid_time: datetime
    ensemble_member: Optional[int] = None
    n_ensemble: Optional[int] = None
    shape: tuple = None


@dataclass
class WriteResult:
    """Result of a region write operation."""
    success: bool
    grib_path: str
    variable: str
    step_index: int
    ensemble_slice: tuple
    elapsed_ms: float
    error: Optional[str] = None


def extract_grib_metadata(grib_path: str) -> GribMetadata:
    """Extract metadata from a GRIB file.
    
    Parameters
    ----------
    grib_path : str
        Path to the GRIB file.
    
    Returns
    -------
    GribMetadata
        Extracted metadata including variable, forecast hour, times, etc.
    
    Raises
    ------
    ImportError
        If eccodes is not available.
    ValueError
        If required metadata cannot be extracted.
    """
    if not HAS_ECCODES:
        raise ImportError(
            "eccodes is required for GRIB metadata extraction. "
            "Install with: pip install eccodes"
        )
    
    grib_path = str(grib_path)
    
    with open(grib_path, 'rb') as f:
        gid = eccodes.codes_grib_new_from_file(f)
        if gid is None:
            raise ValueError(f"Could not read GRIB message from {grib_path}")
        
        try:
            # Extract variable name using CF convention (from cfVarName or fallback mapping)
            var_name = get_cf_varname_from_grib(gid)
            
            # Extract time information
            data_date = eccodes.codes_get(gid, "dataDate")  # YYYYMMDD
            data_time = eccodes.codes_get(gid, "dataTime")  # HHMM
            step = eccodes.codes_get(gid, "step")  # Forecast step in hours
            
            # Parse reference time
            year = data_date // 10000
            month = (data_date % 10000) // 100
            day = data_date % 100
            hour = data_time // 100
            minute = data_time % 100
            reference_time = datetime(year, month, day, hour, minute)
            
            # Calculate valid time
            valid_time = reference_time + timedelta(hours=int(step))
            
            # Extract ensemble information if available
            try:
                perturbation_number = eccodes.codes_get(gid, "perturbationNumber")
                n_ensemble = eccodes.codes_get(gid, "numberOfForecastsInEnsemble")
                # Convert to 0-based index
                ensemble_member = perturbation_number - 1 if perturbation_number > 0 else None
            except eccodes.CodesInternalError:
                ensemble_member = None
                n_ensemble = None
            
            # Get shape
            n_lat = eccodes.codes_get(gid, "Nj")
            n_lon = eccodes.codes_get(gid, "Ni")
            
            return GribMetadata(
                variable=var_name,
                forecast_hour=int(step),
                reference_time=reference_time,
                valid_time=valid_time,
                ensemble_member=ensemble_member,
                n_ensemble=n_ensemble,
                shape=(n_lat, n_lon),
            )
        finally:
            eccodes.codes_release(gid)


def read_grib_data(grib_path: str) -> tuple[np.ndarray, GribMetadata]:
    """Read GRIB file data and metadata.
    
    For files with multiple messages (e.g., all ensemble members), reads
    all messages and stacks them along the ensemble dimension.
    
    Parameters
    ----------
    grib_path : str
        Path to the GRIB file.
    
    Returns
    -------
    tuple[np.ndarray, GribMetadata]
        Data array with shape (n_ensemble, n_lat, n_lon) and metadata.
    """
    if not HAS_ECCODES:
        raise ImportError("eccodes is required for GRIB reading")
    
    grib_path = str(grib_path)
    messages = []
    metadata = None
    
    with open(grib_path, 'rb') as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            
            try:
                # Get values
                values = eccodes.codes_get_values(gid)
                n_lat = eccodes.codes_get(gid, "Nj")
                n_lon = eccodes.codes_get(gid, "Ni")
                
                # Reshape to 2D grid
                data = values.reshape(n_lat, n_lon).astype(np.float32)
                messages.append(data)
                
                # Extract metadata from first message
                if metadata is None:
                    var_name = get_cf_varname_from_grib(gid)
                    data_date = eccodes.codes_get(gid, "dataDate")
                    data_time = eccodes.codes_get(gid, "dataTime")
                    step = eccodes.codes_get(gid, "step")
                    
                    year = data_date // 10000
                    month = (data_date % 10000) // 100
                    day = data_date % 100
                    hour = data_time // 100
                    minute = data_time % 100
                    reference_time = datetime(year, month, day, hour, minute)
                    valid_time = reference_time + timedelta(hours=int(step))
                    
                    metadata = GribMetadata(
                        variable=var_name,
                        forecast_hour=int(step),
                        reference_time=reference_time,
                        valid_time=valid_time,
                        shape=(n_lat, n_lon),
                    )
            finally:
                eccodes.codes_release(gid)
    
    if not messages:
        raise ValueError(f"No GRIB messages found in {grib_path}")
    
    # Stack messages along ensemble dimension
    data = np.stack(messages, axis=0)
    metadata.n_ensemble = len(messages)
    
    return data, metadata


def read_grib_with_cfgrib(grib_path: str) -> tuple[np.ndarray, GribMetadata]:
    """Read GRIB file using cfgrib/xarray (alternative to eccodes).
    
    Parameters
    ----------
    grib_path : str
        Path to the GRIB file.
    
    Returns
    -------
    tuple[np.ndarray, GribMetadata]
        Data array and metadata.
    """
    if not HAS_CFGRIB:
        raise ImportError("cfgrib is required. Install with: pip install cfgrib")
    
    ds = xr.open_dataset(grib_path, engine='cfgrib')
    
    # Get the data variable (should be only one)
    # cfgrib returns CF convention names directly (t2m, u10, v10, etc.)
    var_name = list(ds.data_vars)[0]
    data_var = ds[var_name]
    
    # Extract metadata
    metadata = GribMetadata(
        variable=var_name,
        forecast_hour=int(ds.step.values / np.timedelta64(1, 'h')) if 'step' in ds.coords else 0,
        reference_time=pd.Timestamp(ds.time.values).to_pydatetime() if 'time' in ds.coords else datetime.utcnow(),
        valid_time=pd.Timestamp(ds.valid_time.values).to_pydatetime() if 'valid_time' in ds.coords else datetime.utcnow(),
        shape=data_var.shape[-2:],  # Last two dims are lat/lon
    )
    
    # Get data, ensuring it has ensemble dimension
    data = data_var.values
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # Add ensemble dimension
    elif data.ndim == 3 and 'number' in data_var.dims:
        pass  # Already has ensemble dimension
    
    metadata.n_ensemble = data.shape[0]
    
    ds.close()
    return data.astype(np.float32), metadata


def write_grib_to_zarr_region(
    grib_path: str,
    zarr_store_path: str,
    hour_to_index: Optional[dict[int, int]] = None,
    expected_reference_time: Optional[datetime] = None,
    use_cfgrib: bool = False,
    stage_locally: bool = True,
    staging_dir: str = DEFAULT_STAGING_DIR,
    cleanup_staged: bool = True,
) -> WriteResult:
    """Write GRIB file data to a specific region in the Zarr store.
    
    This function reads a GRIB file, determines the appropriate time index
    from the forecast hour, and writes the data to that specific "slot"
    in the pre-allocated Zarr store using region writes.
    
    Parameters
    ----------
    grib_path : str
        Path to the GRIB file.
    zarr_store_path : str
        Path to the pre-allocated Zarr store.
    hour_to_index : dict[int, int], optional
        Mapping from forecast hour to array index. If None, builds from
        default forecast steps.
    expected_reference_time : datetime, optional
        Expected reference time. If provided, validates that the GRIB file
        matches this reference time.
    use_cfgrib : bool
        Use cfgrib instead of eccodes for reading (default: False).
    stage_locally : bool
        Copy file to local SSD before reading (default: True).
        Significantly speeds up reads from cloud storage.
    staging_dir : str
        Directory for local staging (default: /local_disk0/grib_staging).
    cleanup_staged : bool
        Remove staged file after processing (default: True).
    
    Returns
    -------
    WriteResult
        Result containing success status, timing, and metadata.
    
    Examples
    --------
    >>> hour_map = build_hour_to_index_map(generate_forecast_steps())
    >>> result = write_grib_to_zarr_region(
    ...     grib_path="forecast_t2m_step6.grib",
    ...     zarr_store_path="forecast.zarr",
    ...     hour_to_index=hour_map,
    ... )
    >>> print(f"Wrote to step index {result.step_index} in {result.elapsed_ms:.1f}ms")
    """
    import time
    start_time = time.perf_counter()
    timing = {}  # Track timing breakdown
    
    original_grib_path = str(grib_path)
    zarr_store_path = str(zarr_store_path)
    local_grib_path = None
    
    # Build default hour-to-index mapping if not provided
    if hour_to_index is None:
        forecast_steps = generate_forecast_steps()
        hour_to_index = build_hour_to_index_map(forecast_steps)
    
    try:
        # Stage file locally for faster reading
        t0 = time.perf_counter()
        if stage_locally:
            local_grib_path = stage_file_locally(original_grib_path, staging_dir)
            grib_path = local_grib_path
        else:
            grib_path = original_grib_path
        timing['stage_ms'] = (time.perf_counter() - t0) * 1000
        
        # Read GRIB data
        t0 = time.perf_counter()
        if use_cfgrib:
            data, metadata = read_grib_with_cfgrib(grib_path)
        else:
            data, metadata = read_grib_data(grib_path)
        timing['read_ms'] = (time.perf_counter() - t0) * 1000
        
        # Validate reference time if expected
        if expected_reference_time is not None:
            if metadata.reference_time != expected_reference_time:
                raise ValueError(
                    f"Reference time mismatch: expected {expected_reference_time}, "
                    f"got {metadata.reference_time}"
                )
        
        # Map forecast hour to step index
        if metadata.forecast_hour not in hour_to_index:
            raise ValueError(
                f"Forecast hour {metadata.forecast_hour} not in step mapping. "
                f"Valid hours: {sorted(hour_to_index.keys())[:10]}..."
            )
        step_index = hour_to_index[metadata.forecast_hour]
        
        # Open Zarr store
        t0 = time.perf_counter()
        ds = xr.open_zarr(zarr_store_path, consolidated=True)
        timing['zarr_open_ms'] = (time.perf_counter() - t0) * 1000
        
        # Verify variable exists
        var_name = metadata.variable
        if var_name not in ds.data_vars:
            raise ValueError(
                f"Variable '{var_name}' not found in Zarr store. "
                f"Available: {list(ds.data_vars)}"
            )
        
        # Verify shapes match
        expected_ensemble = ds.sizes['number']
        expected_lat = ds.sizes['latitude']
        expected_lon = ds.sizes['longitude']
        
        actual_ensemble, actual_lat, actual_lon = data.shape
        
        if actual_lat != expected_lat or actual_lon != expected_lon:
            raise ValueError(
                f"Spatial shape mismatch: expected ({expected_lat}, {expected_lon}), "
                f"got ({actual_lat}, {actual_lon})"
            )
        
        if actual_ensemble != expected_ensemble:
            logger.warning(
                f"Ensemble size mismatch: expected {expected_ensemble}, "
                f"got {actual_ensemble}. Will write partial data."
            )
        
        ds.close()
        
        # Create dataset for the single time step
        write_ds = xr.Dataset({
            var_name: (['step', 'number', 'latitude', 'longitude'], 
                      data[np.newaxis, :, :, :])  # Add step dimension
        })
        
        # Define the region to write
        region = {
            'step': slice(step_index, step_index + 1),
            'number': slice(0, actual_ensemble),
            'latitude': slice(None),
            'longitude': slice(None),
        }
        
        # Perform region write
        t0 = time.perf_counter()
        write_ds.to_zarr(
            zarr_store_path,
            mode='r+',
            region=region,
        )
        timing['zarr_write_ms'] = (time.perf_counter() - t0) * 1000
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Cleanup staged file
        t0 = time.perf_counter()
        if cleanup_staged and local_grib_path:
            cleanup_staged_file(local_grib_path, staging_dir)
        timing['cleanup_ms'] = (time.perf_counter() - t0) * 1000
        
        # Log timing breakdown (INFO level for visibility)
        logger.info(
            f"Timing for {Path(original_grib_path).name}: "
            f"stage={timing.get('stage_ms', 0):.0f}ms, "
            f"read={timing.get('read_ms', 0):.0f}ms, "
            f"zarr_open={timing.get('zarr_open_ms', 0):.0f}ms, "
            f"write={timing.get('zarr_write_ms', 0):.0f}ms, "
            f"cleanup={timing.get('cleanup_ms', 0):.0f}ms, "
            f"total={elapsed_ms:.0f}ms"
        )
        
        return WriteResult(
            success=True,
            grib_path=original_grib_path,
            variable=var_name,
            step_index=step_index,
            ensemble_slice=(0, actual_ensemble),
            elapsed_ms=elapsed_ms,
        )
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Failed to write {original_grib_path}: {e}")
        
        # Cleanup staged file even on error
        if cleanup_staged and local_grib_path:
            cleanup_staged_file(local_grib_path, staging_dir)
        
        return WriteResult(
            success=False,
            grib_path=original_grib_path,
            variable=getattr(metadata, 'variable', 'unknown') if 'metadata' in locals() else 'unknown',
            step_index=-1,
            ensemble_slice=(0, 0),
            elapsed_ms=elapsed_ms,
            error=str(e),
        )


def stage_files_batch(
    grib_paths: list[str],
    staging_dir: str = DEFAULT_STAGING_DIR,
    max_workers: int = 32,
) -> dict[str, str]:
    """Stage multiple files to local storage in parallel.
    
    Parameters
    ----------
    grib_paths : list[str]
        List of GRIB file paths.
    staging_dir : str
        Local directory to copy files to.
    max_workers : int
        Maximum number of parallel copy operations.
    
    Returns
    -------
    dict[str, str]
        Mapping from original path to local staged path.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    os.makedirs(staging_dir, exist_ok=True)
    staged_paths = {}
    
    def stage_one(path: str) -> tuple[str, str]:
        local_path = stage_file_locally(path, staging_dir)
        return (path, local_path)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(stage_one, p): p for p in grib_paths}
        for future in as_completed(futures):
            orig_path, local_path = future.result()
            staged_paths[orig_path] = local_path
    
    return staged_paths


def stage_files_with_azcopy(
    file_paths: list[str],
    staging_dir: str = DEFAULT_STAGING_DIR,
) -> dict[str, str]:
    """Stage files from cloud storage to local SSD using azcopy.
    
    Uses azcopy's --include-path option to download multiple files in a single
    command, which is much faster than individual downloads.
    
    Note: azcopy preserves directory structure, so files are placed in
    subdirectories matching their relative paths within the volume.
    
    Parameters
    ----------
    file_paths : list[str]
        List of file paths (e.g., /Volumes/catalog/schema/volume/path/file.grib2).
    staging_dir : str
        Local directory to copy files to.
    
    Returns
    -------
    dict[str, str]
        Mapping from original path to local staged path.
    """
    import subprocess
    import time
    from pathlib import Path as PathLib
    from .cloud_sync import TokenManager, ensure_azcopy
    
    if not file_paths:
        return {}
    
    os.makedirs(staging_dir, exist_ok=True)
    
    # Extract volume info from first path: /Volumes/catalog/schema/volume/...
    first_path = file_paths[0]
    parts = PathLib(first_path).parts
    
    if len(parts) < 5 or parts[1] != 'Volumes':
        raise ValueError(f"Expected /Volumes/catalog/schema/volume/... path, got: {first_path}")
    
    catalog, schema, volume = parts[2], parts[3], parts[4]
    volume_name = f"{catalog}.{schema}.{volume}"
    
    # Volume prefix: /Volumes/catalog/schema/volume
    volume_prefix = f"/Volumes/{catalog}/{schema}/{volume}"
    
    logger.info(f"Staging {len(file_paths)} files from volume {volume_name} using azcopy")
    
    try:
        # Get token and Azure URL for the volume
        token_manager = TokenManager(volume_name)
        
        # Get the storage location from Volume metadata
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        volume_info = w.volumes.read(volume_name)
        storage_location = volume_info.storage_location  # abfss://container@account.dfs.core.windows.net/path
        
        # Parse the storage URL to get container and account
        from urllib.parse import urlparse
        parsed = urlparse(storage_location)
        container, host_rest = parsed.netloc.split('@', 1)
        storage_account = host_rest.split('.')[0]
        volume_base_path = parsed.path.lstrip('/')
        
        logger.debug(f"Volume storage: container={container}, account={storage_account}, base={volume_base_path}")
        
        # Get SAS token
        sas_url = token_manager.get_sas_url()
        sas_token = sas_url.split('?', 1)[1] if '?' in sas_url else ''
        
        # Ensure azcopy is available
        azcopy_path = ensure_azcopy()
        
        start_time = time.perf_counter()
        
        # Build the --include-path argument with paths relative to the volume root
        # Per Microsoft docs: source should be a directory, include-path is relative to it
        # https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-blobs-download#specify-multiple-complete-blob-names
        relative_paths = []
        staged_paths = {}
        
        # azcopy includes the last component of the source URL path in destination
        # e.g., source .../lseg/netcdf -> creates netcdf/ subdirectory
        source_dir_name = volume_base_path.rstrip('/').split('/')[-1] if volume_base_path else ''
        
        for fpath in file_paths:
            # Get the relative path within the volume
            # /Volumes/stuart/lseg/netcdf/landing/file.grib2 -> landing/file.grib2
            rel_path = fpath[len(volume_prefix):].lstrip('/')
            relative_paths.append(rel_path)
            
            # azcopy creates: staging_dir/source_dir_name/rel_path
            # e.g., /local_disk0/grib_staging/netcdf/landing/file.grib2
            local_path = os.path.join(staging_dir, source_dir_name, rel_path)
            staged_paths[fpath] = local_path
        
        # Build the source URL pointing to the volume root directory (not just container)
        # e.g., https://oneenvadls.blob.core.windows.net/stuart/lseg/netcdf?SAS
        source_url = f"https://{storage_account}.blob.core.windows.net/{container}/{volume_base_path}?{sas_token}"
        
        # Build semicolon-separated list of relative paths
        include_path = ';'.join(relative_paths)
        
        logger.info(f"azcopy source: https://{storage_account}.blob.core.windows.net/{container}/{volume_base_path}")
        logger.info(f"azcopy include-path (first 3): {relative_paths[:3]}")
        
        # Run single azcopy command with --include-path
        cmd = [
            azcopy_path, 'copy',
            source_url,
            staging_dir,
            '--include-path', include_path,
            '--output-level', 'essential',  # Show transfer summary
        ]
        
        # Log full details for debugging (mask SAS token)
        source_url_masked = source_url.split('?')[0] + '?[SAS]'
        logger.info(f"azcopy command:")
        logger.info(f"  source: {source_url_masked}")
        logger.info(f"  destination: {staging_dir}")
        logger.info(f"  include-path ({len(relative_paths)} files): {include_path[:200]}{'...' if len(include_path) > 200 else ''}")
        
        azcopy_start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)
        azcopy_elapsed = (time.perf_counter() - azcopy_start) * 1000
        logger.info(f"azcopy subprocess wall time: {azcopy_elapsed:.0f}ms")
        
        # Log azcopy output
        if result.stdout:
            logger.info(f"azcopy stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"azcopy stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"azcopy failed with code {result.returncode}: {result.stderr}")
        
        total_elapsed = (time.perf_counter() - start_time) * 1000
        
        # Debug: List what's actually in staging_dir to find where files went
        logger.info(f"Listing staging directory contents:")
        for root, dirs, files in os.walk(staging_dir):
            rel_root = os.path.relpath(root, staging_dir)
            for f in files[:5]:  # Limit output
                logger.info(f"  Found: {os.path.join(rel_root, f)}")
            if len(files) > 5:
                logger.info(f"  ... and {len(files) - 5} more files in {rel_root}")
        
        # Verify files were downloaded
        missing = [fp for fp, lp in staged_paths.items() if not os.path.exists(lp)]
        if missing:
            logger.warning(f"azcopy completed but {len(missing)} files missing at expected paths")
            logger.warning(f"Expected path example: {staged_paths[missing[0]]}")
        
        logger.info(
            f"Staged {len(file_paths)} files in {total_elapsed:.0f}ms "
            f"({total_elapsed/len(file_paths):.0f}ms/file avg) via azcopy"
        )
        
        return staged_paths
        
    except Exception as e:
        logger.error(f"azcopy staging failed: {e}")
        raise


def stage_files_with_azure_sdk(
    file_paths: list[str],
    staging_dir: str = DEFAULT_STAGING_DIR,
    max_concurrency: int = 32,
) -> dict[str, str]:
    """Stage files from Azure Blob Storage to local SSD using Azure SDK.
    
    This is an alternative to azcopy that avoids the ~15s startup overhead
    by using persistent HTTP connections via the Azure SDK.
    
    Parameters
    ----------
    file_paths : list[str]
        List of file paths (e.g., /Volumes/catalog/schema/volume/path/file.grib2).
    staging_dir : str
        Local directory to copy files to.
    max_concurrency : int
        Maximum concurrent downloads (default: 32).
    
    Returns
    -------
    dict[str, str]
        Mapping from original path to local staged path.
    """
    import asyncio
    import time
    from pathlib import Path as PathLib
    from .cloud_sync import TokenManager
    
    if not file_paths:
        return {}
    
    os.makedirs(staging_dir, exist_ok=True)
    
    # Extract volume info from first path
    first_path = file_paths[0]
    parts = PathLib(first_path).parts
    
    if len(parts) < 5 or parts[1] != 'Volumes':
        raise ValueError(f"Expected /Volumes/catalog/schema/volume/... path, got: {first_path}")
    
    catalog, schema, volume = parts[2], parts[3], parts[4]
    volume_name = f"{catalog}.{schema}.{volume}"
    volume_prefix = f"/Volumes/{catalog}/{schema}/{volume}"
    
    logger.info(f"Staging {len(file_paths)} files from volume {volume_name} using Azure SDK")
    
    try:
        # Get token and storage info
        token_manager = TokenManager(volume_name)
        
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        volume_info = w.volumes.read(volume_name)
        storage_location = volume_info.storage_location
        
        from urllib.parse import urlparse
        parsed = urlparse(storage_location)
        container, host_rest = parsed.netloc.split('@', 1)
        storage_account = host_rest.split('.')[0]
        volume_base_path = parsed.path.lstrip('/')
        
        # Get SAS token
        sas_url = token_manager.get_sas_url()
        sas_token = sas_url.split('?', 1)[1] if '?' in sas_url else ''
        
        # Build container URL with SAS
        container_url = f"https://{storage_account}.blob.core.windows.net/{container}?{sas_token}"
        
        start_time = time.perf_counter()
        
        # Build list of (blob_path, local_path) tuples
        downloads = []
        staged_paths = {}
        
        for fpath in file_paths:
            rel_path = fpath[len(volume_prefix):].lstrip('/')
            if volume_base_path:
                blob_path = f"{volume_base_path}/{rel_path}"
            else:
                blob_path = rel_path
            
            # Match azcopy behavior: include volume dir name in local path
            source_dir_name = volume_base_path.rstrip('/').split('/')[-1] if volume_base_path else ''
            local_path = os.path.join(staging_dir, source_dir_name, rel_path)
            
            downloads.append((blob_path, local_path))
            staged_paths[fpath] = local_path
        
        # Run async downloads
        async def download_all():
            from azure.storage.blob.aio import ContainerClient
            
            async with ContainerClient.from_container_url(container_url) as client:
                semaphore = asyncio.Semaphore(max_concurrency)
                
                async def download_one(blob_path: str, local_path: str):
                    async with semaphore:
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        blob_client = client.get_blob_client(blob_path)
                        
                        # Download to file
                        with open(local_path, 'wb') as f:
                            stream = await blob_client.download_blob()
                            data = await stream.readall()
                            f.write(data)
                
                tasks = [download_one(bp, lp) for bp, lp in downloads]
                await asyncio.gather(*tasks)
        
        # Run the async event loop
        # Handle case where we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, need to run in executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, download_all())
                future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            asyncio.run(download_all())
        
        total_elapsed = (time.perf_counter() - start_time) * 1000
        
        # Verify files were downloaded
        missing = [fp for fp, lp in staged_paths.items() if not os.path.exists(lp)]
        if missing:
            logger.warning(f"Azure SDK download completed but {len(missing)} files missing")
            logger.warning(f"Expected path example: {staged_paths[missing[0]]}")
        
        logger.info(
            f"Staged {len(file_paths)} files in {total_elapsed:.0f}ms "
            f"({total_elapsed/len(file_paths):.0f}ms/file avg) via Azure SDK"
        )
        
        return staged_paths
        
    except Exception as e:
        logger.error(f"Azure SDK staging failed: {e}")
        raise


def write_grib_to_zarr_direct(
    grib_path: str,
    zarr_arrays: dict,
    hour_to_index: dict[int, int],
    use_cfgrib: bool = False,
) -> WriteResult:
    """Write GRIB data directly to zarr arrays (no xarray overhead).
    
    This is much faster than write_grib_to_zarr_region because:
    - No xarray Dataset creation
    - No metadata parsing per write
    - Direct numpy array assignment
    - No file locking for non-overlapping regions
    
    Parameters
    ----------
    grib_path : str
        Path to the GRIB file (should be local for speed).
    zarr_arrays : dict
        Dict mapping variable names to open zarr arrays.
        Get this from open_zarr_arrays().
    hour_to_index : dict[int, int]
        Mapping from forecast hour to array index.
    use_cfgrib : bool
        Use cfgrib instead of eccodes for reading.
    
    Returns
    -------
    WriteResult
        Result containing success status and timing.
    """
    import time
    start_time = time.perf_counter()
    original_path = str(grib_path)
    
    try:
        # Read GRIB data
        t0 = time.perf_counter()
        if use_cfgrib:
            data, metadata = read_grib_with_cfgrib(grib_path)
        else:
            data, metadata = read_grib_data(grib_path)
        read_ms = (time.perf_counter() - t0) * 1000
        
        # Map forecast hour to step index
        if metadata.forecast_hour not in hour_to_index:
            raise ValueError(
                f"Forecast hour {metadata.forecast_hour} not in step mapping"
            )
        step_index = hour_to_index[metadata.forecast_hour]
        
        # Get the zarr array for this variable
        var_name = metadata.variable
        if var_name not in zarr_arrays:
            raise ValueError(
                f"Variable '{var_name}' not found. Available: {list(zarr_arrays.keys())}"
            )
        zarr_array = zarr_arrays[var_name]
        
        # Direct write - no locking needed for non-overlapping regions!
        t0 = time.perf_counter()
        actual_ensemble = data.shape[0]
        zarr_array[step_index, :actual_ensemble, :, :] = data
        write_ms = (time.perf_counter() - t0) * 1000
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Direct write {Path(original_path).name}: "
            f"read={read_ms:.0f}ms, write={write_ms:.0f}ms, total={elapsed_ms:.0f}ms"
        )
        
        return WriteResult(
            success=True,
            grib_path=original_path,
            variable=var_name,
            step_index=step_index,
            ensemble_slice=(0, actual_ensemble),
            elapsed_ms=elapsed_ms,
        )
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Direct write failed for {original_path}: {e}")
        return WriteResult(
            success=False,
            grib_path=original_path,
            variable='unknown',
            step_index=-1,
            ensemble_slice=(0, 0),
            elapsed_ms=elapsed_ms,
            error=str(e),
        )


def open_zarr_arrays(zarr_store_path: str, variables: list[str] = None) -> dict:
    """Open zarr arrays for direct writing (bypassing xarray).
    
    Call this once per batch, then pass to write_grib_to_zarr_direct().
    
    Parameters
    ----------
    zarr_store_path : str
        Path to the zarr store.
    variables : list[str], optional
        Variables to open. If None, opens all data variables.
    
    Returns
    -------
    dict
        Dict mapping variable names to zarr arrays.
    """
    store = zarr.open(zarr_store_path, mode='r+')
    
    if variables is None:
        # Get all array names (excluding coordinates)
        variables = [k for k in store.array_keys() 
                    if k not in ('step', 'number', 'latitude', 'longitude', 'time', 'valid_time')]
    
    arrays = {}
    for var in variables:
        if var in store:
            arrays[var] = store[var]
            logger.debug(f"Opened zarr array: {var} with shape {store[var].shape}")
    
    return arrays


def cleanup_staging_dir(staging_dir: str = DEFAULT_STAGING_DIR) -> None:
    """Remove all files from the staging directory.
    
    Parameters
    ----------
    staging_dir : str
        The staging directory to clean up.
    """
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)


def write_grib_batch_parallel(
    grib_paths: list[str],
    zarr_store_path: str,
    hour_to_index: Optional[dict[int, int]] = None,
    max_workers: int = 32,
    use_cfgrib: bool = False,
    stage_locally: bool = True,
    staging_dir: str = DEFAULT_STAGING_DIR,
    batch_stage: bool = True,
) -> list[WriteResult]:
    """Write multiple GRIB files to Zarr in parallel.
    
    Uses ThreadPoolExecutor to saturate I/O bandwidth. Optionally stages
    files to local SSD first for faster reads from cloud storage.
    
    Parameters
    ----------
    grib_paths : list[str]
        List of GRIB file paths.
    zarr_store_path : str
        Path to the pre-allocated Zarr store.
    hour_to_index : dict[int, int], optional
        Mapping from forecast hour to array index.
    max_workers : int
        Maximum number of parallel workers (default: 32).
    use_cfgrib : bool
        Use cfgrib instead of eccodes for reading.
    stage_locally : bool
        Copy files to local SSD before reading (default: True).
    staging_dir : str
        Directory for local staging (default: /local_disk0/grib_staging).
    batch_stage : bool
        If True, stage all files first then process. If False, stage
        and process each file individually (default: True).
    
    Returns
    -------
    list[WriteResult]
        List of write results for each file.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    if hour_to_index is None:
        forecast_steps = generate_forecast_steps()
        hour_to_index = build_hour_to_index_map(forecast_steps)
    
    results = []
    start_time = time.perf_counter()
    staged_paths = {}
    
    try:
        # Stage all files first if batch_stage is enabled
        if stage_locally and batch_stage:
            logger.info(f"Staging {len(grib_paths)} files to {staging_dir}...")
            stage_start = time.perf_counter()
            staged_paths = stage_files_batch(grib_paths, staging_dir, max_workers)
            stage_elapsed = (time.perf_counter() - stage_start) * 1000
            logger.info(f"Staging complete in {stage_elapsed:.1f}ms")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for grib_path in grib_paths:
                # Use staged path if available
                process_path = staged_paths.get(grib_path, grib_path)
                
                # Disable per-file staging if we already batch-staged
                per_file_stage = stage_locally and not batch_stage
                
                future = executor.submit(
                    write_grib_to_zarr_region,
                    process_path if batch_stage else grib_path,
                    zarr_store_path,
                    hour_to_index,
                    None,  # expected_reference_time
                    use_cfgrib,
                    per_file_stage,  # stage_locally
                    staging_dir,
                    not batch_stage,  # cleanup_staged (don't cleanup if batch-staged)
                )
                futures[future] = grib_path  # Store original path
            
            for future in as_completed(futures):
                original_path = futures[future]
                try:
                    result = future.result()
                    # Fix the path in result to be original path
                    result = WriteResult(
                        success=result.success,
                        grib_path=original_path,
                        variable=result.variable,
                        step_index=result.step_index,
                        ensemble_slice=result.ensemble_slice,
                        elapsed_ms=result.elapsed_ms,
                        error=result.error,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Unexpected error processing {original_path}: {e}")
                    results.append(WriteResult(
                        success=False,
                        grib_path=original_path,
                        variable='unknown',
                        step_index=-1,
                        ensemble_slice=(0, 0),
                        elapsed_ms=0,
                        error=str(e),
                    ))
        
    finally:
        # Cleanup staging directory if we batch-staged
        if stage_locally and batch_stage and staged_paths:
            cleanup_staging_dir(staging_dir)
    
    elapsed_total = (time.perf_counter() - start_time) * 1000
    successful = sum(1 for r in results if r.success)
    
    logger.info(
        f"Batch write complete: {successful}/{len(results)} successful "
        f"in {elapsed_total:.1f}ms total"
    )
    
    return results


if __name__ == "__main__":
    # Demo usage
    import tempfile
    from datetime import datetime
    from .zarr_init import initialize_zarr_store
    
    logging.basicConfig(level=logging.INFO)
    
    # Show forecast step mapping
    steps = generate_forecast_steps()
    hour_map = build_hour_to_index_map(steps)
    
    print("Forecast hour to index mapping (sample):")
    sample_hours = [0, 1, 6, 12, 24, 48, 90, 93, 96, 144, 150, 180, 360]
    for h in sample_hours:
        if h in hour_map:
            print(f"  Hour {h:3d} -> Index {hour_map[h]:3d}")
        else:
            print(f"  Hour {h:3d} -> NOT IN MAPPING")

