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
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr
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
            # Extract variable name
            short_name = eccodes.codes_get(gid, "shortName")
            
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
                variable=short_name,
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
                    short_name = eccodes.codes_get(gid, "shortName")
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
                        variable=short_name,
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
    
    grib_path = str(grib_path)
    zarr_store_path = str(zarr_store_path)
    
    # Build default hour-to-index mapping if not provided
    if hour_to_index is None:
        forecast_steps = generate_forecast_steps()
        hour_to_index = build_hour_to_index_map(forecast_steps)
    
    try:
        # Read GRIB data
        if use_cfgrib:
            data, metadata = read_grib_with_cfgrib(grib_path)
        else:
            data, metadata = read_grib_data(grib_path)
        
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
        ds = xr.open_zarr(zarr_store_path, consolidated=True)
        
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
        write_ds.to_zarr(
            zarr_store_path,
            mode='r+',
            region=region,
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return WriteResult(
            success=True,
            grib_path=grib_path,
            variable=var_name,
            step_index=step_index,
            ensemble_slice=(0, actual_ensemble),
            elapsed_ms=elapsed_ms,
        )
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Failed to write {grib_path}: {e}")
        
        return WriteResult(
            success=False,
            grib_path=grib_path,
            variable=getattr(metadata, 'variable', 'unknown') if 'metadata' in locals() else 'unknown',
            step_index=-1,
            ensemble_slice=(0, 0),
            elapsed_ms=elapsed_ms,
            error=str(e),
        )


def write_grib_batch_parallel(
    grib_paths: list[str],
    zarr_store_path: str,
    hour_to_index: Optional[dict[int, int]] = None,
    max_workers: int = 32,
    use_cfgrib: bool = False,
) -> list[WriteResult]:
    """Write multiple GRIB files to Zarr in parallel.
    
    Uses ThreadPoolExecutor to saturate I/O bandwidth.
    
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
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                write_grib_to_zarr_region,
                grib_path,
                zarr_store_path,
                hour_to_index,
                None,
                use_cfgrib,
            ): grib_path
            for grib_path in grib_paths
        }
        
        for future in as_completed(futures):
            grib_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Unexpected error processing {grib_path}: {e}")
                results.append(WriteResult(
                    success=False,
                    grib_path=grib_path,
                    variable='unknown',
                    step_index=-1,
                    ensemble_slice=(0, 0),
                    elapsed_ms=0,
                    error=str(e),
                ))
    
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

