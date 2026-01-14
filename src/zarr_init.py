#!/usr/bin/env python3
"""
Zarr Store Initialization Module.

Creates pre-allocated Zarr stores with NaN values for low-latency weather data ingestion.
The store structure is designed to match the chunking strategy specified in the architecture:
- Variable: 1 (each variable in separate array)
- Step: 1 (each forecast step is a separate chunk)
- Ensemble: 50 (all ensemble members in one chunk)
- Lat/Lon: Full (entire spatial field contiguous)
"""

import numpy as np
import xarray as xr
import zarr
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ForecastStepConfig:
    """Configuration for non-uniform forecast step generation.
    
    Default configuration matches ECMWF ENS format:
    - Hours 0-90: Hourly (91 steps)
    - Hours 93-144: 3-hourly (18 steps)
    - Hours 150-360: 6-hourly (36 steps)
    Total: 145 steps
    """
    hourly_end: int = 90
    three_hourly_start: int = 93
    three_hourly_end: int = 144
    six_hourly_start: int = 150
    six_hourly_end: int = 360
    
    def generate_steps(self) -> list[int]:
        """Generate the list of forecast step hours."""
        steps = []
        # Hourly: 0 to hourly_end
        steps.extend(range(0, self.hourly_end + 1))
        # 3-hourly
        steps.extend(range(self.three_hourly_start, self.three_hourly_end + 1, 3))
        # 6-hourly
        steps.extend(range(self.six_hourly_start, self.six_hourly_end + 1, 6))
        return steps


def generate_forecast_steps(config: Optional[ForecastStepConfig] = None) -> list[int]:
    """Generate the 145 non-uniform forecast steps (in hours).
    
    Parameters
    ----------
    config : ForecastStepConfig, optional
        Configuration for step generation. Uses default ECMWF ENS config if None.
    
    Returns
    -------
    list[int]
        List of forecast step hours (145 steps by default).
    
    Examples
    --------
    >>> steps = generate_forecast_steps()
    >>> len(steps)
    145
    >>> steps[:5]
    [0, 1, 2, 3, 4]
    >>> steps[90:95]
    [90, 93, 96, 99, 102]
    """
    if config is None:
        config = ForecastStepConfig()
    return config.generate_steps()


def build_hour_to_index_map(forecast_steps: list[int]) -> dict[int, int]:
    """Build lookup from forecast hour to array index.
    
    Parameters
    ----------
    forecast_steps : list[int]
        List of forecast step hours.
    
    Returns
    -------
    dict[int, int]
        Mapping from forecast hour to array index.
    
    Examples
    --------
    >>> steps = generate_forecast_steps()
    >>> hour_map = build_hour_to_index_map(steps)
    >>> hour_map[0]
    0
    >>> hour_map[90]
    90
    >>> hour_map[93]
    91
    """
    return {hour: idx for idx, hour in enumerate(forecast_steps)}


def _create_time_coordinates(
    reference_time: datetime,
    forecast_steps: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create time coordinates for the Zarr store.
    
    Parameters
    ----------
    reference_time : datetime
        The forecast reference time (base time).
    forecast_steps : list[int]
        List of forecast step hours.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (step_hours, valid_times) arrays.
    """
    step_hours = np.array(forecast_steps, dtype=np.int32)
    valid_times = np.array([
        reference_time + timedelta(hours=int(h))
        for h in forecast_steps
    ], dtype='datetime64[ns]')
    return step_hours, valid_times


def _create_spatial_coordinates(
    lat_size: int,
    lon_size: int,
    lat_start: float = 90.0,
    lat_end: float = -90.0,
    lon_start: float = 0.0,
    lon_end: float = 359.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create latitude and longitude coordinates.
    
    Parameters
    ----------
    lat_size : int
        Number of latitude points.
    lon_size : int
        Number of longitude points.
    lat_start : float
        Starting latitude (default: 90.0 for N pole).
    lat_end : float
        Ending latitude (default: -90.0 for S pole).
    lon_start : float
        Starting longitude (default: 0.0).
    lon_end : float
        Ending longitude (default: 359.5 for 0.5° global grid).
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (latitude, longitude) arrays.
    """
    latitude = np.linspace(lat_start, lat_end, lat_size, dtype=np.float32)
    longitude = np.linspace(lon_start, lon_end, lon_size, dtype=np.float32)
    return latitude, longitude


def initialize_zarr_store(
    output_path: str,
    variables: list[str],
    ensemble_members: int = 50,
    lat_size: int = 361,
    lon_size: int = 720,
    reference_time: Optional[datetime] = None,
    forecast_steps: Optional[list[int]] = None,
    lat_range: tuple[float, float] = (90.0, -90.0),
    lon_range: tuple[float, float] = (0.0, 359.5),
    fill_value: float = np.nan,
    dtype: np.dtype = np.float32,
    consolidate_metadata: bool = True,
) -> zarr.Group:
    """Initialize a pre-allocated Zarr store with NaN values.
    
    Creates the full directory structure for a forecast cycle with all chunks
    initialized to NaN (nodata) values. The chunking strategy follows the
    architecture specification:
    - Variable: 1 (each variable is a separate array)
    - Step: 1 (each forecast step is a separate chunk along time)
    - Ensemble: 50 (all ensemble members in one chunk)
    - Lat/Lon: Full (entire spatial field is contiguous)
    
    Parameters
    ----------
    output_path : str
        Path where the Zarr store will be created. Can be local or cloud path.
    variables : list[str]
        List of variable names to create (e.g., ['t2m', 'u10', 'v10']).
    ensemble_members : int
        Number of ensemble members (default: 50).
    lat_size : int
        Number of latitude points (default: 361 for 0.5° global grid).
    lon_size : int
        Number of longitude points (default: 720 for 0.5° global grid).
    reference_time : datetime, optional
        Forecast reference time. Defaults to current UTC time rounded to nearest 6 hours.
    forecast_steps : list[int], optional
        List of forecast step hours. Defaults to 145 non-uniform steps.
    lat_range : tuple[float, float]
        Latitude range as (start, end). Default: (90.0, -90.0) for N to S.
    lon_range : tuple[float, float]
        Longitude range as (start, end). Default: (0.0, 359.5).
    fill_value : float
        Fill value for unwritten data (default: np.nan).
    dtype : np.dtype
        Data type for variable arrays (default: np.float32).
    consolidate_metadata : bool
        Whether to consolidate Zarr metadata (default: True).
    
    Returns
    -------
    zarr.Group
        The initialized Zarr store root group.
    
    Examples
    --------
    >>> store = initialize_zarr_store(
    ...     output_path="/tmp/forecast.zarr",
    ...     variables=["t2m", "u10", "v10"],
    ...     reference_time=datetime(2024, 1, 1, 0, 0, 0),
    ... )
    >>> store.tree()
    /
     ├── t2m (145, 50, 361, 720) float32
     ├── u10 (145, 50, 361, 720) float32
     └── v10 (145, 50, 361, 720) float32
    """
    output_path = Path(output_path)
    
    # Set defaults
    if reference_time is None:
        # Round to nearest 6-hour cycle
        now = datetime.now(timezone.utc)
        hour = (now.hour // 6) * 6
        reference_time = now.replace(hour=hour, minute=0, second=0, microsecond=0, tzinfo=None)
    
    if forecast_steps is None:
        forecast_steps = generate_forecast_steps()
    
    n_steps = len(forecast_steps)
    logger.info(f"Initializing Zarr store at {output_path}")
    logger.info(f"  Variables: {variables}")
    logger.info(f"  Dimensions: (step={n_steps}, number={ensemble_members}, lat={lat_size}, lon={lon_size})")
    logger.info(f"  Reference time: {reference_time}")
    
    # Create coordinates
    step_hours, valid_times = _create_time_coordinates(reference_time, forecast_steps)
    latitude, longitude = _create_spatial_coordinates(
        lat_size, lon_size,
        lat_start=lat_range[0], lat_end=lat_range[1],
        lon_start=lon_range[0], lon_end=lon_range[1],
    )
    ensemble_numbers = np.arange(ensemble_members, dtype=np.int32)
    
    # Define chunking: (1 step, all ensemble, full lat, full lon)
    # This ensures 1 GRIB file = 1 Zarr chunk
    chunks = (1, ensemble_members, lat_size, lon_size)
    
    # Create coordinate arrays for xarray
    # Note: Using 'h' (hour) as units instead of 'hours' to avoid xarray's
    # automatic timedelta decoding which triggers FutureWarnings
    coords = {
        'step': ('step', step_hours, {'long_name': 'forecast step', 'units': 'h'}),
        'valid_time': ('step', valid_times, {'long_name': 'forecast valid time'}),
        'number': ('number', ensemble_numbers, {'long_name': 'ensemble member number'}),
        'latitude': ('latitude', latitude, {'long_name': 'latitude', 'units': 'degrees_north'}),
        'longitude': ('longitude', longitude, {'long_name': 'longitude', 'units': 'degrees_east'}),
    }
    
    # Create data variables filled with NaN
    data_vars = {}
    for var_name in variables:
        logger.info(f"  Creating variable: {var_name}")
        # Create NaN-filled array
        data = np.full((n_steps, ensemble_members, lat_size, lon_size), fill_value, dtype=dtype)
        data_vars[var_name] = (
            ['step', 'number', 'latitude', 'longitude'],
            data,
            {
                'long_name': var_name,
                '_FillValue': fill_value,
            }
        )
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            'Conventions': 'CF-1.8',
            'institution': 'Weather Data Pipeline',
            'source': 'Pre-allocated forecast store',
            'reference_time': str(reference_time),
            'history': f'Created {datetime.now(timezone.utc).isoformat()}',
        }
    )
    
    # Define encoding with chunking strategy
    encoding = {}
    for var_name in variables:
        encoding[var_name] = {
            'chunks': chunks,
            'dtype': dtype,
        }
    
    # Encode valid_time for serialization
    encoding['valid_time'] = {
        'units': 'hours since 1970-01-01',
        'calendar': 'proleptic_gregorian',
    }
    
    # Write to Zarr
    # Suppress the consolidated metadata warning for Zarr v3 (it's intentional)
    logger.info(f"  Writing to {output_path}...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Consolidated metadata.*')
        ds.to_zarr(
            output_path,
            mode='w',
            encoding=encoding,
            consolidated=consolidate_metadata,
        )
    
    # Open and return the store
    store = zarr.open_group(str(output_path), mode='r+')
    
    logger.info(f"  Zarr store initialized successfully")
    logger.info(f"  Total chunks per variable: {n_steps}")
    
    return store


def get_zarr_store_info(store_path: str) -> dict:
    """Get information about an existing Zarr store.
    
    Parameters
    ----------
    store_path : str
        Path to the Zarr store.
    
    Returns
    -------
    dict
        Dictionary containing store metadata and structure.
    """
    ds = xr.open_zarr(store_path, consolidated=True)
    
    info = {
        'path': store_path,
        'dimensions': dict(ds.sizes),
        'variables': list(ds.data_vars),
        'coordinates': list(ds.coords),
        'attrs': dict(ds.attrs),
        'chunks': {},
        'nan_counts': {},
    }
    
    for var in ds.data_vars:
        info['chunks'][var] = ds[var].encoding.get('chunks', 'unknown')
        # Count NaN values (sample only for performance)
        sample = ds[var].isel(step=0, number=0).values
        info['nan_counts'][var] = {
            'sample_nan_percentage': float(np.isnan(sample).mean() * 100),
        }
    
    ds.close()
    return info


if __name__ == "__main__":
    # Demo usage
    import argparse
    import os
    import tempfile
    import time
    
    parser = argparse.ArgumentParser(
        description="Initialize a pre-allocated Zarr store for weather data."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for Zarr store. If not specified, uses a temporary directory. "
             "Can also be set via ZARR_OUTPUT_PATH environment variable."
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the output in a temp directory (prints path for inspection). "
             "Can also be set via ZARR_KEEP_OUTPUT=1 environment variable."
    )
    args = parser.parse_args()
    
    # Support environment variables (useful for dbconnect execution)
    # CLI args take precedence over env vars
    output_from_env = os.environ.get("ZARR_OUTPUT_PATH")
    keep_from_env = os.environ.get("ZARR_KEEP_OUTPUT", "").lower() in ("1", "true", "yes")
    
    output_arg = args.output or output_from_env
    keep_arg = args.keep or keep_from_env
    
    logging.basicConfig(level=logging.INFO)
    
    # Determine output path
    temp_dir = None
    if output_arg:
        output_path = Path(output_arg)
    elif keep_arg:
        temp_dir = tempfile.mkdtemp(prefix="zarr_benchmark_")
        output_path = Path(temp_dir) / "forecast.zarr"
        print(f"Output will be kept at: {output_path}")
    else:
        temp_dir = tempfile.mkdtemp(prefix="zarr_benchmark_")
        output_path = Path(temp_dir) / "forecast.zarr"
    
    print("Initializing Zarr store...")
    start = time.time()
    
    store = initialize_zarr_store(
        output_path=str(output_path),
        variables=["t2m", "u10", "v10"],
        ensemble_members=50,
        lat_size=361,
        lon_size=720,
        reference_time=datetime(2024, 1, 1, 0, 0, 0),
    )
    
    elapsed = time.time() - start
    print(f"\nInitialization completed in {elapsed:.2f} seconds")
    
    # Show store info
    info = get_zarr_store_info(str(output_path))
    print(f"\nStore info:")
    print(f"  Dimensions: {info['dimensions']}")
    print(f"  Variables: {info['variables']}")
    print(f"  Chunks: {info['chunks']}")
    
    # Show forecast steps
    steps = generate_forecast_steps()
    print(f"\nForecast steps ({len(steps)} total):")
    print(f"  First 10: {steps[:10]}")
    print(f"  Around transition (90-99): {[s for s in steps if 88 <= s <= 100]}")
    print(f"  Last 10: {steps[-10:]}")
    
    # Clean up temp directory unless --keep or -o was specified
    if temp_dir and not keep_arg and not output_arg:
        import shutil
        shutil.rmtree(temp_dir)
    elif keep_arg or output_arg:
        print(f"\n✓ Output saved to: {output_path}")

