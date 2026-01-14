#!/usr/bin/env python3
"""
Generate mock GRIB or NetCDF files for testing the low-latency weather data pipeline.

Creates files with:
- Non-uniform forecast steps (145 total covering 360 hours)
- Multiple variables (t2m, u10, v10, sp)
- 50 ensemble members per file
- One file per (variable, step) combination

This matches the chunking strategy: Variable=1, Step=1, Ensemble=50, Lat/Lon=Full
"""

import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import argparse


# =============================================================================
# Forecast Step Configuration (matches src/zarr_init.py)
# =============================================================================

@dataclass
class ForecastStepConfig:
    """Configuration for a range of forecast steps."""
    start_hour: int
    end_hour: int
    interval: int
    
    def generate_hours(self) -> List[int]:
        """Generate list of forecast hours for this range."""
        return list(range(self.start_hour, self.end_hour + 1, self.interval))


# Non-uniform forecast step structure:
# Hours 0-90: Hourly (91 steps)
# Hours 93-144: 3-hourly (18 steps)  
# Hours 150-360: 6-hourly (36 steps)
# Total: 145 steps
FORECAST_STEP_CONFIGS = [
    ForecastStepConfig(start_hour=0, end_hour=90, interval=1),
    ForecastStepConfig(start_hour=93, end_hour=144, interval=3),
    ForecastStepConfig(start_hour=150, end_hour=360, interval=6),
]


def generate_forecast_steps() -> List[int]:
    """Generate the full list of 145 non-uniform forecast step hours."""
    steps = []
    for config in FORECAST_STEP_CONFIGS:
        steps.extend(config.generate_hours())
    return steps


# =============================================================================
# Variable Configuration
# =============================================================================

@dataclass
class VariableConfig:
    """GRIB encoding parameters for a meteorological variable."""
    name: str              # Our variable name (e.g., "t2m")
    short_name: str        # GRIB shortName (e.g., "2t")
    discipline: int        # GRIB discipline
    param_category: int    # GRIB parameterCategory
    param_number: int      # GRIB parameterNumber
    level_type: int        # GRIB typeOfFirstFixedSurface
    level_value: int       # Height in meters
    units: str             # Physical units
    min_value: float       # Realistic minimum value
    max_value: float       # Realistic maximum value


# Variable definitions matching our Zarr archive
VARIABLE_CONFIGS = {
    "t2m": VariableConfig(
        name="t2m",
        short_name="2t",
        discipline=0,           # Meteorological
        param_category=0,       # Temperature
        param_number=0,         # Temperature
        level_type=103,         # Height above ground
        level_value=2,          # 2 metres
        units="K",
        min_value=220.0,        # ~-53°C
        max_value=320.0,        # ~47°C
    ),
    "u10": VariableConfig(
        name="u10",
        short_name="10u",
        discipline=0,           # Meteorological
        param_category=2,       # Momentum
        param_number=2,         # U-component of wind
        level_type=103,         # Height above ground
        level_value=10,         # 10 metres
        units="m s-1",
        min_value=-50.0,
        max_value=50.0,
    ),
    "v10": VariableConfig(
        name="v10",
        short_name="10v",
        discipline=0,           # Meteorological
        param_category=2,       # Momentum
        param_number=3,         # V-component of wind
        level_type=103,         # Height above ground
        level_value=10,         # 10 metres
        units="m s-1",
        min_value=-50.0,
        max_value=50.0,
    ),
    "sp": VariableConfig(
        name="sp",
        short_name="sp",
        discipline=0,           # Meteorological
        param_category=3,       # Mass
        param_number=0,         # Pressure
        level_type=1,           # Ground or water surface
        level_value=0,          # Surface
        units="Pa",
        min_value=87000.0,      # ~870 hPa (low pressure)
        max_value=108000.0,     # ~1080 hPa (high pressure)
    ),
}


# =============================================================================
# GRIB File Writer
# =============================================================================

def write_grib_file(
    output_path: Path,
    data: np.ndarray,
    variable: VariableConfig,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    reference_time: datetime,
    forecast_hour: int,
    ensemble_members: np.ndarray,
) -> None:
    """
    Write data to a GRIB2 file using eccodes.
    
    Parameters
    ----------
    output_path : Path
        Path where the GRIB file will be saved
    data : np.ndarray
        3D array of shape (num_ensemble, num_lat, num_lon)
    variable : VariableConfig
        Variable configuration with GRIB encoding parameters
    latitudes : np.ndarray
        1D array of latitude values (must be regularly spaced)
    longitudes : np.ndarray
        1D array of longitude values (must be regularly spaced)
    reference_time : datetime
        Forecast reference/analysis time
    forecast_hour : int
        Forecast lead time in hours
    ensemble_members : np.ndarray
        Array of ensemble member numbers (0-indexed)
    """
    try:
        import eccodes
    except ImportError:
        raise ImportError(
            "eccodes is required for GRIB output. Install it with: pip install eccodes\n"
            "You may also need to install the eccodes library system package:\n"
            "  - macOS: brew install eccodes\n"
            "  - Ubuntu/Debian: apt-get install libeccodes-dev\n"
            "  - conda: conda install -c conda-forge eccodes"
        )
    
    n_ensemble, n_lat, n_lon = data.shape
    
    # Calculate grid parameters
    lat_start = float(latitudes[0])
    lat_end = float(latitudes[-1])
    lon_start = float(longitudes[0])
    lon_end = float(longitudes[-1])
    
    # Calculate grid increments
    lat_increment = abs(float(latitudes[1] - latitudes[0])) if n_lat > 1 else 1.0
    lon_increment = abs(float(longitudes[1] - longitudes[0])) if n_lon > 1 else 1.0
    
    # Determine scan direction
    lat_descending = lat_start > lat_end  # True if latitudes go from N to S
    
    with open(output_path, 'wb') as fout:
        for ens_idx, ens_num in enumerate(ensemble_members):
            # Create a new GRIB2 message from sample
            gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            
            try:
                # Set the parameter
                eccodes.codes_set(gid, "discipline", variable.discipline)
                eccodes.codes_set(gid, "parameterCategory", variable.param_category)
                eccodes.codes_set(gid, "parameterNumber", variable.param_number)
                eccodes.codes_set(gid, "shortName", variable.short_name)
                
                # Set level type
                eccodes.codes_set(gid, "typeOfFirstFixedSurface", variable.level_type)
                eccodes.codes_set(gid, "scaleFactorOfFirstFixedSurface", 0)
                eccodes.codes_set(gid, "scaledValueOfFirstFixedSurface", variable.level_value)
                
                # Set grid definition
                eccodes.codes_set(gid, "gridType", "regular_ll")
                eccodes.codes_set(gid, "Ni", n_lon)
                eccodes.codes_set(gid, "Nj", n_lat)
                
                # Set lat/lon bounds
                if lat_descending:
                    eccodes.codes_set(gid, "latitudeOfFirstGridPointInDegrees", lat_start)
                    eccodes.codes_set(gid, "latitudeOfLastGridPointInDegrees", lat_end)
                else:
                    eccodes.codes_set(gid, "latitudeOfFirstGridPointInDegrees", lat_end)
                    eccodes.codes_set(gid, "latitudeOfLastGridPointInDegrees", lat_start)
                
                eccodes.codes_set(gid, "longitudeOfFirstGridPointInDegrees", lon_start)
                eccodes.codes_set(gid, "longitudeOfLastGridPointInDegrees", lon_end)
                
                # Set grid increments
                eccodes.codes_set(gid, "iDirectionIncrementInDegrees", lon_increment)
                eccodes.codes_set(gid, "jDirectionIncrementInDegrees", lat_increment)
                
                # Set scanning mode
                eccodes.codes_set(gid, "jScansPositively", 0 if lat_descending else 1)
                
                # Set reference time (analysis/base time)
                eccodes.codes_set(gid, "dataDate", int(reference_time.strftime("%Y%m%d")))
                eccodes.codes_set(gid, "dataTime", int(reference_time.strftime("%H%M")))
                
                # Set forecast time
                eccodes.codes_set(gid, "stepUnits", 1)  # Hours
                eccodes.codes_set(gid, "forecastTime", forecast_hour)
                eccodes.codes_set(gid, "stepRange", forecast_hour)
                
                # Set Product Definition Template for ensemble data (PDT 4.1)
                eccodes.codes_set(gid, "productDefinitionTemplateNumber", 1)
                
                # Set ensemble information
                eccodes.codes_set(gid, "typeOfGeneratingProcess", 4)  # Ensemble forecast
                eccodes.codes_set(gid, "typeOfEnsembleForecast", 255)  # Unspecified
                eccodes.codes_set(gid, "numberOfForecastsInEnsemble", n_ensemble)
                eccodes.codes_set(gid, "perturbationNumber", int(ens_num) + 1)  # 1-based in GRIB
                
                # Set the data values
                values = data[ens_idx, :, :].flatten().astype(np.float64)
                eccodes.codes_set_values(gid, values)
                
                # Write the message to file
                eccodes.codes_write(gid, fout)
                
            finally:
                eccodes.codes_release(gid)


# =============================================================================
# NetCDF File Writer
# =============================================================================

def write_netcdf_file(
    output_path: Path,
    data: np.ndarray,
    variable: VariableConfig,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    reference_time: datetime,
    forecast_hour: int,
    ensemble_members: np.ndarray,
    engine: str = "netcdf4",
) -> None:
    """
    Write data to a NetCDF file using xarray.
    
    Parameters
    ----------
    output_path : Path
        Path where the NetCDF file will be saved
    data : np.ndarray
        3D array of shape (num_ensemble, num_lat, num_lon)
    variable : VariableConfig
        Variable configuration
    latitudes : np.ndarray
        1D array of latitude values
    longitudes : np.ndarray
        1D array of longitude values
    reference_time : datetime
        Forecast reference/analysis time
    forecast_hour : int
        Forecast lead time in hours
    ensemble_members : np.ndarray
        Array of ensemble member numbers (0-indexed)
    engine : str
        NetCDF engine to use ('netcdf4' or 'h5netcdf')
    """
    # Calculate valid time
    valid_time = reference_time + timedelta(hours=forecast_hour)
    
    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            variable.name: (
                ["number", "latitude", "longitude"],
                data,
                {
                    "units": variable.units,
                    "long_name": variable.name,
                    "standard_name": variable.name,
                },
            ),
        },
        coords={
            "number": (["number"], ensemble_members, {"long_name": "ensemble member"}),
            "latitude": (["latitude"], latitudes, {"units": "degrees_north"}),
            "longitude": (["longitude"], longitudes, {"units": "degrees_east"}),
            "time": reference_time,
            "step": forecast_hour,
            "valid_time": valid_time,
        },
        attrs={
            "Conventions": "CF-1.6",
            "history": f"Generated by generate_mock_data.py at {datetime.now(timezone.utc).isoformat()}",
            "reference_time": reference_time.isoformat(),
            "forecast_hour": forecast_hour,
        },
    )
    
    ds.to_netcdf(output_path, engine=engine)
    ds.close()


# =============================================================================
# Mock Data Generator
# =============================================================================

def generate_mock_file(
    output_dir: Path,
    variable: str,
    forecast_hour: int,
    reference_time: datetime,
    output_format: Literal["grib", "netcdf"] = "grib",
    num_ensemble_members: int = 50,
    lat_size: int = 361,
    lon_size: int = 720,
    netcdf_engine: str = "netcdf4",
    template_path: Optional[Path] = None,
) -> Path:
    """
    Generate a single mock file for one variable at one forecast step.
    
    Parameters
    ----------
    output_dir : Path
        Directory where the file will be saved
    variable : str
        Variable name (t2m, u10, v10, sp)
    forecast_hour : int
        Forecast lead time in hours
    reference_time : datetime
        Forecast reference/analysis time
    output_format : str
        Output format: 'grib' or 'netcdf'
    num_ensemble_members : int
        Number of ensemble members
    lat_size : int
        Number of latitude points (default: 361 for 0.5° global grid)
    lon_size : int
        Number of longitude points (default: 720 for 0.5° global grid)
    netcdf_engine : str
        Engine for NetCDF files ('netcdf4' or 'h5netcdf')
    template_path : Path, optional
        Path to a template NetCDF file to extract grid from
    
    Returns
    -------
    Path
        Path to the generated file
    """
    var_config = VARIABLE_CONFIGS[variable]
    
    # Get coordinates from template or use defaults
    if template_path is not None:
        template_ds = xr.open_dataset(template_path)
        latitudes = template_ds["latitude"].values
        longitudes = template_ds["longitude"].values
        lat_size = len(latitudes)
        lon_size = len(longitudes)
        template_ds.close()
    else:
        # Create coordinate arrays (0.5° global grid, N to S, 0 to 360°)
        latitudes = np.linspace(90, -90, lat_size)
        longitudes = np.linspace(0, 359.5, lon_size)
    
    # Generate random data within realistic bounds
    data = np.random.uniform(
        var_config.min_value,
        var_config.max_value,
        size=(num_ensemble_members, lat_size, lon_size)
    ).astype(np.float32)
    
    # Create descriptive filename
    extension = ".grib2" if output_format == "grib" else ".nc"
    filename = f"{variable}_step{forecast_hour:03d}{extension}"
    output_path = output_dir / filename
    
    # Write the file
    ensemble_members = np.arange(num_ensemble_members)
    
    if output_format == "grib":
        write_grib_file(
            output_path=output_path,
            data=data,
            variable=var_config,
            latitudes=latitudes,
            longitudes=longitudes,
            reference_time=reference_time,
            forecast_hour=forecast_hour,
            ensemble_members=ensemble_members,
        )
    else:
        write_netcdf_file(
            output_path=output_path,
            data=data,
            variable=var_config,
            latitudes=latitudes,
            longitudes=longitudes,
            reference_time=reference_time,
            forecast_hour=forecast_hour,
            ensemble_members=ensemble_members,
            engine=netcdf_engine,
        )
    
    return output_path


def generate_forecast_cycle(
    output_dir: Path,
    variables: List[str],
    reference_time: Optional[datetime] = None,
    output_format: Literal["grib", "netcdf"] = "grib",
    num_ensemble_members: int = 50,
    lat_size: int = 361,
    lon_size: int = 720,
    max_steps: Optional[int] = None,
    netcdf_engine: str = "netcdf4",
    template_path: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    """
    Generate a complete set of mock files for a forecast cycle.
    
    Creates one file per (variable, step) combination, matching the
    pipeline's chunking strategy.
    
    Parameters
    ----------
    output_dir : Path
        Directory where files will be saved
    variables : List[str]
        List of variable names to generate
    reference_time : datetime, optional
        Forecast reference time (default: current time rounded to 6 hours)
    output_format : str
        Output format: 'grib' or 'netcdf'
    num_ensemble_members : int
        Number of ensemble members per file
    lat_size : int
        Number of latitude points
    lon_size : int
        Number of longitude points
    max_steps : int, optional
        Maximum number of steps to generate (for testing)
    netcdf_engine : str
        Engine for NetCDF files ('netcdf4' or 'h5netcdf')
    template_path : Path, optional
        Path to a template NetCDF file to extract grid from
    
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping variable names to list of generated file paths
    """
    # Default reference time: current time rounded to nearest 6 hours
    if reference_time is None:
        now = datetime.now(timezone.utc)
        cycle_hour = (now.hour // 6) * 6
        reference_time = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    
    # Generate forecast steps
    forecast_hours = generate_forecast_steps()
    if max_steps is not None:
        forecast_hours = forecast_hours[:max_steps]
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track generated files
    generated_files: Dict[str, List[Path]] = {var: [] for var in variables}
    
    total_files = len(variables) * len(forecast_hours)
    file_count = 0
    
    format_name = "GRIB" if output_format == "grib" else "NetCDF"
    print(f"Generating {total_files} {format_name} files...")
    print(f"  Variables: {variables}")
    print(f"  Steps: {len(forecast_hours)} (0h to {forecast_hours[-1]}h)")
    print(f"  Ensemble members: {num_ensemble_members}")
    if template_path:
        print(f"  Grid: from template {template_path}")
    else:
        print(f"  Grid: {lat_size} x {lon_size}")
    print(f"  Reference time: {reference_time}")
    print()
    
    for variable in variables:
        for forecast_hour in forecast_hours:
            output_path = generate_mock_file(
                output_dir=output_dir,
                variable=variable,
                forecast_hour=forecast_hour,
                reference_time=reference_time,
                output_format=output_format,
                num_ensemble_members=num_ensemble_members,
                lat_size=lat_size,
                lon_size=lon_size,
                netcdf_engine=netcdf_engine,
                template_path=template_path,
            )
            generated_files[variable].append(output_path)
            file_count += 1
            
            # Progress update
            if file_count % 50 == 0:
                print(f"  Generated {file_count}/{total_files} files...")
    
    print(f"\n✓ Generated {total_files} {format_name} files in {output_dir}")
    
    return generated_files


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for generating mock files."""
    parser = argparse.ArgumentParser(
        description="Generate mock GRIB or NetCDF files for low-latency weather data pipeline testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Forecast Step Structure:
  Hours 0-90:   Hourly    (91 steps)
  Hours 93-144: 3-hourly  (18 steps)
  Hours 150-360: 6-hourly (36 steps)
  Total: 145 steps per variable

Examples:
  %(prog)s output_dir/                              # All 4 variables, all 145 steps (GRIB)
  %(prog)s output_dir/ --format netcdf             # Output as NetCDF instead
  %(prog)s output_dir/ --template data/sample.nc   # Use grid from template file
  %(prog)s output_dir/ --variables t2m u10         # Only temperature and u-wind
  %(prog)s output_dir/ --max-steps 10              # Only first 10 steps (for testing)
  %(prog)s output_dir/ --members 10 --lat 91 --lon 180  # Smaller grid for testing
        """,
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where mock files will be saved",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to a template NetCDF file to extract grid coordinates from",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="grib",
        choices=["grib", "netcdf"],
        help="Output format: 'grib' or 'netcdf' (default: grib)",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["t2m", "u10", "v10", "sp"],
        choices=["t2m", "u10", "v10", "sp"],
        help="Variables to generate (default: all four)",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=50,
        help="Number of ensemble members (default: 50)",
    )
    parser.add_argument(
        "--lat",
        type=int,
        default=361,
        help="Number of latitude points (default: 361 for 0.5° grid)",
    )
    parser.add_argument(
        "--lon",
        type=int,
        default=720,
        help="Number of longitude points (default: 720 for 0.5° grid)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of forecast steps to generate (for testing)",
    )
    parser.add_argument(
        "--reference-time",
        type=str,
        default=None,
        help="Forecast reference time in ISO format (default: current time rounded to 6h)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="netcdf4",
        choices=["netcdf4", "h5netcdf"],
        help="Engine for NetCDF files (default: netcdf4)",
    )
    
    args = parser.parse_args()
    
    # Parse reference time if provided
    reference_time = None
    if args.reference_time:
        reference_time = datetime.fromisoformat(args.reference_time)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    # Parse template path if provided
    template_path = Path(args.template) if args.template else None
    if template_path and not template_path.exists():
        parser.error(f"Template file not found: {template_path}")
    
    # Generate files
    output_dir = Path(args.output_dir)
    
    generated = generate_forecast_cycle(
        output_dir=output_dir,
        variables=args.variables,
        reference_time=reference_time,
        output_format=args.format,
        num_ensemble_members=args.members,
        lat_size=args.lat,
        lon_size=args.lon,
        max_steps=args.max_steps,
        netcdf_engine=args.engine,
        template_path=template_path,
    )
    
    # Summary
    print("\nGenerated files by variable:")
    for var, files in generated.items():
        print(f"  {var}: {len(files)} files")
    
    # Estimate size
    bytes_per_file = args.members * args.lat * args.lon * 4  # float32
    total_files = sum(len(f) for f in generated.values())
    total_bytes = total_files * bytes_per_file
    print(f"\nEstimated total size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
