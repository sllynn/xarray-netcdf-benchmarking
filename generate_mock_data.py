#!/usr/bin/env python3
"""
Generate mock NetCDF or GRIB files based on a template from Copernicus Climate Data Store.
Creates files with ensemble members filled with random noise.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import uuid
import argparse


def write_grib_file(output_path, data, latitudes, longitudes, valid_time, ensemble_members):
    """
    Write data to a GRIB2 file using eccodes directly.
    
    Parameters:
    -----------
    output_path : str or Path
        Path where the GRIB file will be saved
    data : np.ndarray
        4D array of shape (num_ensemble, num_time, num_lat, num_lon)
    latitudes : np.ndarray
        1D array of latitude values (must be regularly spaced)
    longitudes : np.ndarray
        1D array of longitude values (must be regularly spaced)
    valid_time : np.ndarray
        Array of datetime64 values for each time step
    ensemble_members : np.ndarray
        Array of ensemble member numbers
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
    
    n_ensemble, n_time, n_lat, n_lon = data.shape
    
    # Calculate grid parameters
    lat_start = float(latitudes[0])
    lat_end = float(latitudes[-1])
    lon_start = float(longitudes[0])
    lon_end = float(longitudes[-1])
    
    # Calculate grid increments (handle both ascending and descending)
    if n_lat > 1:
        lat_increment = abs(float(latitudes[1] - latitudes[0]))
    else:
        lat_increment = 1.0
    
    if n_lon > 1:
        lon_increment = abs(float(longitudes[1] - longitudes[0]))
    else:
        lon_increment = 1.0
    
    # Determine scan direction
    lat_descending = lat_start > lat_end  # True if latitudes go from N to S
    
    with open(output_path, 'wb') as fout:
        for ens_idx, ens_num in enumerate(ensemble_members):
            for t_idx in range(n_time):
                # Create a new GRIB2 message from sample
                gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
                
                try:
                    # Set the parameter (2m temperature)
                    eccodes.codes_set(gid, "discipline", 0)  # Meteorological
                    eccodes.codes_set(gid, "parameterCategory", 0)  # Temperature
                    eccodes.codes_set(gid, "parameterNumber", 0)  # Temperature
                    eccodes.codes_set(gid, "shortName", "2t")
                    
                    # Set level type (2m above ground)
                    eccodes.codes_set(gid, "typeOfFirstFixedSurface", 103)  # Height above ground
                    eccodes.codes_set(gid, "scaleFactorOfFirstFixedSurface", 0)
                    eccodes.codes_set(gid, "scaledValueOfFirstFixedSurface", 2)  # 2 meters
                    
                    # Set grid definition
                    eccodes.codes_set(gid, "gridType", "regular_ll")
                    eccodes.codes_set(gid, "Ni", n_lon)  # Number of points along longitude
                    eccodes.codes_set(gid, "Nj", n_lat)  # Number of points along latitude
                    
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
                    
                    # Set scanning mode (bit 1: 0=+i, bit 2: 0=-j (N to S), bit 3: 0=consecutive i)
                    scan_mode = 0
                    if not lat_descending:
                        scan_mode |= 64  # bit 2 set: +j direction (S to N)
                    eccodes.codes_set(gid, "jScansPositively", 0 if lat_descending else 1)
                    
                    # Set time information
                    vt = valid_time[t_idx] if n_time > 1 else valid_time
                    # Ensure we have a scalar value (flatten any nested arrays)
                    while isinstance(vt, np.ndarray):
                        vt = vt.flat[0]
                    # Convert to Python datetime using pandas (handles all numpy datetime types)
                    dt = pd.Timestamp(vt).to_pydatetime()
                    
                    eccodes.codes_set(gid, "dataDate", int(dt.strftime("%Y%m%d")))
                    eccodes.codes_set(gid, "dataTime", int(dt.strftime("%H%M")))
                    
                    # Set Product Definition Template for ensemble data (PDT 4.1)
                    # This enables ensemble-related keys
                    eccodes.codes_set(gid, "productDefinitionTemplateNumber", 1)
                    
                    # Set ensemble information
                    eccodes.codes_set(gid, "typeOfGeneratingProcess", 4)  # Ensemble forecast
                    eccodes.codes_set(gid, "typeOfEnsembleForecast", 255)  # Unspecified
                    eccodes.codes_set(gid, "numberOfForecastsInEnsemble", n_ensemble)
                    eccodes.codes_set(gid, "perturbationNumber", int(ens_num) + 1)  # 1-based in GRIB
                    
                    # Set the data values
                    values = data[ens_idx, t_idx, :, :].flatten().astype(np.float64)
                    eccodes.codes_set_values(gid, values)
                    
                    # Write the message to file
                    eccodes.codes_write(gid, fout)
                    
                finally:
                    eccodes.codes_release(gid)


def generate_mock_file(template_path, output_path, num_ensemble_members=50, time_offset_hours=0, 
                       output_format="netcdf", engine="netcdf4"):
    """
    Generate a mock file based on a template.
    
    Parameters:
    -----------
    template_path : str or Path
        Path to the template file
    output_path : str or Path
        Path where the mock file will be saved
    num_ensemble_members : int
        Number of ensemble members (default: 50)
    time_offset_hours : int
        Number of hours to offset the valid_time from the template (default: 0)
    output_format : str
        Output format: 'netcdf' or 'grib' (default: 'netcdf')
    engine : str
        Engine to use for writing NetCDF files ('netcdf4' or 'h5netcdf', default: 'netcdf4')
    """
    # Open the template file
    template_ds = xr.open_dataset(template_path)

    # Get dimensions from template
    n_valid_time = template_ds.sizes["valid_time"]
    n_latitude = template_ds.sizes["latitude"]
    n_longitude = template_ds.sizes["longitude"]

    # Create new dimensions
    new_number = np.arange(num_ensemble_members)

    # Generate random temperature data
    # Using realistic temperature range: 220K to 320K (approximately -53°C to 47°C)
    random_t2m = np.random.uniform(
        220, 320, size=(num_ensemble_members, n_valid_time, n_latitude, n_longitude)
    ).astype(np.float32)

    # Create modified valid_time with offset
    # Add time_offset_hours to the template's time value
    template_time_values = template_ds["valid_time"].values
    if time_offset_hours != 0:
        # Add the offset in hours (time values are typically numpy.datetime64 or similar)
        new_time_values = template_time_values + np.timedelta64(time_offset_hours, 'h')
    else:
        new_time_values = template_time_values

    # CRITICAL: Copy coordinates WITHOUT their coordinate associations
    # Use .values and .attrs separately, not the whole DataArray
    coords = {
        "number": (["number"], new_number, template_ds["number"].attrs.copy()),
        "valid_time": (
            ["valid_time"],
            new_time_values,  # Use the offset time values
            template_ds["valid_time"].attrs.copy()
        ),
        "latitude": (
            ["latitude"],
            template_ds["latitude"].values,
            template_ds["latitude"].attrs.copy()
        ),
        "longitude": (
            ["longitude"],
            template_ds["longitude"].values,
            template_ds["longitude"].attrs.copy()
        ),
    }
        # Copy t2m attributes but fix the coordinates attribute
    t2m_attrs = template_ds["t2m"].attrs.copy()
    if "coordinates" in t2m_attrs:
        del t2m_attrs["coordinates"]

    # Create new dataset with modified dimensions
    new_ds = xr.Dataset(
        data_vars={
            "t2m": (
                ["number", "valid_time", "latitude", "longitude"],
                random_t2m,
                t2m_attrs,
            ),
        },
        coords=coords,
        attrs=template_ds.attrs.copy(),
    )

    # Update the GRIB_totalNumber attribute to reflect new ensemble size
    if "GRIB_totalNumber" in new_ds["t2m"].attrs:
        new_ds["t2m"].attrs["GRIB_totalNumber"] = num_ensemble_members
    
    # Save to file based on output format
    if output_format == "grib":
        write_grib_file(
            output_path=output_path,
            data=random_t2m,
            latitudes=template_ds["latitude"].values,
            longitudes=template_ds["longitude"].values,
            valid_time=new_time_values,
            ensemble_members=new_number
        )
    else:
        # Save to NetCDF file
        new_ds.to_netcdf(output_path, engine=engine)

    # Close datasets
    template_ds.close()
    new_ds.close()

    return output_path


def main():
    """Main function to generate mock files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate mock NetCDF or GRIB files based on a template from Copernicus Climate Data Store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s template.nc output_dir/
  %(prog)s data/example/652f73a7818c431a469c7ed3e9054e0a.nc data/ --num-files 50 --num-members 25
  %(prog)s template.nc data/ --num-files 100 --time-step-hours 6   # Files 6 hours apart (typical)
  %(prog)s template.nc data/ --num-files 365 --time-step-hours 24  # Daily for 1 year
  %(prog)s template.nc data/ --output grib                         # Output as GRIB files
        """,
    )
    parser.add_argument(
        "template_file", type=str, help="Path to the template NetCDF file"
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory where mock files will be saved"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=100,
        help="Number of mock files to generate (default: 100)",
    )
    parser.add_argument(
        "--num-members",
        type=int,
        default=50,
        help="Number of ensemble members per file (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="netcdf",
        choices=["netcdf", "grib"],
        help="Output file format: 'netcdf' or 'grib' (default: netcdf). Note: GRIB output requires eccodes library.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="netcdf4",
        choices=["netcdf4", "h5netcdf"],
        help="Engine to use for writing NetCDF files (default: netcdf4). Only applies when --output=netcdf.",
    )
    parser.add_argument(
        "--time-step-hours",
        type=int,
        default=6,
        help="Number of hours between each file's valid_time (default: 6, typical for weather forecasts)",
    )
    
    args = parser.parse_args()

    # Convert to Path objects
    template_file = Path(args.template_file)
    output_dir = Path(args.output_dir)

    # Validate template file exists
    if not template_file.exists():
        parser.error(f"Template file not found: {template_file}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension based on output format
    file_extension = ".grib" if args.output == "grib" else ".nc"
    format_name = "GRIB" if args.output == "grib" else "NetCDF"

    print(
        f"Generating {args.num_files} mock {format_name} files with {args.num_members} ensemble members each..."
    )
    print(f"Template: {template_file}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.output}")
    if args.output == "netcdf":
        print(f"NetCDF engine: {args.engine}")
    print(f"Time step: {args.time_step_hours} hour(s) between files")
    print()

    # Generate mock files
    for i in range(args.num_files):
        # Generate unique filename using UUID (similar to the template)
        filename = f"{uuid.uuid4().hex}{file_extension}"
        output_path = output_dir / filename

        # Calculate time offset for this file (each file is time_step_hours apart)
        time_offset = i * args.time_step_hours

        # Generate the mock file
        generate_mock_file(
            template_file, 
            output_path, 
            num_ensemble_members=args.num_members, 
            time_offset_hours=time_offset,
            output_format=args.output,
            engine=args.engine
        )

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_files} files...")

    print()
    print(f"✓ Successfully generated {args.num_files} mock {format_name} files in {output_dir}")
    print(f"  Each file contains {args.num_members} ensemble members with random temperature data.")
    total_hours = (args.num_files - 1) * args.time_step_hours
    if total_hours > 0:
        total_days = total_hours / 24
        print(f"  Time range: {total_hours} hours ({total_days:.1f} days, {args.num_files} time steps, {args.time_step_hours}h apart)")


if __name__ == "__main__":
    main()
