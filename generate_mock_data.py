#!/usr/bin/env python3
"""
Generate mock NetCDF files based on a template from Copernicus Climate Data Store.
Creates 100 files with 50 ensemble members each, filled with random noise.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import uuid
import argparse


def generate_mock_file(template_path, output_path, num_ensemble_members=50, engine="netcdf4"):
    """
    Generate a mock NetCDF file based on a template.
    
    Parameters:
    -----------
    template_path : str or Path
        Path to the template NetCDF file
    output_path : str or Path
        Path where the mock file will be saved
    num_ensemble_members : int
        Number of ensemble members (default: 50)
    engine : str
        Engine to use for writing ('netcdf4' or 'h5netcdf', default: 'netcdf4')
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

    # Prepare coordinates dictionary
    coords = {
        "number": (["number"], new_number, template_ds["number"].attrs.copy()),
        "valid_time": template_ds["valid_time"],
        "latitude": template_ds["latitude"],
        "longitude": template_ds["longitude"],
    }

    # Add expver as a coordinate (not a data variable!)
    if "expver" in template_ds:
        coords["expver"] = template_ds["expver"]

        # Copy t2m attributes but fix the coordinates attribute
    t2m_attrs = template_ds["t2m"].attrs.copy()
    
    # Update the 'coordinates' attribute to match our actual coordinates
    if "expver" in coords:
        t2m_attrs["coordinates"] = "number valid_time latitude longitude expver"
    else:
        t2m_attrs["coordinates"] = "number valid_time latitude longitude"

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
    
    # Save to NetCDF file
    new_ds.to_netcdf(output_path, engine=engine)

    # Close datasets
    template_ds.close()
    new_ds.close()

    return output_path


def main():
    """Main function to generate 100 mock files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate mock NetCDF files based on a template from Copernicus Climate Data Store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s template.nc output_dir/
  %(prog)s data/example/652f73a7818c431a469c7ed3e9054e0a.nc data/ --num-files 50 --num-members 25
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
        "--engine",
        type=str,
        default="netcdf4",
        choices=["netcdf4", "h5netcdf"],
        help="Engine to use for writing files (default: netcdf4). Note: h5netcdf may not work on Databricks.",
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

    print(
        f"Generating {args.num_files} mock NetCDF files with {args.num_members} ensemble members each..."
    )
    print(f"Template: {template_file}")
    print(f"Output directory: {output_dir}")
    print(f"Engine: {args.engine}")
    print()

    # Generate mock files
    for i in range(args.num_files):
        # Generate unique filename using UUID (similar to the template)
        filename = f"{uuid.uuid4().hex}.nc"
        output_path = output_dir / filename

        # Generate the mock file
        generate_mock_file(
            template_file, output_path, num_ensemble_members=args.num_members, engine=args.engine
        )

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_files} files...")

    print()
    print(f"✓ Successfully generated {args.num_files} mock files in {output_dir}")
    print(
        f"  Each file contains {args.num_members} ensemble members with random temperature data."
    )


if __name__ == "__main__":
    main()
