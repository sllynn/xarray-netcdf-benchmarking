#!/usr/bin/env python3
"""
Verify the generated mock NetCDF files.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import argparse


def verify_files(data_dir):
    """Verify the generated mock files.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing NetCDF files to verify
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"✗ Error: Directory not found: {data_dir}")
        return False
    
    if not data_dir.is_dir():
        print(f"✗ Error: Not a directory: {data_dir}")
        return False

    # Get all NC files (excluding the example subdirectory)
    nc_files = [f for f in data_dir.glob("*.nc")]

    print(f"Found {len(nc_files)} NetCDF files in {data_dir}")
    print()

    if len(nc_files) == 0:
        print("⚠️  No NetCDF files found in directory!")
        print("    Make sure you've generated mock data first.")
        return False

    # Check a few random files
    sample_file = nc_files[0]
    print(f"Examining sample file: {sample_file.name}")
    print("-" * 70)

    ds = xr.open_dataset(sample_file)

    # Display dimensions
    print("\nDimensions:")
    for dim, size in ds.sizes.items():
        print(f"  {dim}: {size}")

    # Display variables
    print("\nMain data variable:")
    print(f"  t2m: {ds['t2m'].shape}")
    print(f"  Data type: {ds['t2m'].dtype}")
    print(f"  Units: {ds['t2m'].attrs.get('units', 'N/A')}")

    # Show data statistics
    print("\nTemperature statistics (first ensemble member):")
    t2m_sample = ds["t2m"].values[0, 0, :, :]
    print(f"  Min: {np.min(t2m_sample):.2f} K")
    print(f"  Max: {np.max(t2m_sample):.2f} K")
    print(f"  Mean: {np.mean(t2m_sample):.2f} K")
    print(f"  Std: {np.std(t2m_sample):.2f} K")

    # Verify ensemble members
    print(f"\nEnsemble members: {ds.sizes['number']}")
    print(f"  Number range: {ds['number'].values[0]} to {ds['number'].values[-1]}")

    # Verify spatial dimensions
    print("\nSpatial dimensions:")
    print(
        f"  Longitude: {ds.sizes['longitude']} points ({ds['longitude'].values[0]:.1f}° to {ds['longitude'].values[-1]:.1f}°)"
    )
    print(
        f"  Latitude: {ds.sizes['latitude']} points ({ds['latitude'].values[0]:.1f}° to {ds['latitude'].values[-1]:.1f}°)"
    )

    # Store values before closing
    num_ensemble = ds.sizes['number']
    num_latitude = ds.sizes['latitude']
    num_longitude = ds.sizes['longitude']
    
    ds.close()

    print()
    print("-" * 70)
    print("✓ Verification complete!")
    print(f"  Total files: {len(nc_files)}")
    print(f"  Ensemble members per file: {num_ensemble}")
    print(f"  Spatial resolution: {num_latitude} x {num_longitude} (lat x lon)")
    
    return True


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Verify generated mock NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/
  %(prog)s /path/to/netcdf/files/
        """,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing NetCDF files to verify",
    )
    
    args = parser.parse_args()
    
    # Verify files
    success = verify_files(args.data_dir)
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
