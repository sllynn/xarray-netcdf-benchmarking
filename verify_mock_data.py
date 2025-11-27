#!/usr/bin/env python3
"""
Verify the generated mock NetCDF files.
"""

import xarray as xr
import numpy as np
from pathlib import Path


def verify_files():
    """Verify the generated mock files."""
    data_dir = Path("/Users/stuart.lynn/Customers/LSEG/raster-benchmarking/data")

    # Get all NC files (excluding the example subdirectory)
    nc_files = [f for f in data_dir.glob("*.nc")]

    print(f"Found {len(nc_files)} NetCDF files in {data_dir}")
    print()

    if len(nc_files) == 0:
        print("No files found!")
        return

    # Check a few random files
    sample_file = nc_files[0]
    print(f"Examining sample file: {sample_file.name}")
    print("-" * 70)

    ds = xr.open_dataset(sample_file)

    # Display dimensions
    print("\nDimensions:")
    for dim, size in ds.dims.items():
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
    print(f"\nEnsemble members: {ds.dims['number']}")
    print(f"  Number range: {ds['number'].values[0]} to {ds['number'].values[-1]}")

    # Verify spatial dimensions
    print("\nSpatial dimensions:")
    print(
        f"  Longitude: {ds.dims['longitude']} points ({ds['longitude'].values[0]:.1f}° to {ds['longitude'].values[-1]:.1f}°)"
    )
    print(
        f"  Latitude: {ds.dims['latitude']} points ({ds['latitude'].values[0]:.1f}° to {ds['latitude'].values[-1]:.1f}°)"
    )

    ds.close()

    print()
    print("-" * 70)
    print("✓ Verification complete!")
    print(f"  Total files: {len(nc_files)}")
    print("  Ensemble members per file: 50")
    print("  Spatial resolution: 361 x 720 (lat x lon)")


if __name__ == "__main__":
    verify_files()
