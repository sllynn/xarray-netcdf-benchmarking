#!/usr/bin/env python3
"""
Diagnostic script for h5netcdf issues on Databricks.

This script helps diagnose HDF5 dimension scale issues when reading
NetCDF files with h5netcdf that were created with netcdf4.
"""

import sys
from pathlib import Path
import argparse


def check_library_versions():
    """Check versions of relevant libraries."""
    print("=" * 70)
    print("LIBRARY VERSIONS")
    print("=" * 70)
    
    try:
        import h5py
        print(f"✓ h5py: {h5py.__version__}")
        print(f"  HDF5 C library: {h5py.version.hdf5_version}")
    except ImportError:
        print("✗ h5py: Not installed")
    
    try:
        import h5netcdf
        print(f"✓ h5netcdf: {h5netcdf.__version__}")
    except ImportError:
        print("✗ h5netcdf: Not installed")
    
    try:
        import netCDF4
        print(f"✓ netCDF4: {netCDF4.__version__}")
        print(f"  NetCDF C library: {netCDF4.__netcdf4libversion__}")
        print(f"  HDF5 C library: {netCDF4.__hdf5libversion__}")
    except ImportError:
        print("✗ netCDF4: Not installed")
    
    try:
        import xarray as xr
        print(f"✓ xarray: {xr.__version__}")
    except ImportError:
        print("✗ xarray: Not installed")
    
    print()


def test_read_with_different_methods(file_path):
    """Test reading a file with different methods."""
    import xarray as xr
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return
    
    print("=" * 70)
    print(f"TESTING FILE: {file_path.name}")
    print("=" * 70)
    print()
    
    methods = [
        {
            "name": "netCDF4 engine (default)",
            "kwargs": {"engine": "netcdf4"},
        },
        {
            "name": "h5netcdf engine (default)",
            "kwargs": {"engine": "h5netcdf"},
        },
        {
            "name": "h5netcdf with phony_dims='sort'",
            "kwargs": {
                "engine": "h5netcdf",
                "backend_kwargs": {"phony_dims": "sort"},
            },
        },
        {
            "name": "h5netcdf with phony_dims='access'",
            "kwargs": {
                "engine": "h5netcdf",
                "backend_kwargs": {"phony_dims": "access"},
            },
        },
        {
            "name": "h5netcdf with decode_cf=False",
            "kwargs": {
                "engine": "h5netcdf",
                "decode_cf": False,
            },
        },
    ]
    
    for method in methods:
        print(f"Testing: {method['name']}")
        print("-" * 70)
        try:
            ds = xr.open_dataset(file_path, **method["kwargs"])
            print(f"✓ SUCCESS")
            print(f"  Dimensions: {dict(ds.sizes)}")
            print(f"  Variables: {list(ds.data_vars)}")
            ds.close()
        except Exception as e:
            print(f"✗ FAILED: {type(e).__name__}")
            print(f"  Error: {str(e)[:200]}")
        print()


def inspect_hdf5_structure(file_path):
    """Inspect HDF5 structure directly."""
    import h5py
    
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return
    
    print("=" * 70)
    print(f"HDF5 STRUCTURE INSPECTION: {file_path.name}")
    print("=" * 70)
    print()
    
    try:
        with h5py.File(file_path, "r") as f:
            print("Groups and Datasets:")
            
            def print_structure(name, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}{name}: {obj.shape} {obj.dtype}")
                    # Check for dimension scales
                    if hasattr(obj, "dims"):
                        try:
                            print(f"{indent}  Dimension scales: {len(obj.dims[0])}")
                        except Exception as e:
                            print(f"{indent}  Dimension scales error: {e}")
                else:
                    print(f"{indent}{name}/ (group)")
            
            f.visititems(print_structure)
            
            print("\n✓ HDF5 structure inspected successfully")
    except Exception as e:
        print(f"✗ FAILED to inspect: {type(e).__name__}")
        print(f"  Error: {str(e)}")
    print()


def suggest_solutions():
    """Suggest solutions for the h5netcdf issue."""
    print("=" * 70)
    print("SUGGESTED SOLUTIONS")
    print("=" * 70)
    print()
    
    print("If h5netcdf fails with 'H5DSget_num_scales' error:\n")
    
    print("1. Use phony_dims parameter (RECOMMENDED):")
    print("   ```python")
    print("   ds = xr.open_dataset(")
    print("       file_path,")
    print("       engine='h5netcdf',")
    print("       backend_kwargs={'phony_dims': 'sort'}")
    print("   )")
    print("   ```\n")
    
    print("2. Write files with h5netcdf instead of netcdf4:")
    print("   In generate_mock_data.py, change:")
    print("   ```python")
    print("   new_ds.to_netcdf(output_path, engine='h5netcdf')")
    print("   ```\n")
    
    print("3. Stick with netcdf4 engine:")
    print("   The netcdf4 engine handles these files correctly")
    print("   and was found to be faster in benchmarks anyway.\n")
    
    print("4. Check HDF5 library versions:")
    print("   Ensure h5py and netCDF4 use compatible HDF5 versions")
    print("   On Databricks, you may need to reinstall libraries.\n")
    
    print("5. Use decode_cf=False (last resort):")
    print("   This disables CF conventions but may allow reading:")
    print("   ```python")
    print("   ds = xr.open_dataset(file_path, engine='h5netcdf', decode_cf=False)")
    print("   ```\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Diagnose h5netcdf issues with NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=str,
        help="Path to a NetCDF file to test",
    )
    parser.add_argument(
        "--versions-only",
        action="store_true",
        help="Only show library versions",
    )
    
    args = parser.parse_args()
    
    print()
    check_library_versions()
    
    if args.versions_only:
        return
    
    if args.file_path:
        test_read_with_different_methods(args.file_path)
        print()
        inspect_hdf5_structure(args.file_path)
    
    suggest_solutions()


if __name__ == "__main__":
    main()

