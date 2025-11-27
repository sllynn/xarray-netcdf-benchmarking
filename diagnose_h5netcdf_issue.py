#!/usr/bin/env python3
"""
Diagnostic script for h5netcdf issues on Databricks.

This script helps diagnose HDF5 dimension scale issues when reading
NetCDF files with h5netcdf that were created with netcdf4.
"""

from pathlib import Path
import argparse


def check_library_versions():
    """Check versions of relevant libraries."""
    print("=" * 70)
    print("LIBRARY VERSIONS")
    print("=" * 70)
    
    versions = {}
    
    try:
        import h5py
        print(f"✓ h5py: {h5py.__version__}")
        print(f"  HDF5 C library: {h5py.version.hdf5_version}")
        versions['h5py_hdf5'] = h5py.version.hdf5_version
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
        versions['netcdf4_hdf5'] = netCDF4.__hdf5libversion__
    except ImportError:
        print("✗ netCDF4: Not installed")
    
    try:
        import xarray as xr
        print(f"✓ xarray: {xr.__version__}")
    except ImportError:
        print("✗ xarray: Not installed")
    
    # Check for HDF5 version mismatch
    if 'h5py_hdf5' in versions and 'netcdf4_hdf5' in versions:
        if versions['h5py_hdf5'] != versions['netcdf4_hdf5']:
            import os
            is_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ
            
            print()
            if is_databricks:
                print("⚠️  WARNING: HDF5 version mismatch detected!")
                print(f"   h5py uses:     HDF5 {versions['h5py_hdf5']}")
                print(f"   netCDF4 uses:  HDF5 {versions['netcdf4_hdf5']}")
                print("   ⚠️  This CAN cause problems on Databricks!")
                print("   → Each package trying to use shared system HDF5")
            else:
                print("ℹ️  Note: HDF5 version difference detected")
                print(f"   h5py uses:     HDF5 {versions['h5py_hdf5']}")
                print(f"   netCDF4 uses:  HDF5 {versions['netcdf4_hdf5']}")
                print("   ✅ This is NORMAL on local machines (packages use bundled HDF5)")
                print("   ✅ Each package has its own isolated HDF5 library")
                print("   ⚠️  Only a concern on Databricks with shared system HDF5")
    
    print()


def check_file_and_environment(file_path):
    """Check file accessibility and environment details."""
    import os
    import platform
    
    file_path = Path(file_path)
    
    print("=" * 70)
    print("FILE AND ENVIRONMENT CHECK")
    print("=" * 70)
    print()
    
    # System info
    print("System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    
    # Check if running on Databricks
    is_databricks = 'DATABRICKS_RUNTIME_VERSION' in os.environ
    if is_databricks:
        print(f"  Databricks Runtime: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Unknown')}")
        print("  ⚠️  Running on Databricks - see special notes below")
    
    print()
    
    # File info
    print(f"File: {file_path}")
    if not file_path.exists():
        print("  ✗ File does not exist")
        return False
    
    print(f"  ✓ File exists")
    print(f"  Path: {file_path.absolute()}")
    print(f"  Size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Check file permissions
    try:
        print(f"  Readable: {os.access(file_path, os.R_OK)}")
        print(f"  Writable: {os.access(file_path, os.W_OK)}")
    except Exception as e:
        print(f"  Permission check failed: {e}")
    
    # Check if file is on DBFS
    path_str = str(file_path.absolute())
    if '/dbfs/' in path_str:
        print("  ⚠️  File is on DBFS (may have FUSE mount issues)")
        print("      Consider using: /tmp/ or file:// paths")
    elif path_str.startswith('/tmp'):
        print("  ℹ️  File is in /tmp (local cluster storage)")
    
    print()
    return True


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
    """Suggest solutions for NetCDF/HDF5 issues."""
    print("=" * 70)
    print("SUGGESTED SOLUTIONS")
    print("=" * 70)
    print()
    
    print("=" * 70)
    print("FOR: NetCDF4 'HDF error' on Databricks")
    print("=" * 70)
    print()
    print("If netCDF4 fails with '[Errno -101] NetCDF: HDF error':\n")
    
    print("1. Reinstall netCDF4 with pip (RECOMMENDED for Databricks):")
    print("   ```python")
    print("   %pip uninstall -y netCDF4")
    print("   %pip install --no-cache-dir netCDF4==1.7.3")
    print("   dbutils.library.restartPython()")
    print("   ```")
    print("   This ensures pip-installed library instead of system library.\n")
    
    print("2. Copy files to local /tmp instead of DBFS:")
    print("   ```python")
    print("   # DBFS has FUSE mount issues")
    print("   import shutil")
    print("   shutil.copy('/dbfs/path/to/file.nc', '/tmp/file.nc')")
    print("   ds = xr.open_dataset('/tmp/file.nc')")
    print("   ```\n")
    
    print("3. Use h5netcdf engine instead:")
    print("   ```python")
    print("   ds = xr.open_dataset(")
    print("       file_path,")
    print("       engine='h5netcdf',")
    print("       backend_kwargs={'phony_dims': 'sort'}")
    print("   )")
    print("   ```")
    print("   h5netcdf may work better on Databricks.\n")
    
    print("4. Install netcdf4 from conda-forge:")
    print("   If using a conda environment:")
    print("   ```bash")
    print("   conda install -c conda-forge netcdf4=1.7.3")
    print("   ```\n")
    
    print("5. Check file was created correctly:")
    print("   Try opening with h5py directly:")
    print("   ```python")
    print("   import h5py")
    print("   with h5py.File(file_path, 'r') as f:")
    print("       print(list(f.keys()))")
    print("   ```\n")
    
    print()
    print("=" * 70)
    print("FOR: h5netcdf 'H5DSget_num_scales' error")
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
    
    print()
    print("=" * 70)
    print("DATABRICKS-SPECIFIC NOTES")
    print("=" * 70)
    print()
    print("• Library conflicts: System libraries may conflict with pip packages")
    print("• DBFS FUSE: /dbfs/ paths use FUSE which can cause HDF5 issues")
    print("• Best practice: Use /tmp/ or /local_disk0/ for NetCDF files")
    print("• Cluster restart: May be needed after installing new libraries")
    print("• DBR version: Try using DBR ML which has better data science libraries")
    print()


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
        # Check file and environment
        file_ok = check_file_and_environment(args.file_path)
        
        if file_ok:
            # Test reading methods
            test_read_with_different_methods(args.file_path)
            print()
            
            # Inspect HDF5 structure
            inspect_hdf5_structure(args.file_path)
    
    suggest_solutions()


if __name__ == "__main__":
    main()

