#!/usr/bin/env python3
"""
Convert NetCDF files to Zarr format.

Supports two conversion modes:
1. Individual: Convert each NetCDF file to a separate Zarr directory
2. Stacked: Combine all NetCDF files into a single Zarr, stacked along valid_time
"""

import xarray as xr
import time
from pathlib import Path
import argparse
from glob import glob
from datetime import datetime
import json
import warnings

# Suppress Zarr V3 warnings about unstable specifications
warnings.filterwarnings('ignore', category=UserWarning, module='zarr')


def convert_individual(nc_files, output_dir, engine="netcdf4"):
    """
    Convert each NetCDF file to an individual Zarr directory.
    
    Parameters:
    -----------
    nc_files : list
        List of NetCDF file paths
    output_dir : str or Path
        Output directory for Zarr files
    engine : str
        NetCDF engine to use for reading
        
    Returns:
    --------
    dict with timing and results
    """
    print("\n" + "="*70)
    print("INDIVIDUAL CONVERSION: NetCDF → Zarr")
    print("="*70)
    print(f"Number of files: {len(nc_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Reading engine: {engine}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        "mode": "individual",
        "total_files": len(nc_files),
        "successful": 0,
        "failed": 0,
        "file_times": [],
        "errors": []
    }
    
    overall_start = time.time()
    
    for i, nc_file in enumerate(nc_files, 1):
        nc_path = Path(nc_file)
        zarr_name = nc_path.stem + ".zarr"
        zarr_path = output_path / zarr_name
        
        print(f"\n[{i}/{len(nc_files)}] Converting: {nc_path.name}")
        
        file_start = time.time()
        try:
            # Read NetCDF
            backend_kwargs = {}
            if engine == "h5netcdf":
                backend_kwargs = {"phony_dims": "sort"}
                
            ds = xr.open_dataset(nc_file, engine=engine, backend_kwargs=backend_kwargs)
            
            # Load data into memory first to avoid chunking issues
            ds_loaded = ds.load()
            ds.close()
            
            # Convert Unicode string types to object dtype to avoid Zarr v3 warnings
            # This handles the FixedLengthUTF32 issue with coordinates like 'expver'
            for var in ds_loaded.coords:
                if ds_loaded[var].dtype.kind == 'U':  # Unicode string
                    # Convert to object dtype (variable-length strings)
                    ds_loaded[var] = ds_loaded[var].astype(object)
            
            # Write to Zarr
            ds_loaded.to_zarr(zarr_path, mode='w')
            
            file_time = time.time() - file_start
            results["file_times"].append({
                "file": nc_path.name,
                "time": file_time,
                "success": True
            })
            results["successful"] += 1
            
            print(f"  ✓ Completed in {file_time:.3f}s → {zarr_name}")
            
        except Exception as e:
            file_time = time.time() - file_start
            error_msg = str(e)
            results["file_times"].append({
                "file": nc_path.name,
                "time": file_time,
                "success": False,
                "error": error_msg
            })
            results["failed"] += 1
            results["errors"].append({
                "file": nc_path.name,
                "error": error_msg
            })
            
            print(f"  ✗ Failed in {file_time:.3f}s: {error_msg}")
    
    results["total_time"] = time.time() - overall_start
    
    print("\n" + "="*70)
    print("Individual conversion complete!")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Successful: {results['successful']}/{results['total_files']}")
    print(f"  Failed: {results['failed']}/{results['total_files']}")
    print(f"  Avg time per file: {results['total_time']/len(nc_files):.3f}s")
    print("="*70)
    
    return results


def convert_stacked(nc_files, output_path, engine="netcdf4"):
    """
    Convert all NetCDF files into a single stacked Zarr.
    
    Parameters:
    -----------
    nc_files : list
        List of NetCDF file paths
    output_path : str or Path
        Output path for the stacked Zarr directory
    engine : str
        NetCDF engine to use for reading
        
    Returns:
    --------
    dict with timing and results
    """
    print("\n" + "="*70)
    print("STACKED CONVERSION: Multiple NetCDF → Single Zarr")
    print("="*70)
    print(f"Number of files: {len(nc_files)}")
    print(f"Output: {output_path}")
    print(f"Reading engine: {engine}")
    
    results = {
        "mode": "stacked",
        "total_files": len(nc_files),
        "output_path": str(output_path)
    }
    
    overall_start = time.time()
    
    try:
        print("\nStep 1: Opening multi-file dataset...")
        open_start = time.time()
        
        backend_kwargs = {}
        if engine == "h5netcdf":
            backend_kwargs = {"phony_dims": "sort"}
        
        # Open all files as a single dataset, concatenating along valid_time
        ds = xr.open_mfdataset(
            nc_files,
            engine=engine,
            combine="nested",
            concat_dim="valid_time",
            backend_kwargs=backend_kwargs
        )
        
        results["open_time"] = time.time() - open_start
        print(f"  ✓ Opened in {results['open_time']:.3f}s")
        print(f"  Dataset shape: {dict(ds.sizes)}")
        print(f"  Variables: {list(ds.data_vars)}")
        
        print("\nStep 2: Writing to Zarr...")
        write_start = time.time()
        
        # Ensure output directory doesn't exist
        output_path = Path(output_path)
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        
        # Load data into memory first (safer for large multi-file operations)
        print("  Loading data into memory...")
        ds_loaded = ds.load()
        ds.close()
        
        # Convert Unicode string types to object dtype to avoid Zarr v3 warnings
        print("  Converting data types for Zarr v3 compatibility...")
        for var in ds_loaded.coords:
            if ds_loaded[var].dtype.kind == 'U':  # Unicode string
                # Convert to object dtype (variable-length strings)
                ds_loaded[var] = ds_loaded[var].astype(object)
        
        # Set explicit encoding for time coordinate to avoid serialization warnings
        encoding = {}
        if 'valid_time' in ds_loaded.coords:
            # Use hours as the time unit to handle sub-daily time steps
            encoding['valid_time'] = {
                'units': 'hours since 1970-01-01',
                'calendar': 'proleptic_gregorian'
            }
        
        # Write to Zarr
        print("  Writing to disk...")
        ds_loaded.to_zarr(output_path, mode='w', encoding=encoding)
        
        results["write_time"] = time.time() - write_start
        print(f"  ✓ Written in {results['write_time']:.3f}s")
        
        ds.close()
        
        results["total_time"] = time.time() - overall_start
        results["success"] = True
        
        print("\n" + "="*70)
        print("Stacked conversion complete!")
        print(f"  Open time: {results['open_time']:.2f}s")
        print(f"  Write time: {results['write_time']:.2f}s")
        print(f"  Total time: {results['total_time']:.2f}s")
        print("="*70)
        
    except Exception as e:
        results["total_time"] = time.time() - overall_start
        results["success"] = False
        results["error"] = str(e)
        
        print("\n" + "="*70)
        print("✗ Stacked conversion failed!")
        print(f"  Error: {str(e)}")
        print(f"  Time elapsed: {results['total_time']:.2f}s")
        print("="*70)
    
    return results


def save_results(results, output_dir):
    """Save conversion results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = results.get("mode", "conversion")
    json_file = output_path / f"conversion_{mode}_{timestamp}.json"
    
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_file}")
    return json_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert NetCDF files to Zarr format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conversion Modes:
  individual    Convert each NetCDF to a separate Zarr directory
  stacked       Combine all NetCDF files into a single stacked Zarr

Examples:
  # Convert to individual Zarr files
  %(prog)s individual "data/*.nc" zarr_individual/
  
  # Convert to a single stacked Zarr
  %(prog)s stacked "data/*.nc" zarr_stacked.zarr
  
  # Use h5netcdf engine
  %(prog)s individual "data/*.nc" zarr_data/ --engine h5netcdf
        """,
    )
    
    parser.add_argument(
        "mode",
        choices=["individual", "stacked"],
        help="Conversion mode: 'individual' or 'stacked'"
    )
    parser.add_argument(
        "input_pattern",
        type=str,
        help='Glob pattern for input NetCDF files (e.g., "data/*.nc")'
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output directory (individual mode) or Zarr path (stacked mode)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="netcdf4",
        choices=["netcdf4", "h5netcdf"],
        help="NetCDF reading engine (default: netcdf4)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save conversion results to JSON"
    )
    
    args = parser.parse_args()
    
    # Find files
    nc_files = sorted(glob(args.input_pattern))
    
    if len(nc_files) == 0:
        print(f"\n⚠️  Error: No files found matching pattern: {args.input_pattern}")
        print("Please check the pattern and try again.")
        return 1
    
    print("\n" + "="*70)
    print("NETCDF → ZARR CONVERSION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input pattern: {args.input_pattern}")
    print(f"Files found: {len(nc_files)}")
    
    # Run conversion
    if args.mode == "individual":
        results = convert_individual(nc_files, args.output_path, args.engine)
    else:  # stacked
        results = convert_stacked(nc_files, args.output_path, args.engine)
    
    # Save results if requested
    if args.save_results:
        save_results(results, Path(args.output_path).parent if args.mode == "stacked" else args.output_path)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if results.get("success", results.get("successful", 0) > 0) else 1


if __name__ == "__main__":
    exit(main())

