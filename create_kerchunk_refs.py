#!/usr/bin/env python3
"""
Create Kerchunk reference files from NetCDF files.

Generates:
1. Individual JSON reference files for each NetCDF file
2. A combined master.json that virtually concatenates all files
"""

import fsspec
import glob as glob_module
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.combine import MultiZarrToZarr


def create_single_reference(nc_file, output_dir, inline_threshold=100):
    """
    Create a Kerchunk reference for a single NetCDF file.
    
    Parameters:
    -----------
    nc_file : str
        Path to the NetCDF file
    output_dir : Path
        Output directory for the reference file
    inline_threshold : int
        Threshold for inlining small data chunks
        
    Returns:
    --------
    tuple: (reference_file_path, success, error_message)
    """
    nc_path = Path(nc_file)
    ref_file = output_dir / f"{nc_path.stem}.json"
    
    try:
        with fsspec.open(nc_file, mode='rb') as f:
            h5chunks = SingleHdf5ToZarr(f, nc_file, inline_threshold=inline_threshold)
            refs = h5chunks.translate()
        
        with open(ref_file, 'w') as f:
            json.dump(refs, f)
        
        return str(ref_file), True, None
        
    except Exception as e:
        return str(ref_file), False, str(e)


def create_references(
    nc_pattern,
    output_dir,
    concat_dims=None,
    identical_dims=None,
    inline_threshold=100,
    skip_combine=False,
):
    """
    Create Kerchunk references for multiple NetCDF files.
    
    Parameters:
    -----------
    nc_pattern : str
        Glob pattern for NetCDF files
    output_dir : str or Path
        Output directory for reference files
    concat_dims : list
        Dimensions to concatenate along (default: ['valid_time'])
    identical_dims : list
        Dimensions that are identical across files (default: ['number', 'latitude', 'longitude'])
    inline_threshold : int
        Threshold for inlining small data chunks
    skip_combine : bool
        If True, skip creating the combined master.json
        
    Returns:
    --------
    dict with timing and results
    """
    if concat_dims is None:
        concat_dims = ['valid_time']
    if identical_dims is None:
        identical_dims = ['number', 'latitude', 'longitude']
    
    print("\n" + "=" * 70)
    print("KERCHUNK REFERENCE GENERATION")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nc_files = sorted(glob_module.glob(nc_pattern))
    print(f"Input pattern: {nc_pattern}")
    print(f"Files found: {len(nc_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Concat dimensions: {concat_dims}")
    print(f"Identical dimensions: {identical_dims}")
    
    if len(nc_files) == 0:
        print("\n⚠️  No files found matching pattern!")
        return {
            "success": False,
            "error": "No files found",
            "total_files": 0,
        }
    
    results = {
        "total_files": len(nc_files),
        "successful": 0,
        "failed": 0,
        "file_times": [],
        "errors": [],
    }
    
    overall_start = time.time()
    
    # Step 1: Create individual references
    print("\n" + "-" * 70)
    print("Step 1: Creating individual reference files...")
    print("-" * 70)
    
    reference_files = []
    
    for i, nc_file in enumerate(nc_files, 1):
        file_start = time.time()
        nc_name = Path(nc_file).name
        
        ref_file, success, error = create_single_reference(
            nc_file, output_dir, inline_threshold
        )
        
        file_time = time.time() - file_start
        
        if success:
            reference_files.append(ref_file)
            results["successful"] += 1
            results["file_times"].append({
                "file": nc_name,
                "time": file_time,
                "success": True,
            })
            if i % 10 == 0 or i == len(nc_files):
                print(f"  ✓ Created {i}/{len(nc_files)} references...")
        else:
            results["failed"] += 1
            results["file_times"].append({
                "file": nc_name,
                "time": file_time,
                "success": False,
                "error": error,
            })
            results["errors"].append({
                "file": nc_name,
                "error": error,
            })
            print(f"  ✗ Failed: {nc_name} - {error}")
    
    results["individual_time"] = time.time() - overall_start
    print(f"\n  Created {len(reference_files)} reference files in {results['individual_time']:.2f}s")
    
    # Step 2: Combine references
    if not skip_combine and len(reference_files) > 0:
        print("\n" + "-" * 70)
        print("Step 2: Combining references into master.json...")
        print("-" * 70)
        
        combine_start = time.time()
        
        try:
            mzz = MultiZarrToZarr(
                reference_files,
                concat_dims=concat_dims,
                identical_dims=identical_dims,
                coo_map={concat_dims[0]: 'INDEX'} if concat_dims else {},
            )
            
            combined_refs = mzz.translate()
            
            combined_file = output_dir / 'master.json'
            with open(combined_file, 'w') as f:
                json.dump(combined_refs, f)
            
            results["combine_time"] = time.time() - combine_start
            results["combined_file"] = str(combined_file)
            results["combine_success"] = True
            
            print(f"  ✓ Combined reference saved to: {combined_file}")
            print(f"  ✓ Combine time: {results['combine_time']:.2f}s")
            
        except Exception as e:
            results["combine_time"] = time.time() - combine_start
            results["combine_success"] = False
            results["combine_error"] = str(e)
            print(f"  ✗ Failed to combine references: {str(e)}")
    else:
        if skip_combine:
            print("\n  Skipping combine step (--skip-combine flag)")
        results["combine_success"] = None
    
    results["total_time"] = time.time() - overall_start
    results["success"] = results["successful"] > 0
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total files: {results['total_files']}")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Individual refs time: {results['individual_time']:.2f}s")
    if 'combine_time' in results:
        print(f"  Combine time: {results['combine_time']:.2f}s")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Avg time per file: {results['individual_time']/len(nc_files):.3f}s")
    print("=" * 70)
    
    return results


def save_results(results, output_dir):
    """Save generation results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output_path / f"kerchunk_generation_{timestamp}.json"
    
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_file}")
    return json_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create Kerchunk reference files from NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - create references and master.json
  %(prog)s "data/*.nc" references/
  
  # Only create individual references (skip master.json)
  %(prog)s "data/*.nc" references/ --skip-combine
  
  # Custom concatenation dimensions
  %(prog)s "data/*.nc" refs/ --concat-dims valid_time --identical-dims latitude longitude
  
  # Save generation results to JSON
  %(prog)s "data/*.nc" references/ --save-results
        """,
    )
    
    parser.add_argument(
        "input_pattern",
        type=str,
        help='Glob pattern for input NetCDF files (e.g., "data/*.nc")'
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for reference files"
    )
    parser.add_argument(
        "--concat-dims",
        type=str,
        nargs="+",
        default=["valid_time"],
        help="Dimensions to concatenate along (default: valid_time)"
    )
    parser.add_argument(
        "--identical-dims",
        type=str,
        nargs="+",
        default=["number", "latitude", "longitude"],
        help="Dimensions that are identical across files (default: number latitude longitude)"
    )
    parser.add_argument(
        "--inline-threshold",
        type=int,
        default=100,
        help="Threshold for inlining small data chunks (default: 100)"
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip creating the combined master.json"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save generation results to JSON"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("NETCDF → KERCHUNK REFERENCE GENERATION")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run generation
    results = create_references(
        nc_pattern=args.input_pattern,
        output_dir=args.output_dir,
        concat_dims=args.concat_dims,
        identical_dims=args.identical_dims,
        inline_threshold=args.inline_threshold,
        skip_combine=args.skip_combine,
    )
    
    # Save results if requested
    if args.save_results:
        save_results(results, args.output_dir)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if results.get("success", False) else 1


if __name__ == "__main__":
    exit(main())

