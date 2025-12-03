#!/usr/bin/env python3
"""
Benchmark different methods of loading Zarr files with xarray.

Tests combinations of:
- Single stacked Zarr vs multiple Zarr directories
- Different chunk configurations
- Caching options
"""

import xarray as xr
import time
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import argparse
from glob import glob
import warnings


def benchmark_zarr_loading(
    zarr_path, chunks=None, cache=True, is_single=False, description="", suppress_warnings=False
):
    """
    Benchmark loading Zarr with xarray.

    Parameters:
    -----------
    zarr_path : str or list
        Path to Zarr directory (single) or glob pattern/list (multiple)
    chunks : dict, None, or 'auto'
        Chunking strategy for dask
        - None: No rechunking (use native Zarr chunks)
        - {}: Auto rechunking with dask
        - 'auto': Auto rechunking with dask
        - dict: Specific chunk sizes
    cache : bool
        Whether to cache the data
    is_single : bool
        True if loading a single stacked Zarr, False for multiple Zarrs
    description : str
        Description of the test
    suppress_warnings : bool
        Whether to suppress chunk alignment warnings (for tests that intentionally rechunk)

    Returns:
    --------
    dict with timing results and metadata
    """
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"  Path: {zarr_path if is_single else f'{zarr_path} (pattern)'}")
    print(f"  Cache: {cache}")
    print(f"  Chunks: {chunks}")
    print(f"{'='*70}")

    # Start timing
    start_time = time.time()

    try:
        # Suppress chunk alignment warnings if requested (for tests that intentionally rechunk)
        if suppress_warnings:
            warnings.filterwarnings('ignore', message='.*chunks separate the stored chunks.*')
        
        # Open the Zarr dataset
        if is_single:
            # Single stacked Zarr
            ds = xr.open_zarr(
                zarr_path,
                chunks=chunks,
                consolidated=False  # May not have consolidated metadata
            )
        else:
            # Multiple Zarr directories
            if isinstance(zarr_path, str):
                zarr_files = sorted(glob(zarr_path))
            else:
                zarr_files = zarr_path
                
            ds = xr.open_mfdataset(
                zarr_files,
                engine='zarr',
                chunks=chunks,
                combine="nested",
                concat_dim="valid_time",
                parallel=True
            )

        open_time = time.time() - start_time
        print(f"✓ Opened dataset in {open_time:.3f} seconds")

        # Get dataset info
        dataset_info = {
            "shape": dict(ds.sizes),
            "variables": list(ds.data_vars),
            "chunks": str(ds.chunks) if hasattr(ds, 'chunks') else "None",
        }
        print(f"  Dataset shape: {dataset_info['shape']}")
        print(f"  Variables: {dataset_info['variables']}")
        print(f"  Chunks: {dataset_info['chunks']}")

        # Test accessing data
        access_start = time.time()
        
        # Determine coordinate names (handle both single and multi-file cases)
        if 'file' in ds.dims:
            # Multi-file case
            sample_value = float(
                ds["t2m"]
                .isel(file=0, number=0, valid_time=0, latitude=180, longitude=360)
                .compute()
            )
        else:
            # Single stacked Zarr case
            time_dim = 'valid_time' if 'valid_time' in ds.dims else list(ds.dims)[0]
            sample_value = float(
                ds["t2m"]
                .isel({
                    'number': 0,
                    time_dim: 0,
                    'latitude': 180,
                    'longitude': 360
                })
                .compute()
            )
        
        access_time = time.time() - access_start
        print(f"✓ Accessed sample data in {access_time:.3f} seconds")
        print(f"  Sample value: {sample_value:.2f} K")

        # Close dataset
        ds.close()

        # Reset warning filters
        if suppress_warnings:
            warnings.resetwarnings()

        total_time = time.time() - start_time
        print(f"✓ Total time: {total_time:.3f} seconds")

        return {
            "description": description,
            "is_single": is_single,
            "cache": cache,
            "chunks": str(chunks),
            "open_time": open_time,
            "access_time": access_time,
            "total_time": total_time,
            "success": True,
            "error": None,
            "dataset_info": dataset_info,
        }

    except Exception as e:
        # Reset warning filters
        if suppress_warnings:
            warnings.resetwarnings()
            
        error_time = time.time() - start_time
        print(f"✗ Error after {error_time:.3f} seconds: {str(e)}")
        return {
            "description": description,
            "is_single": is_single,
            "cache": cache,
            "chunks": str(chunks),
            "open_time": None,
            "access_time": None,
            "total_time": error_time,
            "success": False,
            "error": str(e),
            "dataset_info": None,
        }


def run_benchmarks(zarr_path, is_single=False, output_dir="."):
    """Run comprehensive benchmarks.

    Parameters:
    -----------
    zarr_path : str
        Path to Zarr (single directory or glob pattern for multiple)
    is_single : bool
        True if single stacked Zarr, False for multiple Zarrs
    output_dir : str
        Directory where results will be saved
    """
    print("\n" + "=" * 70)
    print("XARRAY ZARR LOADING BENCHMARK")
    print("=" * 70)
    print(f"Zarr path: {zarr_path}")
    print(f"Mode: {'Single stacked Zarr' if is_single else 'Multiple Zarr directories'}")

    # Count files/directories if multi-file
    if not is_single:
        zarr_files = glob(zarr_path)
        print(f"Number of Zarr directories: {len(zarr_files)}")
        
        if len(zarr_files) == 0:
            print(f"\n⚠️  Warning: No Zarr directories found matching pattern: {zarr_path}")
            print("Please check the pattern and try again.")
            return []

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Test configurations
    if is_single:
        # Single stacked Zarr tests
        test_configs = [
            {
                "chunks": None,
                "cache": True,
                "description": "Native Zarr chunks, with cache",
            },
            {
                "chunks": None,
                "cache": False,
                "description": "Native Zarr chunks, no cache",
            },
            {
                "chunks": {},
                "cache": True,
                "description": "Auto rechunking, with cache",
            },
            {
                "chunks": {"number": 7, "valid_time": -1, "latitude": 184, "longitude": 360},
                "cache": True,
                "description": "Time-optimized rechunking (all time loaded), with cache",
                "suppress_warnings": True  # Intentional rechunking for this use case
            },
        ]
    else:
        # Multiple Zarr directories tests
        test_configs = [
            {
                "chunks": None,
                "cache": True,
                "description": "Native Zarr chunks, with cache",
            },
            {
                "chunks": None,
                "cache": False,
                "description": "Native Zarr chunks, no cache",
            },
            {
                "chunks": {},
                "cache": True,
                "description": "Auto rechunking, with cache",
            },
            # Note: Explicit chunks cause warnings due to irregular boundary chunks
            # Only included to test rechunking performance
        ]

    # Run each test
    for i, config in enumerate(test_configs, 1):
        print(f"\n\nTest {i}/{len(test_configs)}")
        result = benchmark_zarr_loading(
            zarr_path,
            chunks=config["chunks"],
            cache=config["cache"],
            is_single=is_single,
            description=config["description"],
            suppress_warnings=config.get("suppress_warnings", False)
        )
        results.append(result)

        # Small delay between tests
        time.sleep(1)

    return results


def save_and_display_results(results, output_dir=".", mode_label=""):
    """Save results and display summary.

    Parameters:
    -----------
    results : list
        List of benchmark results
    output_dir : str
        Directory where results will be saved
    mode_label : str
        Label for the mode (e.g., "single" or "multiple")
    """
    if not results:
        print("\n⚠️  No results to save.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_path / f"benchmark_zarr_{mode_label}_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n\n{'='*70}")
    print(f"Results saved to: {csv_file}")

    # Save detailed results to JSON
    json_file = output_path / f"benchmark_zarr_{mode_label}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {json_file}")

    # Display summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    # Filter successful tests
    successful = df[df["success"]]

    if len(successful) > 0:
        # Sort by total time
        successful_sorted = successful.sort_values("total_time")

        print("Results (sorted by total time):\n")
        print(f"{'Rank':<5} {'Total Time':<12} {'Open Time':<12} {'Description':<50}")
        print("-" * 79)

        for idx, (_, row) in enumerate(successful_sorted.iterrows(), 1):
            print(
                f"{idx:<5} {row['total_time']:>8.3f}s    "
                f"{row['open_time']:>8.3f}s    {row['description'][:50]}"
            )

        # Show winner
        winner = successful_sorted.iloc[0]
        print(f"\n{'='*70}")
        print("🏆 FASTEST METHOD:")
        print(f"{'='*70}")
        print(f"  Description: {winner['description']}")
        print(f"  Chunks: {winner['chunks']}")
        print(f"  Cache: {winner['cache']}")
        print(f"  Open time: {winner['open_time']:.3f}s")
        print(f"  Access time: {winner['access_time']:.3f}s")
        print(f"  Total time: {winner['total_time']:.3f}s")

    # Show failed tests
    failed = df[~df["success"]]
    if len(failed) > 0:
        print(f"\n{'='*70}")
        print(f"FAILED TESTS ({len(failed)}):")
        print(f"{'='*70}")
        for _, row in failed.iterrows():
            print(f"\n  {row['description']}")
            print(f"    Error: {row['error']}")

    print(f"\n{'='*70}")
    print(f"Benchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark different methods of loading Zarr files with xarray.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark single stacked Zarr
  %(prog)s --single zarr_stacked.zarr
  
  # Benchmark multiple Zarr directories
  %(prog)s "zarr_individual/*.zarr"
  
  # With custom output directory
  %(prog)s --single zarr_data.zarr -o results/
        """,
    )
    parser.add_argument(
        "zarr_path",
        type=str,
        help='Path to Zarr (single directory) or glob pattern (multiple directories)',
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Treat as single stacked Zarr (default: multiple Zarr directories)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Directory where benchmark results will be saved (default: current directory)",
    )

    args = parser.parse_args()

    # Determine mode
    is_single = args.single
    
    # Validate path
    if is_single:
        if not Path(args.zarr_path).exists():
            print(f"\n⚠️  Error: Zarr directory not found: {args.zarr_path}")
            return 1
    else:
        # Check if any files match the pattern
        matches = glob(args.zarr_path)
        if len(matches) == 0:
            print(f"\n⚠️  Error: No Zarr directories found matching: {args.zarr_path}")
            return 1

    # Run benchmarks
    mode_label = "single" if is_single else "multiple"
    results = run_benchmarks(args.zarr_path, is_single, args.output_dir)

    # Save and display results
    save_and_display_results(results, args.output_dir, mode_label)
    
    return 0


if __name__ == "__main__":
    exit(main())

