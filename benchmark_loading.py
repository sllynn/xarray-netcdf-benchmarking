#!/usr/bin/env python3
"""
Benchmark different methods of loading NetCDF files with xarray.

Tests combinations of:
- Engine: netcdf4 vs h5netcdf (h5py backend)
- Caching: True vs False
- Dask: None (no dask) vs {} (auto chunks) vs specific chunk sizes
"""

import xarray as xr
import time
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import argparse
from glob import glob


def benchmark_loading(
    file_pattern, engine="netcdf4", cache=True, chunks=None, description=""
):
    """
    Benchmark loading files with xarray.open_mfdataset.

    Parameters:
    -----------
    file_pattern : str
        Glob pattern for files to load
    engine : str
        Engine to use ('netcdf4' or 'h5netcdf')
    cache : bool
        Whether to cache the data
    chunks : dict, None, or 'auto'
        Chunking strategy for dask
        - None: No dask (load into memory)
        - {}: Auto chunking with dask
        - 'auto': Auto chunking with dask
        - dict: Specific chunk sizes
    description : str
        Description of the test

    Returns:
    --------
    dict with timing results and metadata
    """
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"  Engine: {engine}")
    print(f"  Cache: {cache}")
    print(f"  Chunks: {chunks}")
    print(f"{'='*70}")

    # Start timing
    start_time = time.time()

    try:
        # Open the multi-file dataset
        # For h5netcdf: use phony_dims='sort' to handle dimension scale issues
        backend_kwargs = {}
        if engine == "h5netcdf":
            backend_kwargs = {"phony_dims": "sort"}
        
        ds = xr.open_mfdataset(
            file_pattern,
            engine=engine,
            cache=cache,
            chunks=chunks,
            combine="nested",
            concat_dim="file",
            backend_kwargs=backend_kwargs,
        )

        open_time = time.time() - start_time
        print(f"✓ Opened dataset in {open_time:.3f} seconds")

        # Get dataset info
        dataset_info = {
            "shape": dict(ds.sizes),
            "variables": list(ds.data_vars),
            "chunks": str(ds.chunks) if chunks is not None else "None (eager)",
        }
        print(f"  Dataset shape: {dataset_info['shape']}")
        print(f"  Variables: {dataset_info['variables']}")
        print(f"  Chunks: {dataset_info['chunks']}")

        # Test accessing data (trigger computation if using dask)
        access_start = time.time()
        if chunks is None:
            # Data is already in memory
            sample_value = float(ds["t2m"].values[0, 0, 0, 180, 360])
        else:
            # Using dask - trigger computation on a small sample
            sample_value = float(
                ds["t2m"]
                .isel(file=0, number=0, valid_time=0, latitude=180, longitude=360)
                .compute()
            )
        access_time = time.time() - access_start
        print(f"✓ Accessed sample data in {access_time:.3f} seconds")
        print(f"  Sample value: {sample_value:.2f} K")

        # Close dataset
        ds.close()

        total_time = time.time() - start_time
        print(f"✓ Total time: {total_time:.3f} seconds")

        return {
            "description": description,
            "engine": engine,
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
        error_time = time.time() - start_time
        print(f"✗ Error after {error_time:.3f} seconds: {str(e)}")
        return {
            "description": description,
            "engine": engine,
            "cache": cache,
            "chunks": str(chunks),
            "open_time": None,
            "access_time": None,
            "total_time": error_time,
            "success": False,
            "error": str(e),
            "dataset_info": None,
        }


def run_benchmarks(file_pattern, output_dir="."):
    """Run comprehensive benchmarks.

    Parameters:
    -----------
    file_pattern : str
        Glob pattern for files to benchmark (e.g., 'data/*.nc')
    output_dir : str
        Directory where results will be saved
    """
    print("\n" + "=" * 70)
    print("XARRAY LOADING BENCHMARK")
    print("=" * 70)
    print(f"File pattern: {file_pattern}")

    # Count files
    nc_files = glob(file_pattern)
    print(f"Number of files: {len(nc_files)}")

    if len(nc_files) == 0:
        print(f"\n⚠️  Warning: No files found matching pattern: {file_pattern}")
        print("Please check the file pattern and try again.")
        return []

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Test configurations
    test_configs = [
        # Test 1: netcdf4 engine, no dask, with cache
        {
            "engine": "netcdf4",
            "cache": True,
            "chunks": None,
            "description": "netCDF4 engine, no Dask, with cache",
        },
        # Test 2: netcdf4 engine, no dask, no cache
        {
            "engine": "netcdf4",
            "cache": False,
            "chunks": None,
            "description": "netCDF4 engine, no Dask, no cache",
        },
        # Test 3: netcdf4 engine, auto dask chunks, with cache
        {
            "engine": "netcdf4",
            "cache": True,
            "chunks": {},
            "description": "netCDF4 engine, auto Dask chunks, with cache",
        },
        # Test 4: netcdf4 engine, auto dask chunks, no cache
        {
            "engine": "netcdf4",
            "cache": False,
            "chunks": {},
            "description": "netCDF4 engine, auto Dask chunks, no cache",
        },
        # Test 5: h5netcdf engine, no dask, with cache
        {
            "engine": "h5netcdf",
            "cache": True,
            "chunks": None,
            "description": "h5netcdf engine (h5py), no Dask, with cache",
        },
        # Test 6: h5netcdf engine, no dask, no cache
        {
            "engine": "h5netcdf",
            "cache": False,
            "chunks": None,
            "description": "h5netcdf engine (h5py), no Dask, no cache",
        },
        # Test 7: h5netcdf engine, auto dask chunks, with cache
        {
            "engine": "h5netcdf",
            "cache": True,
            "chunks": {},
            "description": "h5netcdf engine (h5py), auto Dask chunks, with cache",
        },
        # Test 8: h5netcdf engine, auto dask chunks, no cache
        {
            "engine": "h5netcdf",
            "cache": False,
            "chunks": {},
            "description": "h5netcdf engine (h5py), auto Dask chunks, no cache",
        },
        # Test 9: netcdf4 with specific chunks
        {
            "engine": "netcdf4",
            "cache": True,
            "chunks": {"number": 10, "latitude": 100, "longitude": 100},
            "description": "netCDF4 engine, specific chunks (10x100x100), with cache",
        },
        # Test 10: h5netcdf with specific chunks
        {
            "engine": "h5netcdf",
            "cache": True,
            "chunks": {"number": 10, "latitude": 100, "longitude": 100},
            "description": "h5netcdf engine (h5py), specific chunks (10x100x100), with cache",
        },
    ]

    # Run each test
    for i, config in enumerate(test_configs, 1):
        print(f"\n\nTest {i}/{len(test_configs)}")
        result = benchmark_loading(
            file_pattern,
            engine=config["engine"],
            cache=config["cache"],
            chunks=config["chunks"],
            description=config["description"],
        )
        results.append(result)

        # Small delay between tests
        time.sleep(1)

    return results


def save_and_display_results(results, output_dir="."):
    """Save results and display summary.

    Parameters:
    -----------
    results : list
        List of benchmark results
    output_dir : str
        Directory where results will be saved
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
    csv_file = output_path / f"benchmark_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n\n{'='*70}")
    print(f"Results saved to: {csv_file}")

    # Save detailed results to JSON
    json_file = output_path / f"benchmark_results_{timestamp}.json"
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
        print(f"  Engine: {winner['engine']}")
        print(f"  Cache: {winner['cache']}")
        print(f"  Chunks: {winner['chunks']}")
        print(f"  Open time: {winner['open_time']:.3f}s")
        print(f"  Access time: {winner['access_time']:.3f}s")
        print(f"  Total time: {winner['total_time']:.3f}s")

        # Compare with netcdf4
        netcdf4_tests = successful[successful["engine"] == "netcdf4"]
        h5netcdf_tests = successful[successful["engine"] == "h5netcdf"]

        if len(netcdf4_tests) > 0 and len(h5netcdf_tests) > 0:
            best_netcdf4 = netcdf4_tests.loc[netcdf4_tests["total_time"].idxmin()]
            best_h5netcdf = h5netcdf_tests.loc[h5netcdf_tests["total_time"].idxmin()]

            print(f"\n{'='*70}")
            print("ENGINE COMPARISON:")
            print(f"{'='*70}")
            print(f"  Best netCDF4: {best_netcdf4['total_time']:.3f}s")
            print(f"  Best h5netcdf: {best_h5netcdf['total_time']:.3f}s")

            if best_h5netcdf["total_time"] < best_netcdf4["total_time"]:
                speedup = best_netcdf4["total_time"] / best_h5netcdf["total_time"]
                print(f"  → h5netcdf is {speedup:.2f}x faster!")
            else:
                speedup = best_h5netcdf["total_time"] / best_netcdf4["total_time"]
                print(f"  → netCDF4 is {speedup:.2f}x faster!")

    # Show failed tests
    failed = df[df["success"]]
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
        description="Benchmark different methods of loading NetCDF files with xarray.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "data/*.nc"
  %(prog)s "data/*.nc" --output-dir results/
  %(prog)s "/path/to/files/*.nc" -o benchmarks/
        """,
    )
    parser.add_argument(
        "file_pattern",
        type=str,
        help='Glob pattern for NetCDF files to benchmark (e.g., "data/*.nc")',
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Directory where benchmark results will be saved (default: current directory)",
    )

    args = parser.parse_args()

    # Run benchmarks
    results = run_benchmarks(args.file_pattern, args.output_dir)

    # Save and display results
    save_and_display_results(results, args.output_dir)


if __name__ == "__main__":
    main()
