#!/usr/bin/env python3
"""
Benchmark different methods of loading Kerchunk reference files with xarray.

Tests combinations of:
- Individual JSON references vs combined master.json
- Different chunk configurations
- Caching options
"""

import xarray as xr
import fsspec
import time
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import argparse
from glob import glob
import warnings


def open_kerchunk_single(reference_file, chunks=None, remote_protocol="file"):
    """
    Open a dataset from a single Kerchunk JSON reference file.

    Parameters:
    -----------
    reference_file : str
        Path to the Kerchunk JSON reference file
    chunks : dict, None, or 'auto'
        Chunking strategy for dask
    remote_protocol : str
        Protocol for accessing the underlying data files (e.g., 'file', 'abfss')

    Returns:
    --------
    xarray.Dataset
    """
    with open(reference_file) as f:
        refs = json.load(f)

    fs = fsspec.filesystem(
        "reference",
        fo=refs,
        remote_protocol=remote_protocol,
    )

    mapper = fs.get_mapper("")

    # Build open_dataset kwargs
    open_kwargs = {
        "engine": "zarr",
        "backend_kwargs": {"consolidated": False},
    }
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    ds = xr.open_dataset(mapper, **open_kwargs)
    return ds


def open_kerchunk_multiple(reference_files, chunks=None, remote_protocol="file", concat_dim="valid_time"):
    """
    Open and combine multiple Kerchunk JSON reference files.

    Parameters:
    -----------
    reference_files : list
        List of paths to Kerchunk JSON reference files
    chunks : dict, None, or 'auto'
        Chunking strategy for dask
    remote_protocol : str
        Protocol for accessing the underlying data files
    concat_dim : str
        Dimension to concatenate along

    Returns:
    --------
    xarray.Dataset
    """
    datasets = []
    for ref_file in sorted(reference_files):
        ds = open_kerchunk_single(ref_file, chunks=chunks, remote_protocol=remote_protocol)
        datasets.append(ds)

    combined = xr.concat(datasets, dim=concat_dim)
    return combined


def benchmark_kerchunk_loading(
    reference_path,
    chunks=None,
    cache=True,
    is_combined=False,
    remote_protocol="file",
    concat_dim="valid_time",
    description="",
    suppress_warnings=False,
):
    """
    Benchmark loading Kerchunk references with xarray.

    Parameters:
    -----------
    reference_path : str
        Path to master.json (if is_combined=True) or glob pattern for individual JSONs
    chunks : dict, None, or 'auto'
        Chunking strategy for dask
        - None: No rechunking (use native chunks)
        - {}: Auto rechunking with dask
        - 'auto': Auto rechunking with dask
        - dict: Specific chunk sizes
    cache : bool
        Whether to cache the data (note: mainly affects xarray caching behavior)
    is_combined : bool
        True if loading a combined master.json, False for multiple individual JSONs
    remote_protocol : str
        Protocol for accessing the underlying data files
    concat_dim : str
        Dimension to concatenate along (for multiple files)
    description : str
        Description of the test
    suppress_warnings : bool
        Whether to suppress chunk alignment warnings

    Returns:
    --------
    dict with timing results and metadata
    """
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"  Path: {reference_path}")
    print(f"  Mode: {'Combined master.json' if is_combined else 'Individual JSON references'}")
    print(f"  Chunks: {chunks}")
    print(f"  Remote protocol: {remote_protocol}")
    print(f"{'='*70}")

    # Start timing
    start_time = time.time()

    try:
        # Suppress chunk alignment warnings if requested
        if suppress_warnings:
            warnings.filterwarnings('ignore', message='.*chunks separate the stored chunks.*')

        # Open the dataset
        if is_combined:
            # Single combined master.json
            ds = open_kerchunk_single(
                reference_path,
                chunks=chunks,
                remote_protocol=remote_protocol,
            )
        else:
            # Multiple individual JSON references
            if isinstance(reference_path, str):
                reference_files = sorted(glob(reference_path))
                # Exclude master.json from individual file loading
                reference_files = [f for f in reference_files if not f.endswith('master.json')]
            else:
                reference_files = reference_path

            if len(reference_files) == 0:
                raise ValueError(f"No reference files found matching: {reference_path}")

            print(f"  Found {len(reference_files)} reference files")

            ds = open_kerchunk_multiple(
                reference_files,
                chunks=chunks,
                remote_protocol=remote_protocol,
                concat_dim=concat_dim,
            )

        open_time = time.time() - start_time
        print(f"✓ Opened dataset in {open_time:.3f} seconds")

        # Get dataset info
        dataset_info = {
            "shape": dict(ds.sizes),
            "variables": list(ds.data_vars),
            "chunks": str(ds.chunks) if hasattr(ds, 'chunks') and ds.chunks else "None",
        }
        print(f"  Dataset shape: {dataset_info['shape']}")
        print(f"  Variables: {dataset_info['variables']}")
        print(f"  Chunks: {dataset_info['chunks']}")

        # Test accessing data
        access_start = time.time()

        # Find the appropriate dimensions for indexing
        dims = ds.dims
        time_dim = 'valid_time' if 'valid_time' in dims else list(dims)[0]

        # Build isel kwargs based on available dimensions
        isel_kwargs = {}
        if 'number' in dims:
            isel_kwargs['number'] = 0
        if time_dim in dims:
            isel_kwargs[time_dim] = 0
        if 'latitude' in dims:
            lat_idx = min(180, ds.sizes['latitude'] - 1)
            isel_kwargs['latitude'] = lat_idx
        if 'longitude' in dims:
            lon_idx = min(360, ds.sizes['longitude'] - 1)
            isel_kwargs['longitude'] = lon_idx

        # Get first data variable
        data_var = list(ds.data_vars)[0]
        sample_value = float(ds[data_var].isel(**isel_kwargs).compute())

        access_time = time.time() - access_start
        print(f"✓ Accessed sample data in {access_time:.3f} seconds")
        print(f"  Sample value: {sample_value:.2f}")

        # Close dataset
        ds.close()

        # Reset warning filters
        if suppress_warnings:
            warnings.resetwarnings()

        total_time = time.time() - start_time
        print(f"✓ Total time: {total_time:.3f} seconds")

        return {
            "description": description,
            "is_combined": is_combined,
            "chunks": str(chunks),
            "remote_protocol": remote_protocol,
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
        import traceback
        traceback.print_exc()
        return {
            "description": description,
            "is_combined": is_combined,
            "chunks": str(chunks),
            "remote_protocol": remote_protocol,
            "open_time": None,
            "access_time": None,
            "total_time": error_time,
            "success": False,
            "error": str(e),
            "dataset_info": None,
        }


def run_benchmarks(
    reference_path,
    is_combined=False,
    remote_protocol="file",
    concat_dim="valid_time",
    output_dir=".",
):
    """Run comprehensive benchmarks.

    Parameters:
    -----------
    reference_path : str
        Path to master.json (if is_combined=True) or glob pattern for individual JSONs
    is_combined : bool
        True if single combined reference, False for multiple individual references
    remote_protocol : str
        Protocol for accessing the underlying data files
    concat_dim : str
        Dimension to concatenate along (for multiple files)
    output_dir : str
        Directory where results will be saved
    """
    print("\n" + "=" * 70)
    print("XARRAY KERCHUNK LOADING BENCHMARK")
    print("=" * 70)
    print(f"Reference path: {reference_path}")
    print(f"Mode: {'Combined master.json' if is_combined else 'Individual JSON references'}")
    print(f"Remote protocol: {remote_protocol}")

    # Count files if multi-file mode
    if not is_combined:
        reference_files = glob(reference_path)
        # Exclude master.json from individual file count
        reference_files = [f for f in reference_files if not f.endswith('master.json')]
        print(f"Number of reference files: {len(reference_files)}")

        if len(reference_files) == 0:
            print(f"\n⚠️  Warning: No reference files found matching pattern: {reference_path}")
            print("Please check the pattern and try again.")
            return []

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Test configurations
    if is_combined:
        # Combined master.json tests
        test_configs = [
            {
                "chunks": None,
                "description": "Combined master.json, native chunks",
            },
            {
                "chunks": {},
                "description": "Combined master.json, auto rechunking",
            },
            {
                "chunks": {"number": 7, "valid_time": -1, "latitude": 184, "longitude": 360},
                "description": "Combined master.json, time-optimized chunks",
                "suppress_warnings": True,
            },
            {
                "chunks": {"number": -1, "valid_time": 1, "latitude": 184, "longitude": 360},
                "description": "Combined master.json, ensemble-optimized chunks",
                "suppress_warnings": True,
            },
        ]
    else:
        # Individual JSON references tests
        test_configs = [
            {
                "chunks": None,
                "description": "Individual JSONs, native chunks",
            },
            {
                "chunks": {},
                "description": "Individual JSONs, auto rechunking",
            },
            {
                "chunks": {"number": 7, "latitude": 184, "longitude": 360},
                "description": "Individual JSONs, custom chunks",
                "suppress_warnings": True,
            },
        ]

    # Run each test
    for i, config in enumerate(test_configs, 1):
        print(f"\n\nTest {i}/{len(test_configs)}")

        # For individual files, exclude master.json
        if not is_combined:
            ref_path = reference_path
        else:
            ref_path = reference_path

        result = benchmark_kerchunk_loading(
            ref_path,
            chunks=config["chunks"],
            cache=True,
            is_combined=is_combined,
            remote_protocol=remote_protocol,
            concat_dim=concat_dim,
            description=config["description"],
            suppress_warnings=config.get("suppress_warnings", False),
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
        Label for the mode (e.g., "combined" or "individual")
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
    csv_file = output_path / f"benchmark_kerchunk_{mode_label}_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n\n{'='*70}")
    print(f"Results saved to: {csv_file}")

    # Save detailed results to JSON
    json_file = output_path / f"benchmark_kerchunk_{mode_label}_{timestamp}.json"
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
        description="Benchmark different methods of loading Kerchunk reference files with xarray.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark combined master.json
  %(prog)s --combined /path/to/references/master.json
  
  # Benchmark individual JSON references
  %(prog)s "/path/to/references/*.json"
  
  # With custom output directory
  %(prog)s --combined /path/to/master.json -o results/
  
  # For remote data (e.g., Azure Blob Storage)
  %(prog)s --combined /path/to/master.json --remote-protocol abfss
        """,
    )
    parser.add_argument(
        "reference_path",
        type=str,
        help='Path to master.json (with --combined) or glob pattern for individual JSONs',
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Treat as combined master.json (default: individual JSON references)"
    )
    parser.add_argument(
        "--remote-protocol",
        type=str,
        default="file",
        help="Protocol for accessing underlying data files (default: file). Use 'abfss' for Azure, 's3' for AWS, etc."
    )
    parser.add_argument(
        "--concat-dim",
        type=str,
        default="valid_time",
        help="Dimension to concatenate along for individual files (default: valid_time)"
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
    is_combined = args.combined

    # Validate path
    if is_combined:
        if not Path(args.reference_path).exists():
            print(f"\n⚠️  Error: Reference file not found: {args.reference_path}")
            return 1
    else:
        # Check if any files match the pattern (excluding master.json)
        matches = [f for f in glob(args.reference_path) if not f.endswith('master.json')]
        if len(matches) == 0:
            print(f"\n⚠️  Error: No reference files found matching: {args.reference_path}")
            return 1

    # Run benchmarks
    mode_label = "combined" if is_combined else "individual"
    results = run_benchmarks(
        args.reference_path,
        is_combined=is_combined,
        remote_protocol=args.remote_protocol,
        concat_dim=args.concat_dim,
        output_dir=args.output_dir,
    )

    # Save and display results
    save_and_display_results(results, args.output_dir, mode_label)

    return 0


if __name__ == "__main__":
    exit(main())

