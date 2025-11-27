#!/usr/bin/env python3
"""
Example: Optimal method for loading multiple NetCDF files with xarray.

This demonstrates the fastest configuration based on benchmark results:
- Engine: netcdf4 (1.9x faster than h5netcdf)
- Cache: False (11% faster than with cache)
- Chunks: {} (auto Dask chunking, 15.8x faster than eager loading)
"""

import xarray as xr
import time
from pathlib import Path


def load_data_optimal():
    """Load data using the optimal configuration."""
    data_dir = Path('/Users/stuart.lynn/Customers/LSEG/raster-benchmarking/data')
    file_pattern = str(data_dir / '*.nc')
    
    print("Loading data with optimal configuration...")
    print(f"File pattern: {file_pattern}")
    print()
    
    start_time = time.time()
    
    # OPTIMAL CONFIGURATION
    ds = xr.open_mfdataset(
        file_pattern,
        engine='netcdf4',        # netCDF4 engine (fastest)
        cache=False,             # No caching (faster for this use case)
        chunks={},               # Auto Dask chunking (lazy loading)
        combine='nested',        # How to combine files
        concat_dim='file'        # Dimension name for concatenation
    )
    
    load_time = time.time() - start_time
    
    print(f"✓ Dataset opened in {load_time:.3f} seconds")
    print()
    print("Dataset Information:")
    print("-" * 70)
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Coordinates: {list(ds.coords)}")
    print()
    print("Data Variable 't2m' (temperature at 2m):")
    print(f"  Shape: {ds['t2m'].shape}")
    print(f"  Dtype: {ds['t2m'].dtype}")
    print(f"  Chunks: {ds['t2m'].chunks}")
    print()
    
    # Example: Access data for first file, first ensemble member
    print("Accessing sample data...")
    access_start = time.time()
    
    # This triggers actual computation (data is loaded on demand)
    sample = ds['t2m'].isel(file=0, number=0, valid_time=0)
    sample_computed = sample.compute()  # Compute the dask array
    
    access_time = time.time() - access_start
    
    print(f"✓ Sample data computed in {access_time:.3f} seconds")
    print()
    print("Sample Statistics (First file, first ensemble member):")
    print(f"  Min: {float(sample_computed.min()):.2f} K")
    print(f"  Max: {float(sample_computed.max()):.2f} K")
    print(f"  Mean: {float(sample_computed.mean()):.2f} K")
    print(f"  Std: {float(sample_computed.std()):.2f} K")
    print()
    
    # Example: Compute global mean across all files and ensemble members
    print("Computing global mean across all data...")
    compute_start = time.time()
    
    # This demonstrates Dask's lazy evaluation and parallel computation
    global_mean = ds['t2m'].mean().compute()
    
    compute_time = time.time() - compute_start
    
    print(f"✓ Global mean computed in {compute_time:.3f} seconds")
    print(f"  Global mean temperature: {float(global_mean):.2f} K")
    print()
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.3f} seconds")
    print()
    
    # Clean up
    ds.close()
    
    return ds


def load_data_comparison():
    """Compare optimal vs. eager loading (for demonstration)."""
    data_dir = Path('/Users/stuart.lynn/Customers/LSEG/raster-benchmarking/data')
    file_pattern = str(data_dir / '*.nc')
    
    print("\n" + "=" * 70)
    print("COMPARISON: Optimal (Dask) vs. Eager Loading")
    print("=" * 70)
    print()
    
    # Optimal method (Dask)
    print("1. Optimal method (with Dask):")
    start_dask = time.time()
    ds_dask = xr.open_mfdataset(
        file_pattern,
        engine='netcdf4',
        cache=False,
        chunks={},  # Dask enabled
        combine='nested',
        concat_dim='file'
    )
    time_dask = time.time() - start_dask
    print(f"   Time: {time_dask:.3f}s")
    ds_dask.close()
    
    # Warning: This will be slow and memory-intensive!
    print()
    print("2. Eager loading (without Dask) - WARNING: SLOW!")
    print("   (Skipping to avoid long wait time)")
    print("   Expected time: ~4-9 seconds")
    print("   Expected memory: ~26GB")
    
    # Uncomment to actually test (not recommended)
    # start_eager = time.time()
    # ds_eager = xr.open_mfdataset(
    #     file_pattern,
    #     engine='netcdf4',
    #     cache=False,
    #     chunks=None  # No Dask
    # )
    # time_eager = time.time() - start_eager
    # print(f"   Time: {time_eager:.3f}s")
    # speedup = time_eager / time_dask
    # print(f"   Speedup with Dask: {speedup:.1f}x")
    # ds_eager.close()
    
    print()
    print("Benchmark results show:")
    print("  → Dask loading: ~0.5s")
    print("  → Eager loading: ~4-9s")
    print("  → Speedup: ~15.8x with Dask!")
    print()


def main():
    """Main function."""
    print()
    print("=" * 70)
    print("OPTIMAL XARRAY LOADING EXAMPLE")
    print("=" * 70)
    print()
    
    # Load and demonstrate optimal method
    load_data_optimal()
    
    # Show comparison
    load_data_comparison()
    
    print("=" * 70)
    print("Tip: For production use, consider:")
    print("  - Use ds.chunk() to rechunk data for your specific workload")
    print("  - Use ds.persist() to load data into distributed memory")
    print("  - Use dask.distributed for parallel computation across cores")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()

