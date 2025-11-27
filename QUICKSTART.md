# Quick Start Guide

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify installation
python -c "import xarray, dask; print('✓ Environment ready!')"
```

## Available Scripts

### 1. Generate Mock Data
Creates mock NetCDF files with specified ensemble members.

```bash
# Generate 100 files with 50 ensemble members (default)
python generate_mock_data.py data/example/652f73a7818c431a469c7ed3e9054e0a.nc data/

# Generate custom number of files and members
python generate_mock_data.py data/example/652f73a7818c431a469c7ed3e9054e0a.nc data/ --num-files 50 --num-members 25
```

**Output**: NetCDF files in specified output directory

---

### 2. Verify Data
Checks the generated files and displays statistics.

```bash
# Verify files in data directory
python verify_mock_data.py data/
```

**Output**: File count, dimensions, sample statistics

---

### 3. Run Benchmarks
Tests different loading strategies and generates performance report.

```bash
# Run benchmarks on all files in data directory
python benchmark_loading.py "data/*.nc"

# Save results to a specific directory
python benchmark_loading.py "data/*.nc" --output-dir results/
```

**Output**: 
- Console output with detailed timing
- `benchmark_results_TIMESTAMP.csv` - Tabular results
- `benchmark_results_TIMESTAMP.json` - Detailed results

**Duration**: ~40 seconds for 10 tests

---

### 4. Example: Optimal Loading
Demonstrates the fastest way to load the data.

```bash
python example_optimal_loading.py
```

**Output**: Interactive demonstration with timing and statistics

---

## Optimal Loading Code

Based on benchmark results, use this configuration:

```python
import xarray as xr

# Fastest method: 0.5 seconds for 100 files
ds = xr.open_mfdataset(
    'data/*.nc',
    engine='netcdf4',     # 1.9x faster than h5netcdf
    cache=False,          # 11% faster than cache=True
    chunks={},            # 15.8x faster than chunks=None
    combine='nested',
    concat_dim='file'
)

# Access data (lazy loading)
temperature = ds['t2m']

# Compute specific subset
sample = ds['t2m'].isel(file=0, number=0).compute()

# Clean up
ds.close()
```

## Key Results

| Method | Time | Speedup |
|--------|------|---------|
| **netCDF4 + Dask + no cache** ⭐ | **0.56s** | **15.8x** |
| netCDF4 + Dask + cache | 0.62s | 14.3x |
| h5netcdf + Dask + cache | 1.06s | 8.4x |
| netCDF4 + no Dask | 3.93s | 2.3x |
| netCDF4 + no Dask + cache | 8.88s | 1.0x (baseline) |

**Winner**: netCDF4 engine with Dask auto-chunking and no cache

## Project Structure

```
raster-benchmarking/
├── .venv/                          # Virtual environment (created by uv)
├── .gitignore                      # Git ignore rules
├── README.md                       # Main documentation
├── QUICKSTART.md                   # This file
├── BENCHMARK_RESULTS.md            # Detailed benchmark analysis
├── requirements.txt                # Package dependencies
├── requirements.lock               # Locked versions (for reproducibility)
├── generate_mock_data.py           # Create 100 mock files
├── verify_mock_data.py             # Verify generated files
├── benchmark_loading.py            # Performance benchmarking
├── example_optimal_loading.py      # Usage example
├── data/
│   ├── example/
│   │   └── 652f73a7818c...nc      # Original template (10 members)
│   ├── <uuid1>.nc                 # Generated file 1 (50 members)
│   ├── <uuid2>.nc                 # Generated file 2 (50 members)
│   └── ...                        # 98 more files
└── benchmark_results_*.csv         # Benchmark output files
```

## Troubleshooting

### h5netcdf Error on Databricks

If you see `H5DSget_num_scales` error with h5netcdf:

**Quick fix**: Already handled in `benchmark_loading.py` with `phony_dims='sort'`

See [DATABRICKS_TROUBLESHOOTING.md](DATABRICKS_TROUBLESHOOTING.md) for details.

## FAQ

### Q: Why is netCDF4 faster than h5netcdf?

A: For this specific workload (reading NetCDF files with multi-file aggregation), the netCDF4 library has better optimizations. h5netcdf might be faster for:
- Direct HDF5 file access
- Threading-based workflows
- Specific cloud storage scenarios

**Note**: h5netcdf may have compatibility issues on Databricks (see troubleshooting above).

### Q: Why disable caching?

A: Caching adds overhead for one-time data access patterns. If you're repeatedly accessing the same data, caching might help.

### Q: When should I not use Dask?

A: Avoid Dask (use `chunks=None`) only if:
- Dataset fits comfortably in memory
- You need random access to all data immediately
- You're doing very simple operations

For 100 files (5GB), Dask is strongly recommended.

### Q: Can I customize chunk sizes?

A: Yes, but auto-chunking (`chunks={}`) performed better than custom chunks in our tests. Customize only if you have specific access patterns:

```python
# Custom chunking
ds = xr.open_mfdataset(
    'data/*.nc',
    chunks={'number': 10, 'latitude': 100, 'longitude': 100},
    ...
)
```

### Q: How do I process data in parallel?

A: Use Dask's distributed scheduler:

```python
from dask.distributed import Client

# Create local cluster
client = Client(n_workers=4)

# Your xarray code here
ds = xr.open_mfdataset('data/*.nc', chunks={}, ...)
result = ds['t2m'].mean(dim=['latitude', 'longitude']).compute()

client.close()
```

## Reproducibility

All dependencies are locked in `requirements.lock`. To recreate the exact environment:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.lock
```

## Further Reading

- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) - Detailed performance analysis
- [README.md](README.md) - Full project documentation
- [xarray docs](https://docs.xarray.dev/) - xarray documentation
- [Dask docs](https://docs.dask.org/) - Dask documentation

