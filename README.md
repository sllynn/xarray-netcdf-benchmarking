# Raster Benchmarking

Tools for benchmarking cloud-optimized geospatial data formats and access patterns on Databricks, with a focus on NetCDF, Zarr, and Kerchunk.

## Overview

This repository provides a comprehensive toolkit for:

1. **Generating mock data** - Create realistic NetCDF test datasets based on Copernicus Climate Data Store format
2. **Format conversion** - Convert NetCDF files to cloud-optimized Zarr format (individual or stacked)
3. **Kerchunk indexing** - Create virtual Zarr references from existing NetCDF files without data duplication
4. **Performance benchmarking** - Compare loading performance across formats and storage locations

## Key Findings

**Remote Storage Performance (Unity Catalog Volumes, 100 files, ~1.3B data points):**

| Format | Open Time | Access Time | Total Time |
|--------|-----------|-------------|------------|
| Zarr Stacked | 2.66s | 1.41s | **4.08s** |
| Kerchunk Combined | 1.49s | 3.31s | 4.80s |
| Zarr Individual (100 dirs) | 18.0s | 1.16s | 19.2s |

**Recommendations:**
- ✅ **Kerchunk Combined** delivers ~95% of Zarr's performance with zero data duplication
- ✅ **Zarr Stacked** is optimal when read latency is critical (~4s)
- ❌ Avoid accessing many individual files over network storage (4-5× slower)

## Dataset Specifications

The benchmark dataset consists of **100 files**, each containing temperature forecast data:

- **Ensemble members per file**: 50
- **Spatial dimensions**: 361 × 720 (latitude × longitude)
- **Resolution**: 0.5° × 0.5° global grid
- **Variable**: 2-meter temperature (t2m) in Kelvin
- **File size**: ~50 MB per file (NetCDF)

## Scripts

### Data Generation

#### `generate_mock_data.py`

Generates mock NetCDF files based on a template file.

```bash
# Generate 100 files with 50 ensemble members (default)
python generate_mock_data.py data/example/template.nc data/

# Generate custom number of files and members
python generate_mock_data.py data/example/template.nc data/ --num-files 50 --num-members 25
```

#### `verify_mock_data.py`

Verifies generated files and displays statistics.

```bash
python verify_mock_data.py data/
```

### Format Conversion

#### `convert_to_zarr.py`

Converts NetCDF files to Zarr format.

```bash
# Convert to individual Zarr directories (one per NetCDF)
python convert_to_zarr.py individual "data/*.nc" zarr_individual/

# Convert to a single stacked Zarr (all files combined)
python convert_to_zarr.py stacked "data/*.nc" zarr_stacked.zarr
```

#### `create_kerchunk_refs.py`

Creates Kerchunk reference files from NetCDF without data duplication.

```bash
# Create individual JSON references + combined master.json
python create_kerchunk_refs.py "data/*.nc" references/

# Only create individual references (skip master.json)
python create_kerchunk_refs.py "data/*.nc" references/ --skip-combine
```

### Benchmarking

#### `benchmark_loading.py`

Benchmarks NetCDF loading with different xarray configurations.

```bash
python benchmark_loading.py "data/*.nc" --output-dir results/
```

Tests: netcdf4 vs h5netcdf engines, caching, Dask chunking strategies.

#### `benchmark_loading_zarr.py`

Benchmarks Zarr loading performance.

```bash
# Benchmark stacked Zarr
python benchmark_loading_zarr.py --single zarr_stacked.zarr -o results/

# Benchmark individual Zarr directories
python benchmark_loading_zarr.py "zarr_individual/*.zarr" -o results/
```

#### `benchmark_loading_kerchunk.py`

Benchmarks Kerchunk reference loading performance.

```bash
# Benchmark combined master.json
python benchmark_loading_kerchunk.py --combined references/master.json -o results/

# Benchmark individual JSON references
python benchmark_loading_kerchunk.py "references/*.json" -o results/
```

#### Zarr Read Benchmarks (Databricks Volumes)

Use the Databricks notebook `[notebooks/07_read_benchmarks.py](notebooks/07_read_benchmarks.py)` to:

- Measure metadata open time (`xarray.open_zarr`)
- Measure single-chunk vs multi-chunk slice reads
- Measure scaling across multiple Zarr stores (forecast cycles)

The notebook optionally generates a week of forecast-cycle Zarr stores locally
under `/local_disk0/zarr_fixtures` and syncs them to:
`/Volumes/<catalog>/<schema>/<volume_name>/read_benchmarks/zarr_fixtures/`

Results are written to:
`/Volumes/<catalog>/<schema>/<volume_name>/read_benchmarks/results/`

## Environment Setup

This project uses `uv` for fast, reproducible package management.

**All requirements match Databricks ML Runtime 17.3LTS for consistency.**

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Compile and install locked requirements
uv pip compile requirements.txt -o requirements.lock
uv pip install -r requirements.lock
```

### Requirements

All versions match **Databricks ML Runtime 17.3LTS** (see `requirements.txt`):

- xarray 2025.11.0
- zarr 3.1.5
- netCDF4 1.7.3
- numpy 2.1.3
- pandas 2.2.3
- dask[complete] 2025.11.0
- h5py 3.12.1 (matches DBR 17.3LTS pre-installed version)
- h5netcdf 1.3.0 (compatible with h5py 3.12.1)
- kerchunk 0.2.9
- fsspec 2025.10.0

Exact versions are locked in `requirements.lock` for reproducibility.

## Workflow Example

Complete workflow from raw NetCDF to benchmarking:

```bash
# 1. Generate mock data
python generate_mock_data.py template.nc data/ --num-files 100

# 2. Create Kerchunk references (no data duplication)
python create_kerchunk_refs.py "data/*.nc" references/

# 3. Convert to Zarr (creates copy of data)
python convert_to_zarr.py stacked "data/*.nc" zarr_stacked.zarr

# 4. Run benchmarks
python benchmark_loading_kerchunk.py --combined references/master.json -o results/kerchunk/
python benchmark_loading_zarr.py --single zarr_stacked.zarr -o results/zarr/
```

## Project Structure

```
raster-benchmarking/
├── data/                           # NetCDF files
│   └── example/                    # Template file
├── references/                     # Kerchunk JSON references
│   ├── *.json                      # Per-file references
│   └── master.json                 # Combined virtual dataset
├── results/                        # Benchmark results
│   ├── local/                      # Local storage benchmarks
│   └── remote/                     # Remote storage benchmarks
├── generate_mock_data.py           # Create mock NetCDF files
├── verify_mock_data.py             # Verify generated files
├── convert_to_zarr.py              # NetCDF → Zarr conversion
├── create_kerchunk_refs.py         # Create Kerchunk references
├── benchmark_loading.py            # NetCDF benchmarks
├── benchmark_loading_zarr.py       # Zarr benchmarks
├── benchmark_loading_kerchunk.py   # Kerchunk benchmarks
├── requirements.txt                # Package dependencies
└── requirements.lock               # Locked versions
```

## Further Reading

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) - Detailed benchmark analysis
