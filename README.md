# Raster Benchmarking - Mock NetCDF Data Generator

This repository contains tools for generating mock NetCDF files based on Copernicus Climate Data Store format.

## Overview

The mock dataset consists of **100 NetCDF files**, each containing temperature forecast data with **50 ensemble members** (increased from the original 10 in the template file).

## Dataset Specifications

- **Number of files**: 100
- **Ensemble members per file**: 50
- **Spatial dimensions**: 361 × 720 (latitude × longitude)
- **Resolution**: 0.5° × 0.5° global grid
- **Variable**: 2-meter temperature (t2m) in Kelvin
- **Data**: Random values between 220K and 320K (-53°C to 47°C)
- **File size**: ~50 MB per file

## Structure

```
data/
├── example/
│   └── 652f73a7818c431a469c7ed3e9054e0a.nc  (original template with 10 ensemble members)
├── <uuid1>.nc  (50 ensemble members)
├── <uuid2>.nc  (50 ensemble members)
├── ...
└── <uuid100>.nc  (50 ensemble members)
```

## Scripts

### `generate_mock_data.py`

Generates mock NetCDF files based on a template file.

**Usage:**
```bash
python3 generate_mock_data.py <template_file> <output_dir> [options]

# Example: Generate 100 files with 50 ensemble members (default)
python3 generate_mock_data.py data/example/652f73a7818c431a469c7ed3e9054e0a.nc data/

# Example: Generate 50 files with 25 ensemble members
python3 generate_mock_data.py data/example/652f73a7818c431a469c7ed3e9054e0a.nc data/ --num-files 50 --num-members 25
```

**Arguments:**
- `template_file` - Path to the template NetCDF file (required)
- `output_dir` - Directory where mock files will be saved (required)
- `--num-files` - Number of mock files to generate (default: 100)
- `--num-members` - Number of ensemble members per file (default: 50)
- `--engine` - Engine to use for writing ('netcdf4' or 'h5netcdf', default: 'netcdf4')
  - Use `--engine h5netcdf` for better Databricks compatibility

**Features:**
- Preserves all metadata and attributes from the original file
- Generates random temperature data (220-320K range)
- Creates files with unique UUID filenames
- Flexible ensemble size and file count

### `verify_mock_data.py`

Verifies the generated mock files and displays statistics.

**Usage:**
```bash
python3 verify_mock_data.py <data_dir>

# Example: Verify files in data directory
python3 verify_mock_data.py data/
```

**Arguments:**
- `data_dir` - Directory containing NetCDF files to verify (required)

**Output:**
- File count
- Dimensions verification
- Sample data statistics
- Spatial coverage details

### `benchmark_loading.py`

Benchmarks different xarray loading strategies across multiple configurations.

**Usage:**
```bash
source .venv/bin/activate

# Basic usage with default output directory
python benchmark_loading.py "data/*.nc"

# Specify output directory for results
python benchmark_loading.py "data/*.nc" --output-dir results/
```

**Arguments:**
- `file_pattern` - Glob pattern for NetCDF files to benchmark (required)
- `-o, --output-dir` - Directory where results will be saved (default: current directory)

**Tests:**
- Engine comparison: netcdf4 vs h5netcdf
- Caching: enabled vs disabled
- Dask: auto chunks vs specific chunks vs no chunks
- Generates CSV and JSON reports with detailed timing

## Environment Setup

This project uses a virtual environment with `uv` for fast, reproducible package management.

**All requirements match Databricks ML Runtime for consistency.**

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Compile and install locked requirements
uv pip compile requirements.txt -o requirements.lock
uv pip install -r requirements.lock
```

**Key versions** (matching Databricks ML Runtime):
- h5py 3.12.1 (matches DBR ML pre-installed version)
- h5netcdf 1.3.0 (compatible with h5py 3.12.1)
- xarray 2024.3.0
- netCDF4 1.7.3

This ensures **identical behavior** between local development and Databricks.

### Requirements

Core dependencies (see `requirements.txt`):
- xarray 2024.3.0
- netCDF4 1.7.3
- h5py 3.12.1 (matches Databricks ML Runtime)
- h5netcdf 1.3.0 (compatible with h5py 3.12.1)
- numpy 2.3.5
- dask[complete] 2025.11.0
- pandas 2.3.3

Exact versions are locked in `requirements.lock` for reproducibility.

**Note:** These versions match Databricks ML Runtime to ensure consistent behavior.

## NetCDF File Structure

Each generated file contains:

**Dimensions:**
- `number`: 50 (ensemble members)
- `valid_time`: 1 (single forecast time)
- `latitude`: 361 (90° to -90°)
- `longitude`: 720 (0° to 359.5°)

**Variables:**
- `t2m`: Temperature at 2 meters, shape (50, 1, 361, 720)
- `number`: Ensemble member IDs (0-49)
- `valid_time`: Forecast validation time
- `latitude`: Latitude coordinates
- `longitude`: Longitude coordinates
- `expver`: Experiment version

**Data Variable Attributes:**
- Units: Kelvin (K)
- Type: float32
- FillValue: NaN
- Plus various GRIB metadata attributes

## Technical Details

The mock data generation uses **xarray** for reading and writing NetCDF files, which provides:
- Seamless handling of NetCDF metadata and attributes
- Preservation of coordinate reference systems
- Compatibility with CF conventions
- Efficient I/O operations

Random data is generated using NumPy's uniform distribution to simulate realistic temperature ranges across the globe.

## Benchmarking Results

**🏆 Optimal Loading Configuration:**

```python
import xarray as xr

ds = xr.open_mfdataset(
    'data/*.nc',
    engine='netcdf4',    # netCDF4 is 1.9x faster than h5netcdf
    cache=False,         # Caching degrades performance
    chunks={},           # Auto Dask chunking: 15x faster than eager
    combine='nested',
    concat_dim='file'
)
```

**Performance**: Loads 100 files (5GB) in ~0.5 seconds

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed analysis.

### Key Findings
- ✅ **netCDF4 engine** outperforms h5netcdf by 1.89x
- ✅ **Dask with auto-chunking** provides 15.8x speedup
- ✅ **Disabling cache** improves performance by 11%
- ❌ Customer's hypothesis that h5netcdf would be faster was not confirmed

## Use Cases

This mock dataset is suitable for:
- Benchmarking raster processing pipelines
- Testing distributed computing frameworks
- Performance analysis of geospatial algorithms
- Development without requiring large real datasets from Copernicus

## Troubleshooting

### Diagnostic Script

Use `diagnose_h5netcdf_issue.py` to troubleshoot reading issues:

```bash
# Check library versions
python diagnose_h5netcdf_issue.py --versions-only

# Test reading a specific file
python diagnose_h5netcdf_issue.py data/my_file.nc
```

## Notes

- File names are generated using UUID4 to ensure uniqueness
- The original template file is preserved in `data/example/`
- All generated files maintain the same structure and metadata as the template
- Random seed is not set, so each run produces different data

