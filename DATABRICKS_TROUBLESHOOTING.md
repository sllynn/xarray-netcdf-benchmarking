# Databricks Troubleshooting Guide

## h5netcdf "H5DSget_num_scales" Error

### Problem

When running benchmarks on Databricks with files stored in `/tmp` (local cluster storage), you may encounter:

```
Unspecified error in H5DSget_num_scales (return value <0)
```

This error occurs specifically with the **h5netcdf engine**, while netcdf4 works fine.

### Root Cause

This is a **dimension scale metadata incompatibility** between:
- **NetCDF4-C library** (used when writing files with `engine='netcdf4'`)
- **HDF5 library** (used by h5py/h5netcdf when reading files)

The issue is more common on Databricks due to:
1. Different HDF5 library versions between local and cluster environments
2. How system libraries are compiled and linked
3. Potential conflicts in the Python environment

### Solutions

#### Solution 1: Use `phony_dims` parameter (RECOMMENDED) ✅

The `benchmark_loading.py` script has been updated to automatically use this fix:

```python
import xarray as xr

ds = xr.open_dataset(
    file_path,
    engine='h5netcdf',
    backend_kwargs={'phony_dims': 'sort'}  # This fixes the issue
)
```

**What it does**: 
- `phony_dims='sort'` tells h5netcdf to handle dimension scales more permissively
- It works around the strict dimension scale checking that causes the error

#### Solution 2: Write files with h5netcdf engine

If you're generating files on Databricks, modify `generate_mock_data.py`:

```python
# Instead of:
new_ds.to_netcdf(output_path, engine='netcdf4')

# Use:
new_ds.to_netcdf(output_path, engine='h5netcdf')
```

This ensures dimension scales are written in a format h5netcdf can read.

#### Solution 3: Stick with netcdf4 engine (EASIEST)

Since the benchmarks showed **netcdf4 is 1.9x faster** anyway, you can:

1. Skip h5netcdf tests entirely
2. Only use netcdf4 engine for production workloads

Modify test configs in `benchmark_loading.py` to remove h5netcdf tests.

#### Solution 4: Reinstall libraries on Databricks

Install compatible versions explicitly:

```bash
%pip install --force-reinstall h5py==3.15.1 h5netcdf==1.7.3 netCDF4==1.7.3
```

Or use the locked requirements:

```bash
%pip install -r requirements.lock
```

### Diagnostic Steps

#### 1. Check Library Versions

Run the diagnostic script:

```bash
python diagnose_h5netcdf_issue.py --versions-only
```

Look for:
- HDF5 C library versions (should match between h5py and netCDF4)
- Python package versions

#### 2. Test with a Single File

```bash
python diagnose_h5netcdf_issue.py /tmp/my_file.nc
```

This will test multiple reading methods and show which ones work.

#### 3. Check File Location

The error is more common with files on local disk (`/tmp`, `/dbfs/tmp`). Try:
- Moving files to DBFS: `/dbfs/mnt/...`
- Using cloud storage: `s3://...` or `abfss://...`

### Databricks-Specific Considerations

#### Environment Setup

On Databricks, use a notebook cell to install packages:

```python
%pip install xarray==2024.3.0 netCDF4==1.7.3 h5netcdf==1.7.3 h5py==3.15.1 dask[complete]==2025.11.0
```

#### DBR (Databricks Runtime) Version

Different DBR versions have different system libraries:
- **DBR 13.x+**: Generally has newer HDF5 libraries
- **DBR ML**: May have different versions pre-installed

Try using the latest DBR ML runtime for better library compatibility.

#### Cluster Configuration

For best results:
- Use a single-node cluster for testing
- Install libraries at cluster scope, not notebook scope
- Restart Python kernel after installing libraries

### Code Examples for Databricks

#### Generate Files on Databricks

```python
# Notebook cell 1: Install libraries
%pip install xarray netCDF4 h5netcdf numpy

# Notebook cell 2: Generate mock data
!python generate_mock_data.py \
    /dbfs/mnt/data/template.nc \
    /dbfs/tmp/mock_data/ \
    --num-files 10 \
    --num-members 50
```

#### Run Benchmarks on Databricks

```python
# Notebook cell 1: Setup
%pip install xarray netCDF4 h5netcdf h5py dask pandas

# Notebook cell 2: Run benchmarks
!python benchmark_loading.py \
    "/dbfs/tmp/mock_data/*.nc" \
    --output-dir /dbfs/tmp/results/
```

#### Workaround in Notebook Code

If you want to handle the error in your own code:

```python
import xarray as xr

def safe_open_dataset(path, engine='h5netcdf', **kwargs):
    """Open dataset with h5netcdf, falling back to netcdf4 on error."""
    try:
        if engine == 'h5netcdf':
            # Try with phony_dims first
            return xr.open_dataset(
                path, 
                engine='h5netcdf',
                backend_kwargs={'phony_dims': 'sort'},
                **kwargs
            )
        else:
            return xr.open_dataset(path, engine=engine, **kwargs)
    except Exception as e:
        print(f"Warning: h5netcdf failed ({e}), falling back to netcdf4")
        return xr.open_dataset(path, engine='netcdf4', **kwargs)

# Use it
ds = safe_open_dataset('/tmp/my_file.nc')
```

### Understanding the Error

**What are HDF5 Dimension Scales?**

NetCDF4 files are actually HDF5 files with conventions. Dimension scales are HDF5's way of storing coordinate information (latitude, longitude, time, etc.).

**Why the incompatibility?**

1. **NetCDF4-C library** writes dimension scales following NetCDF conventions
2. **h5py** (used by h5netcdf) reads HDF5 directly and expects strict HDF5 dimension scale format
3. Minor differences in format cause `H5DSget_num_scales` to fail

**Why does netcdf4 engine work?**

The netcdf4 Python library uses NetCDF4-C library for both reading and writing, so there's no incompatibility.

### Performance Comparison

Based on benchmarks, **you probably don't need h5netcdf**:

| Engine | Time | Use Case |
|--------|------|----------|
| **netcdf4** | 0.56s | ✅ **Best for NetCDF files** |
| h5netcdf | 1.06s | Use for pure HDF5 or threading |

### Additional Resources

- [h5netcdf documentation](https://h5netcdf.org/)
- [xarray backend engine docs](https://docs.xarray.dev/en/stable/user-guide/io.html)
- [HDF5 dimension scales](https://docs.hdfgroup.org/hdf5/develop/_dims.html)

### Quick Reference

```python
# ✅ WORKS: netcdf4 engine (RECOMMENDED)
ds = xr.open_dataset(path, engine='netcdf4')

# ✅ WORKS: h5netcdf with phony_dims
ds = xr.open_dataset(path, engine='h5netcdf', 
                     backend_kwargs={'phony_dims': 'sort'})

# ❌ FAILS: h5netcdf without phony_dims (on Databricks)
ds = xr.open_dataset(path, engine='h5netcdf')  # May fail

# ✅ WORKS: Write with h5netcdf, read with h5netcdf
ds.to_netcdf(path, engine='h5netcdf')
ds2 = xr.open_dataset(path, engine='h5netcdf')  # OK
```

---

**Bottom Line**: Use the updated `benchmark_loading.py` script which now includes the `phony_dims='sort'` fix, or stick with netcdf4 engine which is faster and more reliable.

