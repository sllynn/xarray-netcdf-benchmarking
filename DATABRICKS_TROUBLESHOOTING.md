# Databricks Troubleshooting Guide

## Common NetCDF/HDF5 Errors on Databricks

### Error 1: netCDF4 "HDF error" ⚠️

#### Problem

```
[Errno -101] NetCDF: HDF error: '/path/to/file.nc'
```

This error occurs with the **netCDF4 engine** and indicates an underlying HDF5 library incompatibility.

#### Root Cause

1. **Library version mismatch**: System HDF5 library vs pip-installed netCDF4
2. **DBFS FUSE issues**: Files on `/dbfs/` paths accessed through FUSE mount
3. **Compiled library conflicts**: netCDF4 compiled against different HDF5 than available
4. **File corruption**: Files transferred incorrectly or corrupted

#### Solutions

##### Solution 1: Reinstall netCDF4 with pip (FASTEST FIX) ✅

```python
# In Databricks notebook cell
%pip uninstall -y netCDF4
%pip install --no-cache-dir netCDF4==1.7.3
dbutils.library.restartPython()
```

This forces pip to build/install against compatible libraries.

##### Solution 2: Use local disk instead of DBFS

```python
# Copy files from DBFS to local /tmp
import shutil
import glob

dbfs_files = glob.glob('/dbfs/path/to/*.nc')
for f in dbfs_files:
    local_path = f'/tmp/{Path(f).name}'
    shutil.copy(f, local_path)
    
# Now read from /tmp
ds = xr.open_mfdataset('/tmp/*.nc', engine='netcdf4')
```

**Why?**: DBFS uses FUSE (Filesystem in Userspace) which can cause issues with HDF5's low-level file operations.

##### Solution 3: Use h5netcdf engine instead

```python
ds = xr.open_dataset(
    file_path,
    engine='h5netcdf',
    backend_kwargs={'phony_dims': 'sort'}
)
```

h5netcdf may have better compatibility on Databricks.

##### Solution 4: Generate files on Databricks

Instead of uploading files, generate them directly on the cluster:

```python
# Generate directly on Databricks
!python generate_mock_data.py \
    /dbfs/mnt/template.nc \
    /tmp/ \
    --num-files 100
```

---

### Error 2: h5netcdf "H5DSget_num_scales" Error

#### Problem

```
Unspecified error in H5DSget_num_scales (return value <0)
```

This error occurs specifically with the **h5netcdf engine** when reading files created by netCDF4.

### Root Cause

This is a **dimension scale metadata incompatibility** between:
- **NetCDF4-C library** (used when writing files with `engine='netcdf4'`)
- **HDF5 library** (used by h5py/h5netcdf when reading files)

The issue is more common on Databricks due to:
1. Different HDF5 library versions between local and cluster environments
2. How system libraries are compiled and linked
3. Potential conflicts in the Python environment

### Solutions

#### Solution 1: Generate files with h5netcdf engine (MOST RELIABLE) ✅

**If `phony_dims` doesn't work**, the files are fundamentally incompatible. Generate them with h5netcdf:

```bash
# On Databricks, generate files with h5netcdf engine
python generate_mock_data.py \
    /dbfs/template.nc \
    /tmp/ \
    --num-files 100 \
    --engine h5netcdf
```

**Why this works**: Files written with h5netcdf can be read by h5netcdf without dimension scale issues.

#### Solution 2: Use `phony_dims` parameter

The `benchmark_loading.py` script automatically uses this:

```python
import xarray as xr

ds = xr.open_dataset(
    file_path,
    engine='h5netcdf',
    backend_kwargs={
        'phony_dims': 'sort',
        'invalid_netcdf': True  # Also helps on Databricks
    }
)
```

**What it does**: 
- `phony_dims='sort'` tells h5netcdf to handle dimension scales more permissively
- `invalid_netcdf=True` allows reading non-compliant files
- Works around the strict dimension scale checking that causes the error

**Note**: If the error still occurs during file opening (in H5DSget_num_scales), use Solution 1 instead.

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

### Quick Diagnosis: Which Error Do I Have?

Run the diagnostic script to identify your specific issue:

```bash
python diagnose_h5netcdf_issue.py /tmp/your_file.nc
```

**Look for:**
- ❌ netCDF4 engine fails → You have **Error 1** (HDF library issue)
- ❌ h5netcdf engine fails → You have **Error 2** (dimension scales issue)
- ⚠️ HDF5 version mismatch warning → Reinstall libraries

### Combined Solution for Both Errors

If both engines fail, try this comprehensive fix:

```python
# Notebook Cell 1: Clean reinstall
%pip uninstall -y netCDF4 h5py h5netcdf
%pip install --no-cache-dir netCDF4==1.7.3 h5py==3.15.1 h5netcdf==1.7.3
dbutils.library.restartPython()

# Notebook Cell 2: Copy files to local disk
import shutil
from pathlib import Path

source_dir = '/dbfs/your/files/'
dest_dir = '/tmp/netcdf_files/'
Path(dest_dir).mkdir(exist_ok=True)

for f in Path(source_dir).glob('*.nc'):
    shutil.copy(f, dest_dir)

# Notebook Cell 3: Test reading
import xarray as xr

# Try netCDF4 first (usually faster)
try:
    ds = xr.open_mfdataset(
        '/tmp/netcdf_files/*.nc',
        engine='netcdf4',
        combine='nested',
        concat_dim='file'
    )
    print("✓ netCDF4 engine works!")
except Exception as e:
    print(f"✗ netCDF4 failed: {e}")
    
    # Fall back to h5netcdf
    ds = xr.open_mfdataset(
        '/tmp/netcdf_files/*.nc',
        engine='h5netcdf',
        backend_kwargs={'phony_dims': 'sort'},
        combine='nested',
        concat_dim='file'
    )
    print("✓ h5netcdf engine works!")
```

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

