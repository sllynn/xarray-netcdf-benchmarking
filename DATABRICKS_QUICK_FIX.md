# Databricks Quick Fix Guide

## ⚠️ CRITICAL: Files created with netCDF4 cannot be read by h5netcdf

If you generated files locally with netCDF4 and uploaded them to Databricks, h5netcdf **will not work** regardless of `phony_dims` or `invalid_netcdf` settings.

## ✅ THE SOLUTION: Generate files on Databricks with h5netcdf engine

## 🚨 Error: h5netcdf fails even with phony_dims

If you get `RuntimeError: Unspecified error in H5DSget_num_scales` even WITH `phony_dims='sort'` and `invalid_netcdf=True`:

**Root Cause**: Files created with netCDF4 engine have dimension scales that h5netcdf cannot read at all. The error happens during file opening, before any workarounds can apply.

**Definitive Solution**: Generate files with h5netcdf engine:

```python
# On Databricks, generate with h5netcdf engine
!python generate_mock_data.py \
    /dbfs/template.nc \
    /tmp/ \
    --num-files 100 \
    --num-members 50 \
    --engine h5netcdf
```

This creates files that h5netcdf can read. Then benchmark normally.

---

## 🚨 Error: `[Errno -101] NetCDF: HDF error`

### Quick Fix (Copy & Paste)

```python
# === STEP 1: Reinstall libraries ===
%pip uninstall -y netCDF4 h5py h5netcdf
%pip install --no-cache-dir netCDF4==1.7.3 h5py==3.15.1 h5netcdf==1.7.3 xarray==2024.3.0
dbutils.library.restartPython()
```

After restart:

```python
# === STEP 2: Copy files to local disk ===
import shutil
from pathlib import Path

# Source: your DBFS location
source_files = '/dbfs/path/to/your/*.nc'

# Destination: local cluster storage
dest_dir = '/tmp/netcdf_work/'
Path(dest_dir).mkdir(parents=True, exist_ok=True)

# Copy files
import glob
for f in glob.glob(source_files):
    dest = Path(dest_dir) / Path(f).name
    shutil.copy(f, dest)
    print(f"Copied: {Path(f).name}")

print(f"\n✓ Files ready in {dest_dir}")
```

```python
# === STEP 3: Test reading ===
import xarray as xr

try:
    # Try netCDF4 (faster)
    ds = xr.open_mfdataset(
        f'{dest_dir}*.nc',
        engine='netcdf4',
        combine='nested',
        concat_dim='file'
    )
    print("✓ SUCCESS with netCDF4 engine")
    print(f"Loaded {len(ds.file)} files")
    print(f"Shape: {dict(ds.sizes)}")
    
except Exception as e:
    print(f"netCDF4 failed: {e}")
    print("\nTrying h5netcdf...")
    
    # Fallback to h5netcdf
    ds = xr.open_mfdataset(
        f'{dest_dir}*.nc',
        engine='h5netcdf',
        backend_kwargs={'phony_dims': 'sort'},
        combine='nested',
        concat_dim='file'
    )
    print("✓ SUCCESS with h5netcdf engine")
    print(f"Loaded {len(ds.file)} files")
    print(f"Shape: {dict(ds.sizes)}")
```

---

## 🔍 Diagnosis

Run this to check your setup:

```python
# Check library versions
import netCDF4, h5py, xarray as xr

print(f"netCDF4: {netCDF4.__version__}")
print(f"  NetCDF C lib: {netCDF4.__netcdf4libversion__}")
print(f"  HDF5 C lib: {netCDF4.__hdf5libversion__}")
print(f"\nh5py: {h5py.__version__}")
print(f"  HDF5 C lib: {h5py.version.hdf5_version}")
print(f"\nxarray: {xr.__version__}")

# Check if HDF5 versions match
if netCDF4.__hdf5libversion__ != h5py.version.hdf5_version:
    print("\n⚠️  HDF5 VERSION MISMATCH - This causes errors!")
else:
    print("\n✓ HDF5 versions match")
```

---

## 📋 Checklist

- [ ] Uninstalled and reinstalled netCDF4/h5py/h5netcdf
- [ ] Restarted Python kernel after install
- [ ] Copied files from `/dbfs/` to `/tmp/`
- [ ] Confirmed files exist and are readable
- [ ] Tested with both engines (netCDF4 and h5netcdf)

---

## ❓ Why This Happens

1. **DBFS FUSE**: `/dbfs/` paths use FUSE mount which causes HDF5 file locking issues
2. **System libraries**: Databricks system HDF5 conflicts with pip-installed versions
3. **Pre-installed packages**: Databricks comes with pre-installed scientific packages that may be incompatible

---

## 💡 Best Practices for Databricks

### ✅ DO:
- Generate files directly on cluster (`/tmp/` or `/local_disk0/`)
- Use pip to install specific library versions
- Copy files to local disk before processing
- Restart Python after installing libraries
- Use DBR ML runtime for better library support

### ❌ DON'T:
- Read NetCDF files directly from `/dbfs/` paths
- Mix conda and pip installations
- Rely on pre-installed library versions
- Process files on DBFS-mounted paths
- Skip kernel restart after package install

---

## 🔗 More Help

- Full guide: [DATABRICKS_TROUBLESHOOTING.md](DATABRICKS_TROUBLESHOOTING.md)
- Diagnostic tool: `python diagnose_h5netcdf_issue.py /tmp/file.nc`
- Benchmark script handles both engines automatically

---

## 🎯 One-Line Test

```python
# Quick test if netCDF4 works
!python -c "import xarray as xr; ds = xr.open_dataset('/tmp/test.nc', engine='netcdf4'); print('✓ Works!')"
```

