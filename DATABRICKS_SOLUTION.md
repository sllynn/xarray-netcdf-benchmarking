# ✅ DEFINITIVE DATABRICKS SOLUTION

## The Problem

You're getting `RuntimeError: Unspecified error in H5DSget_num_scales` even with:
- ✅ `phony_dims='sort'` in backend_kwargs
- ✅ `invalid_netcdf=True` in backend_kwargs  
- ✅ Files on `/tmp/` (not DBFS)

**Why the workarounds don't work:**

The error happens during h5netcdf's file opening initialization, **before** any workaround parameters take effect. The files have dimension scale metadata written by netCDF4-C that h5py (used by h5netcdf) fundamentally cannot parse.

## The Root Cause

```
netCDF4 (write) → HDF5 dimension scales → h5netcdf (read) ❌ INCOMPATIBLE
```

Your files were created with `engine='netcdf4'` which uses the NetCDF4-C library. This library writes HDF5 dimension scale metadata in a format that h5py's `H5DSget_num_scales()` cannot read, causing the error at line 738 in h5netcdf's `_unlabeled_dimension_mix` function.

## The Solution

### ✅ Generate files WITH h5netcdf engine

```python
# On Databricks
python generate_mock_data.py \
    /dbfs/template.nc \
    /tmp/benchmark_data/ \
    --num-files 100 \
    --num-members 50 \
    --engine h5netcdf    # ← THE KEY
```

**Why this works:**
- h5netcdf writes dimension scales that h5netcdf can read
- No incompatibility between writer and reader
- All workarounds are unnecessary

## Complete Databricks Workflow

```python
# ====================================================================
# STEP 1: Generate files with h5netcdf engine
# ====================================================================
%sh
cd /Workspace/path/to/project
python generate_mock_data.py \
    /dbfs/mnt/template.nc \
    /tmp/benchmark_data/ \
    --num-files 100 \
    --num-members 50 \
    --engine h5netcdf

# ====================================================================
# STEP 2: Run benchmarks
# ====================================================================
%sh
cd /Workspace/path/to/project
python benchmark_loading.py \
    "/tmp/benchmark_data/*.nc" \
    --output-dir /tmp/results/

# ====================================================================
# STEP 3: View results
# ====================================================================
import pandas as pd
results = pd.read_csv('/tmp/results/benchmark_results_latest.csv')
display(results.sort_values('total_time'))
```

## Comparison Table

| Scenario | netCDF4 Engine | h5netcdf Engine | Works? |
|----------|---------------|-----------------|--------|
| Files written with netCDF4 | ✅ | ❌ | netCDF4 only |
| Files written with h5netcdf | ⚠️ Sometimes | ✅ | Both work |
| Files on DBFS | ❌ | ❌ | Neither (use /tmp/) |
| Files on /tmp/ | ✅ (if written with netCDF4) | ✅ (if written with h5netcdf) | Depends on writer |

## What About Performance?

Your benchmarks showed **netCDF4 is 1.9x faster**, but that was:
- On local machine (not Databricks)
- With files written by netCDF4
- Without DBFS/FUSE complications

On Databricks, if h5netcdf works and netCDF4 doesn't, speed doesn't matter!

**Recommendation**: 
1. Generate files with h5netcdf on Databricks
2. Benchmark BOTH engines on Databricks
3. Use whichever is faster (probably still netCDF4)

## Updated benchmark_loading.py

The script now includes:
```python
backend_kwargs = {
    "phony_dims": "sort",
    "invalid_netcdf": True
}
```

But these won't help if files were created with netCDF4. You MUST use `--engine h5netcdf` when generating.

## Key Takeaways

1. ✅ **DO**: Generate files on Databricks with `--engine h5netcdf`
2. ✅ **DO**: Store files in `/tmp/` not `/dbfs/`
3. ✅ **DO**: Reinstall libraries with pip if you get HDF errors
4. ❌ **DON'T**: Upload files created with netCDF4 and expect h5netcdf to read them
5. ❌ **DON'T**: Rely on `phony_dims` to fix incompatible files
6. ❌ **DON'T**: Use DBFS paths for active file processing

## One-Line Test

After generating with h5netcdf:

```python
import xarray as xr
ds = xr.open_dataset('/tmp/benchmark_data/test.nc', engine='h5netcdf')
print(f"✅ Success! Shape: {dict(ds.sizes)}")
```

## Files Updated

- ✅ `generate_mock_data.py` - Added `--engine` flag
- ✅ `benchmark_loading.py` - Added `invalid_netcdf=True` to backend_kwargs
- ✅ `DATABRICKS_QUICK_FIX.md` - Updated with h5netcdf solution
- ✅ `DATABRICKS_TROUBLESHOOTING.md` - Expanded error solutions
- ✅ `README.md` - Added engine documentation

---

**Summary**: Files created with netCDF4 cannot be read by h5netcdf on Databricks, regardless of workarounds. Generate files with `--engine h5netcdf` instead.
