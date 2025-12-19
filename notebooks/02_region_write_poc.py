# Databricks notebook source
# MAGIC %md
# MAGIC # GRIB to Zarr Region Write - Proof of Concept
# MAGIC
# MAGIC This notebook demonstrates the core GRIB processing logic:
# MAGIC - Reading GRIB files with eccodes
# MAGIC - Mapping forecast hours to step indices
# MAGIC - Writing data to specific Zarr regions (slots)
# MAGIC - Parallel processing with ThreadPoolExecutor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install -r ../requirements.lock

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os

# Configuration
CATALOG = "stuart"
SCHEMA = "lseg"
LANDING_VOLUME = "netcdf"

# Paths
LOCAL_ZARR_PATH = "/tmp/forecast.zarr"
GRIB_LANDING_ZONE = f"/Volumes/{CATALOG}/{SCHEMA}/{LANDING_VOLUME}/landing/"
os.environ["GRIB_LANDING_ZONE"] = GRIB_LANDING_ZONE

# Processing parameters
MAX_WORKERS = 32  # Match cluster core count

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import sys
# sys.path.insert(0, '/Workspace/Repos/your_user/raster-benchmarking')

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.zarr_init import generate_forecast_steps, build_hour_to_index_map
from src.region_writer import (
    write_grib_to_zarr_region,
    write_grib_batch_parallel,
    extract_grib_metadata,
    read_grib_data,
    WriteResult,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Hour-to-Index Mapping
# MAGIC
# MAGIC The forecast uses non-uniform time steps, so we need a lookup table
# MAGIC to map forecast hours to array indices.

# COMMAND ----------

# Build the mapping
forecast_steps = generate_forecast_steps()
hour_to_index = build_hour_to_index_map(forecast_steps)

print(f"Total forecast steps: {len(forecast_steps)}")
print(f"\nSample mappings:")
sample_hours = [0, 6, 12, 24, 48, 90, 93, 96, 144, 150, 180, 360]
for hour in sample_hours:
    if hour in hour_to_index:
        print(f"  Hour {hour:3d} -> Index {hour_to_index[hour]:3d}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## List Available GRIB Files

# COMMAND ----------

# MAGIC %sh mkdir -p $GRIB_LANDING_ZONE

# COMMAND ----------

# List GRIB files in landing zone
grib_files = dbutils.fs.ls(GRIB_LANDING_ZONE)
grib_paths = [f.path for f in grib_files if f.name.endswith(('.grib', '.grib2'))]

print(f"Found {len(grib_paths)} GRIB files")
if grib_paths:
    print(f"\nFirst 5 files:")
    for path in grib_paths[:5]:
        print(f"  {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract GRIB Metadata (Single File)

# COMMAND ----------

if grib_paths:
    # Convert DBFS path to local path for eccodes
    test_file = grib_paths[0].replace("dbfs:", "")
    
    print(f"Extracting metadata from: {test_file}")
    
    try:
        metadata = extract_grib_metadata(test_file)
        print(f"\nMetadata:")
        print(f"  Variable: {metadata.variable}")
        print(f"  Forecast hour: {metadata.forecast_hour}")
        print(f"  Reference time: {metadata.reference_time}")
        print(f"  Valid time: {metadata.valid_time}")
        print(f"  Ensemble members: {metadata.n_ensemble}")
        print(f"  Shape: {metadata.shape}")
        
        if metadata.forecast_hour in hour_to_index:
            print(f"\n  -> Maps to step index: {hour_to_index[metadata.forecast_hour]}")
        else:
            print(f"\n  ! Warning: Hour {metadata.forecast_hour} not in mapping")
            
    except Exception as e:
        print(f"Error extracting metadata: {e}")
else:
    print("No GRIB files available for testing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single Region Write Test

# COMMAND ----------

if grib_paths:
    test_file = grib_paths[0].replace("dbfs:", "")
    
    print(f"Writing single file: {test_file}")
    print(f"Target store: {LOCAL_ZARR_PATH}")
    print(f"Local staging: enabled")
    
    start_time = time.time()
    
    result = write_grib_to_zarr_region(
        grib_path=test_file,
        zarr_store_path=LOCAL_ZARR_PATH,
        hour_to_index=hour_to_index,
        stage_locally=True,  # Copy to local SSD before reading
    )
    
    elapsed = time.time() - start_time
    
    if result.success:
        print(f"\n✓ Write successful!")
        print(f"  Variable: {result.variable}")
        print(f"  Step index: {result.step_index}")
        print(f"  Ensemble slice: {result.ensemble_slice}")
        print(f"  Time: {result.elapsed_ms:.1f}ms")
    else:
        print(f"\n✗ Write failed: {result.error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parallel Batch Write Test

# COMMAND ----------

# Test parallel writes with multiple files
# Key optimization: stage files to local SSD first for faster reads
batch_size = min(MAX_WORKERS, len(grib_paths))

if batch_size > 0:
    test_batch = [p.replace("dbfs:", "") for p in grib_paths[:batch_size]]
    
    print(f"Testing parallel write with {batch_size} files")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Local staging: enabled (files copied to /local_disk0/grib_staging)")
    
    start_time = time.time()
    
    results = write_grib_batch_parallel(
        grib_paths=test_batch,
        zarr_store_path=LOCAL_ZARR_PATH,
        hour_to_index=hour_to_index,
        max_workers=MAX_WORKERS,
        stage_locally=True,  # Copy files to local SSD before reading
        batch_stage=True,    # Stage all files first, then process
    )
    
    elapsed = time.time() - start_time
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\n✓ Batch complete in {elapsed:.2f}s")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Throughput: {successful / elapsed:.1f} files/s")
    
    if failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if not r.success:
                print(f"  {r.grib_path}: {r.error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare: With vs Without Local Staging
# MAGIC
# MAGIC This cell demonstrates the performance difference between reading
# MAGIC directly from cloud storage vs staging to local SSD first.

# COMMAND ----------

# Compare with and without local staging (run a smaller batch for comparison)
if len(grib_paths) >= 8:
    comparison_batch = [p.replace("dbfs:", "") for p in grib_paths[:8]]
    
    print("=" * 60)
    print("PERFORMANCE COMPARISON: Local Staging")
    print("=" * 60)
    
    # Test WITHOUT local staging (direct from cloud)
    print("\n1. WITHOUT local staging (reading directly from Volumes):")
    start_time = time.time()
    results_no_stage = write_grib_batch_parallel(
        grib_paths=comparison_batch,
        zarr_store_path=LOCAL_ZARR_PATH,
        hour_to_index=hour_to_index,
        max_workers=8,
        stage_locally=False,
    )
    elapsed_no_stage = time.time() - start_time
    print(f"   Time: {elapsed_no_stage:.2f}s")
    print(f"   Throughput: {len(comparison_batch) / elapsed_no_stage:.1f} files/s")
    
    # Test WITH local staging
    print("\n2. WITH local staging (copy to /local_disk0 first):")
    start_time = time.time()
    results_with_stage = write_grib_batch_parallel(
        grib_paths=comparison_batch,
        zarr_store_path=LOCAL_ZARR_PATH,
        hour_to_index=hour_to_index,
        max_workers=8,
        stage_locally=True,
        batch_stage=True,
    )
    elapsed_with_stage = time.time() - start_time
    print(f"   Time: {elapsed_with_stage:.2f}s")
    print(f"   Throughput: {len(comparison_batch) / elapsed_with_stage:.1f} files/s")
    
    # Summary
    speedup = elapsed_no_stage / elapsed_with_stage if elapsed_with_stage > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"SPEEDUP with local staging: {speedup:.1f}x")
    print(f"{'=' * 60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Written Data

# COMMAND ----------

import xarray as xr
import numpy as np

# Open store and verify data
ds = xr.open_zarr(LOCAL_ZARR_PATH, consolidated=True)

print("Store dimensions:", dict(ds.sizes))
print("\nVariables:", list(ds.data_vars))

# Check NaN percentage for written steps
if results:
    written_indices = [r.step_index for r in results if r.success and r.step_index >= 0]
    variable = results[0].variable if results else 't2m'
    
    if written_indices and variable in ds:
        for idx in written_indices[:3]:  # Check first 3
            sample = ds[variable].isel(step=idx, number=0, latitude=180, longitude=360).values
            nan_pct = float(sample == sample) * 100  # True if not NaN
            print(f"\nStep {idx}: {'Valid data' if not np.isnan(sample) else 'NaN'}")

ds.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Success Criteria Validation
# MAGIC
# MAGIC Per the architecture document, single file write should complete in < 500ms.

# COMMAND ----------

target_time_ms = 500.0

if results:
    write_times = [r.elapsed_ms for r in results if r.success]
    
    if write_times:
        avg_time = sum(write_times) / len(write_times)
        max_time = max(write_times)
        min_time = min(write_times)
        
        print(f"Write time statistics:")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min: {min_time:.1f}ms")
        print(f"  Max: {max_time:.1f}ms")
        
        if avg_time < target_time_ms:
            print(f"\n✓ PASS: Average write time {avg_time:.1f}ms < {target_time_ms}ms target")
        else:
            print(f"\n✗ FAIL: Average write time {avg_time:.1f}ms >= {target_time_ms}ms target")

