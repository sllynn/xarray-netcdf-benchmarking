# Databricks notebook source
# MAGIC %md
# MAGIC # Zarr Store Initialization
# MAGIC
# MAGIC This notebook demonstrates how to initialize a pre-allocated Zarr store
# MAGIC for low-latency weather data ingestion.
# MAGIC
# MAGIC **Key features:**
# MAGIC - Creates full directory structure for forecast cycle (145 steps, 360 hours)
# MAGIC - Initializes all chunks with NaN (nodata) values
# MAGIC - Applies optimal chunking strategy: 1 variable, 1 step, 50 ensemble, full spatial
# MAGIC - Consolidates metadata for fast access

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install -r ../requirements.lock

# COMMAND ----------

# Configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"
VOLUME_NAME = "silver"

# Paths
LOCAL_ZARR_PATH = "/local_disk0/forecast.zarr"
CLOUD_ZARR_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/forecast.zarr"

# Forecast parameters
VARIABLES = ["t2m", "u10", "v10", "sp"]  # 2m temp, 10m wind u/v, surface pressure
ENSEMBLE_MEMBERS = 50
LAT_SIZE = 361   # 0.5° global grid
LON_SIZE = 720

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import sys
# sys.path.insert(0, '/Workspace/Repos/your_user/raster-benchmarking')

from datetime import datetime
from src.zarr_init import (
    initialize_zarr_store,
    generate_forecast_steps,
    build_hour_to_index_map,
    get_zarr_store_info,
    ForecastStepConfig,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore Forecast Step Structure
# MAGIC
# MAGIC The forecast uses non-uniform time steps covering 360 hours:
# MAGIC - Hours 0-90: Hourly (91 steps)
# MAGIC - Hours 93-144: 3-hourly (18 steps)
# MAGIC - Hours 150-360: 6-hourly (36 steps)

# COMMAND ----------

# Generate and display forecast steps
steps = generate_forecast_steps()
hour_to_index = build_hour_to_index_map(steps)

print(f"Total forecast steps: {len(steps)}")
print(f"\nFirst 15 steps (hourly): {steps[:15]}")
print(f"\nTransition to 3-hourly (88-102): {[s for s in steps if 88 <= s <= 102]}")
print(f"\nTransition to 6-hourly (141-156): {[s for s in steps if 141 <= s <= 156]}")
print(f"\nLast 10 steps: {steps[-10:]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Zarr Store (Local SSD)
# MAGIC
# MAGIC Create the pre-allocated store on local SSD for high-IOPS processing.

# COMMAND ----------

# Get current forecast cycle time (rounded to nearest 6 hours)
now = datetime.utcnow()
cycle_hour = (now.hour // 6) * 6
reference_time = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

print(f"Initializing Zarr store for forecast cycle: {reference_time}")
print(f"Output path: {LOCAL_ZARR_PATH}")
print(f"Variables: {VARIABLES}")
print(f"Dimensions: (step={len(steps)}, ensemble={ENSEMBLE_MEMBERS}, lat={LAT_SIZE}, lon={LON_SIZE})")

# COMMAND ----------

import time

start_time = time.time()

store = initialize_zarr_store(
    output_path=LOCAL_ZARR_PATH,
    variables=VARIABLES,
    ensemble_members=ENSEMBLE_MEMBERS,
    lat_size=LAT_SIZE,
    lon_size=LON_SIZE,
    reference_time=reference_time,
)

elapsed = time.time() - start_time
print(f"\n✓ Zarr store initialized in {elapsed:.2f} seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Store Structure

# COMMAND ----------

# Get store info
info = get_zarr_store_info(LOCAL_ZARR_PATH)

print("Store Information:")
print(f"  Path: {info['path']}")
print(f"  Dimensions: {info['dimensions']}")
print(f"  Variables: {info['variables']}")
print(f"  Coordinates: {info['coordinates']}")
print(f"\nChunking:")
for var, chunks in info['chunks'].items():
    print(f"  {var}: {chunks}")
print(f"\nNaN status (should be 100% initially):")
for var, nan_info in info['nan_counts'].items():
    print(f"  {var}: {nan_info['sample_nan_percentage']:.1f}% NaN")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sync to Cloud Storage (Optional)
# MAGIC
# MAGIC For the initial empty store, we can sync to cloud storage.
# MAGIC During operation, only changed chunks will be synced.

# COMMAND ----------

# Optional: Copy initial store to cloud
# This uses dbutils for the initial copy since the store is empty

# dbutils.fs.cp(f"file:{LOCAL_ZARR_PATH}", CLOUD_ZARR_PATH, recurse=True)
# print(f"✓ Store synced to {CLOUD_ZARR_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Success Criteria Validation
# MAGIC
# MAGIC Per the architecture document, initialization should complete in < 30 seconds.

# COMMAND ----------

target_time = 30.0  # seconds

if elapsed < target_time:
    print(f"✓ PASS: Initialization completed in {elapsed:.2f}s (target: <{target_time}s)")
else:
    print(f"✗ FAIL: Initialization took {elapsed:.2f}s (target: <{target_time}s)")

# Calculate expected chunk count
chunk_count = len(steps) * len(VARIABLES)
print(f"\nExpected chunks per variable: {len(steps)}")
print(f"Total chunks across all variables: {chunk_count}")

# Estimate data size
bytes_per_chunk = ENSEMBLE_MEMBERS * LAT_SIZE * LON_SIZE * 4  # float32
total_bytes = chunk_count * bytes_per_chunk
print(f"\nChunk size: {bytes_per_chunk / 1024 / 1024:.2f} MB")
print(f"Total store size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")

