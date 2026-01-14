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

# MAGIC %restart_python

# COMMAND ----------

# Configuration
CATALOG = "stuart"
SCHEMA = "lseg"
VOLUME_NAME = "netcdf"

# Paths
LOCAL_ZARR_PATH = "/tmp/forecast.zarr"
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
import time

sys.path.insert(0, '/Workspace/Repos/your_user/raster-benchmarking')

from datetime import datetime, timezone
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
now = datetime.now(timezone.utc)
cycle_hour = (now.hour // 6) * 6
reference_time = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

print(f"Initializing Zarr store for forecast cycle: {reference_time}")
print(f"Output path: {LOCAL_ZARR_PATH}")
print(f"Variables: {VARIABLES}")
print(f"Dimensions: (step={len(steps)}, ensemble={ENSEMBLE_MEMBERS}, lat={LAT_SIZE}, lon={LON_SIZE})")

# COMMAND ----------

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
# MAGIC
# MAGIC **Important:** `dbutils.fs.cp` is extremely slow for Zarr stores (many small files).
# MAGIC Use `azcopy` instead for 10-50x faster transfers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Fast sync with azcopy (recommended)

# COMMAND ----------

# Get SAS token for the destination Volume using Databricks SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import PathOperation

w = WorkspaceClient()

volume_root_url = w.volumes.read(f"{CATALOG}.{SCHEMA}.{VOLUME_NAME}").storage_location
zarr_archive_url = CLOUD_ZARR_PATH.replace(f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}", volume_root_url)

print(f"Destination: {zarr_archive_url}")

# Generate temporary write credentials for the Volume
creds = w.temporary_path_credentials.generate_temporary_path_credentials(
        url=volume_root_url,
        operation=PathOperation.PATH_READ_WRITE,
)

# Get the Azure SAS token
sas_token = creds.azure_user_delegation_sas.sas_token
print(f"✓ Got SAS token (expires: {creds.expiration_time})")

# COMMAND ----------

# Construct the Azure Blob URL from the Volume path
# Volume paths map to: https://<storage_account>.blob.core.windows.net/<container>/<path>
# You need to know your storage account and container - check your external location config

STORAGE_ACCOUNT = zarr_archive_url.split("@")[-1].split(".")[0]  # e.g., "mystorageaccount"
CONTAINER = zarr_archive_url.split("@")[0].split("//")[-1]              # e.g., "unity-catalog-volumes"

# Extract the subpath from the Volume path
# /Volumes/catalog/schema/volume_name/subpath -> subpath
volume_subpath = "/".join(CLOUD_ZARR_PATH.split("/")[4:])  # Skip /Volumes/catalog/schema/volume

azure_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER}/{SCHEMA}/{volume_subpath}?{sas_token}"

print(f"Destination: https://{STORAGE_ACCOUNT}.blob.core.windows.net/{CONTAINER}/{SCHEMA}/{volume_subpath}")

# COMMAND ----------

# Install azcopy if not already installed
import subprocess
import shutil
import os

azcopy_path = shutil.which("azcopy")

if not azcopy_path:
    # Check /tmp
    import glob
    existing = glob.glob("/tmp/azcopy/*/azcopy")
    if existing:
        azcopy_path = existing[0]

if not azcopy_path:
    print("Installing azcopy...")
    os.makedirs("/tmp/azcopy", exist_ok=True)
    
    # Download and extract azcopy
    subprocess.run([
        "curl", "-sL", 
        "https://aka.ms/downloadazcopy-v10-linux",
        "-o", "/tmp/azcopy/azcopy.tar.gz"
    ], check=True)
    
    subprocess.run([
        "tar", "-xzf", "/tmp/azcopy/azcopy.tar.gz",
        "-C", "/tmp/azcopy"
    ], check=True)
    
    # Find the extracted binary
    azcopy_path = glob.glob("/tmp/azcopy/*/azcopy")[0]
    print(f"✓ azcopy installed: {azcopy_path}")
else:
    print(f"✓ azcopy already available: {azcopy_path}")

# Verify
result = subprocess.run([azcopy_path, "--version"], capture_output=True, text=True)
print(result.stdout.strip())

# COMMAND ----------

# Run azcopy sync
import subprocess
import time

sync_start = time.time()

result = subprocess.run(
    [
        azcopy_path, "sync",
        LOCAL_ZARR_PATH,
        azure_url,
        "--recursive=true",
        "--log-level=WARNING",
    ],
    capture_output=True,
    text=True,
)

sync_elapsed = time.time() - sync_start

if result.returncode == 0:
    print(f"✓ Synced to cloud in {sync_elapsed:.1f} seconds")
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
else:
    print(f"✗ Sync failed: {result.stderr}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Slow fallback with dbutils (not recommended)

# COMMAND ----------

# WARNING: This is ~10-50x slower than azcopy for Zarr stores!
# Only use if azcopy is not available

# sync_start = time.time()
# dbutils.fs.cp(f"file:{LOCAL_ZARR_PATH}", CLOUD_ZARR_PATH, recurse=True)
# sync_elapsed = time.time() - sync_start
# print(f"✓ Store synced in {sync_elapsed:.1f}s (slow method)")

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

