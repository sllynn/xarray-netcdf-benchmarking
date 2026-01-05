# Databricks notebook source
# MAGIC %md
# MAGIC # Mock Producer (GRIB emitter)
# MAGIC
# MAGIC Writes GRIB files into the landing zone on a controlled schedule to test
# MAGIC the streaming pipeline on incrementally arriving data.
# MAGIC
# MAGIC Correlation strategy
# MAGIC - Each emitted file has a unique `file_id` encoded in the filename.
# MAGIC - A `.json` sidecar manifest is written next to the GRIB with producer timestamps.
# MAGIC
# MAGIC Arrival modes
# MAGIC - steady: 1 file / second
# MAGIC - burst: 200 files back-to-back

# COMMAND ----------

# MAGIC %pip install uv

# COMMAND ----------

# MAGIC %sh uv pip install -r ../requirements.lock

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "stuart"
SCHEMA = "lseg"
VOLUME_NAME = "netcdf"

LANDING_ZONE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/landing/"

# Local staging directory (fast SSD, not a Volume)
LOCAL_STAGING_DIR = "/local_disk0/grib_staging"

# Volume staging directory - OUTSIDE landing zone to avoid AutoLoader picking up files early
# This is a sibling directory to landing, not a child
VOLUME_STAGING_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/_grib_staging/"

# Arrival config
MODE = "steady"  # 'steady' or 'burst'
STEADY_INTERVAL_S = 1.0

# Release method: 'fuse' or 'azure_sdk'
# - 'fuse': Uses os.replace() through FUSE mount (~2s per file due to overhead)
# - 'azure_sdk': Uses Azure Data Lake SDK directly (~100ms per file, bypasses FUSE)
RELEASE_METHOD = "azure_sdk"

# What to emit
VARIABLES = ["t2m", "u10", "v10", "sp"]
FORECAST_HOURS = [0, 1, 2, 3, 4, 5, 6, 9, 12]  # keep small for quick smoke tests

# Burst config
BURST_COUNT = 200

# Generation parallelism
NUM_WORKERS = 8

print(f"Landing zone: {LANDING_ZONE}")
print(f"Local staging (SSD): {LOCAL_STAGING_DIR}")
print(f"Volume staging: {VOLUME_STAGING_DIR}")
print(f"Mode: {MODE}")
print(f"Release method: {RELEASE_METHOD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 1: Generate GRIBs to local disk (fast, parallel)
# MAGIC
# MAGIC This writes to local SSD, not the Volume, so it's fast.

# COMMAND ----------

import itertools
import shutil
import time
from pathlib import Path

from src.benchmarks.streaming_harness import (
    prepare_all_gribs_locally,
    stage_gribs_to_landing,
    release_staged_gribs,
    AzureDataLakeRenamer,
)

# Clear local staging dir
if Path(LOCAL_STAGING_DIR).exists():
    shutil.rmtree(LOCAL_STAGING_DIR)
Path(LOCAL_STAGING_DIR).mkdir(parents=True, exist_ok=True)

# Build the plan
if MODE == "burst":
    base_plan = list(itertools.product(VARIABLES, FORECAST_HOURS))
    plan = (base_plan * ((BURST_COUNT // len(base_plan)) + 1))[:BURST_COUNT]
else:
    plan = list(itertools.product(VARIABLES, FORECAST_HOURS))

print(f"Generating {len(plan)} GRIBs to local disk...")
gen_start = time.time()

prepared = prepare_all_gribs_locally(
    local_staging_dir=LOCAL_STAGING_DIR,
    plan=plan,
    workers=NUM_WORKERS,
)

gen_elapsed = time.time() - gen_start
print(f"✓ Generated {len(prepared)} GRIBs in {gen_elapsed:.1f}s ({len(prepared)/gen_elapsed:.1f} files/sec)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2: Copy GRIBs to Volume staging (slow, one-time)
# MAGIC
# MAGIC This copies files to a staging directory **outside** the landing zone.
# MAGIC This prevents AutoLoader from picking up files prematurely.
# MAGIC
# MAGIC Staging location: `_grib_staging/` (sibling of landing, not child)

# COMMAND ----------

Path(LANDING_ZONE).mkdir(parents=True, exist_ok=True)

print(f"Staging {len(prepared)} GRIBs to {VOLUME_STAGING_DIR} via azcopy...")
print(f"  (Outside landing zone to avoid AutoLoader)")
stage_start = time.time()

staged = stage_gribs_to_landing(
    local_staging_dir=LOCAL_STAGING_DIR,
    landing_dir=LANDING_ZONE,
    prepared=prepared,
    volume_staging_dir=VOLUME_STAGING_DIR,
)

stage_elapsed = time.time() - stage_start
print(f"✓ Staged {len(staged)} GRIBs in {stage_elapsed:.1f}s ({len(staged)/stage_elapsed:.1f} files/sec)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 3: Release GRIBs at scheduled intervals
# MAGIC
# MAGIC Now the GRIBs are staged on the Volume. Each release does:
# MAGIC 1. Write small manifest JSON to landing zone
# MAGIC 2. Atomic rename of GRIB from staging to landing
# MAGIC
# MAGIC When using `azure_sdk` method, this bypasses FUSE for much faster releases (~100ms vs ~2s).

# COMMAND ----------

print(f"Releasing {len(staged)} GRIBs in {MODE} mode...")
print(f"  Release method: {RELEASE_METHOD}")
if MODE == "steady":
    print(f"  Expected duration: {len(staged) * STEADY_INTERVAL_S:.0f}s")

# Initialize Azure SDK renamer if using that method
renamer = None
if RELEASE_METHOD == "azure_sdk":
    print("  Initializing Azure Data Lake SDK renamer...")
    volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
    renamer = AzureDataLakeRenamer.from_volume_path(volume_path)
    print(f"  ✓ Connected to storage account: {renamer.account_name}")

release_start = time.time()

emitted = release_staged_gribs(
    staged_gribs=staged,
    mode=MODE,
    steady_interval_s=STEADY_INTERVAL_S,
    method=RELEASE_METHOD,
    renamer=renamer,
)

release_elapsed = time.time() - release_start
print(f"✓ Released {len(emitted)} files in {release_elapsed:.1f}s ({len(emitted)/release_elapsed:.1f} files/sec)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("Producer Summary")
print("=" * 60)
print(f"Files emitted: {len(emitted)}")
print(f"Mode: {MODE}")
print(f"Release method: {RELEASE_METHOD}")
if MODE == "steady":
    print(f"Target interval: {STEADY_INTERVAL_S}s")
    print(f"Actual rate: {len(emitted)/release_elapsed:.2f} files/sec")
print()
print(f"Phase 1 (local generation): {gen_elapsed:.1f}s")
print(f"Phase 2 (staging to Volume): {stage_elapsed:.1f}s")
print(f"Phase 3 (timed release): {release_elapsed:.1f}s")
print()
print("Sample files:")
for e in emitted[:3]:
    print(f"  {e.landing_path}")
