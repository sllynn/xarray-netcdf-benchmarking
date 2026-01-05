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
# MAGIC 1. Write small manifest JSON to landing zone via FUSE
# MAGIC 2. Atomic rename of GRIB from staging to landing via FUSE `os.replace()`
# MAGIC
# MAGIC Uses FUSE operations (not Azure SDK) so that file notifications are triggered
# MAGIC for AutoLoader to detect new files. FUSE rename is ~0.6s per 38MB file.

# COMMAND ----------

print(f"Releasing {len(staged)} GRIBs in {MODE} mode...")
if MODE == "steady":
    print(f"  Expected duration: {len(staged) * STEADY_INTERVAL_S:.0f}s")
print("  Using FUSE for releases (triggers file notifications for AutoLoader)")

release_start = time.time()

emitted, timings = release_staged_gribs(
    staged_gribs=staged,
    mode=MODE,
    steady_interval_s=STEADY_INTERVAL_S,
    renamer=None,  # Use FUSE os.replace() to trigger file notifications
    collect_timings=True,  # Collect timing breakdown for analysis
)

release_elapsed = time.time() - release_start
print(f"✓ Released {len(emitted)} files in {release_elapsed:.1f}s ({len(emitted)/release_elapsed:.1f} files/sec)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Release Timing Breakdown

# COMMAND ----------

if timings:
    manifest_times = [t.manifest_write_s for t in timings]
    rename_times = [t.grib_rename_s for t in timings]
    total_times = [t.total_s for t in timings]
    
    print("Release Timing Breakdown:")
    print("=" * 50)
    print(f"  Manifest write:  avg={sum(manifest_times)/len(manifest_times):.3f}s  "
          f"min={min(manifest_times):.3f}s  max={max(manifest_times):.3f}s")
    print(f"  GRIB rename:     avg={sum(rename_times)/len(rename_times):.3f}s  "
          f"min={min(rename_times):.3f}s  max={max(rename_times):.3f}s")
    print(f"  Total per file:  avg={sum(total_times)/len(total_times):.3f}s  "
          f"min={min(total_times):.3f}s  max={max(total_times):.3f}s")
    print()
    
    avg_total = sum(total_times) / len(total_times)
    if avg_total < 1.0:
        print(f"✓ Average release time ({avg_total:.3f}s) is under 1s target")
    else:
        print(f"✗ Average release time ({avg_total:.3f}s) exceeds 1s target")
        print(f"  Bottleneck: {'Manifest write' if sum(manifest_times) > sum(rename_times) else 'GRIB rename'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("Producer Summary")
print("=" * 60)
print(f"Files emitted: {len(emitted)}")
print(f"Mode: {MODE}")
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
