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
# VOLUME_NAME = "netcdf"
VOLUME_NAME = "netcdf-grs"

LANDING_ZONE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/landing/"

# Local staging directory (fast SSD, not a Volume)
LOCAL_STAGING_DIR = "/local_disk0/grib_staging"

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

from datetime import datetime
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

# MAGIC %sh ls -l '/local_disk0/grib_staging' | wc -l

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2: Copy GRIBs to landing zone (slow, one-time)
# MAGIC
# MAGIC This copies GRIBs directly to the landing zone using azcopy.
# MAGIC AutoLoader only watches for manifest files (`*.grib2.json`), so GRIBs
# MAGIC without manifests won't trigger any processing until Phase 3.

# COMMAND ----------

print(f"Copying {len(prepared)} GRIBs to {LANDING_ZONE} via azcopy...")
print(f"  (AutoLoader watches manifests, not GRIBs, so these won't trigger processing)")
stage_start = time.time()

staged = stage_gribs_to_landing(
    local_staging_dir=LOCAL_STAGING_DIR,
    landing_dir=LANDING_ZONE,
    prepared=prepared,
)

stage_elapsed = time.time() - stage_start
print(f"✓ Copied {len(staged)} GRIBs in {stage_elapsed:.1f}s ({len(staged)/stage_elapsed:.1f} files/sec)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 3: Release manifests at scheduled intervals
# MAGIC
# MAGIC GRIBs are already in the landing zone (from Phase 2). Each release only writes
# MAGIC a small manifest JSON file, which triggers file notifications for AutoLoader.
# MAGIC This is fast (~10ms per manifest) enabling precise timing control.

# COMMAND ----------

print(f"Releasing {len(staged)} GRIBs in {MODE} mode...")
print(f"Release start time (ISO): {datetime.now().isoformat()}")
if MODE == "steady":
    print(f"  Expected duration: {len(staged) * STEADY_INTERVAL_S:.0f}s")

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
    total_times = [t.total_s for t in timings]
    
    print("Release Timing Breakdown:")
    print("=" * 50)
    print(f"  Manifest write:  avg={sum(manifest_times)/len(manifest_times)*1000:.1f}ms  "
          f"min={min(manifest_times)*1000:.1f}ms  max={max(manifest_times)*1000:.1f}ms")
    print(f"  Total per file:  avg={sum(total_times)/len(total_times)*1000:.1f}ms  "
          f"min={min(total_times)*1000:.1f}ms  max={max(total_times)*1000:.1f}ms")
    print()
    
    avg_total = sum(total_times) / len(total_times)
    if avg_total < 0.1:
        print(f"✓ Average release time ({avg_total*1000:.1f}ms) well under 1s target")

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
