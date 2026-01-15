# Databricks notebook source
# MAGIC %md
# MAGIC # Zarr Read Benchmarks (Volumes)
# MAGIC
# MAGIC Benchmarks:
# MAGIC - metadata open time
# MAGIC - single-chunk vs multi-chunk slice reads
# MAGIC - scaling with number of Zarr stores
# MAGIC
# MAGIC Results are written directly under `/Volumes/<catalog>/<schema>/<volume_name>/...` (no `/dbfs`).

# COMMAND ----------

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.benchmarks.consumer_read import (
    benchmark_metadata_open,
    benchmark_slice_scaling,
    benchmark_multi_store_scaling,
    save_metadata_open_results,
    save_slice_scaling_results,
    save_multi_store_results,
)
from src.benchmarks.zarr_fixtures import generate_forecast_cycle_zarrs_local_and_sync

# COMMAND ----------

# Configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"
VOLUME_NAME = "your_volume"

BASE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/read_benchmarks"
ZARR_BASE_PATH = f"{BASE_PATH}/zarr_fixtures"
LOCAL_FIXTURE_PATH = "/local_disk0/zarr_fixtures"
RESULTS_PATH = f"{BASE_PATH}/results"

CREATE_FIXTURES = False
FIXTURE_START_TIME = "2025-12-29T00:00:00"
FIXTURE_NUM_DAYS = 7
FIXTURE_CYCLE_HOURS = 6
FIXTURE_OVERWRITE = False

# Slice scaling config
STEP_COUNTS = [1, 4, 12, 24]
ENSEMBLE_COUNTS = [1, 10, 50]
SPATIAL_SIZES = [(50, 100), (180, 360), (361, 720)]
TESTS_PER_CASE = 3

# Multi-store config
STORE_COUNTS = [1, 2, 4, 8, 16, 28]
MULTI_STORE_STEP_COUNT = 1
MULTI_STORE_ENSEMBLE_COUNT = 50
MULTI_STORE_SPATIAL_SIZE = (361, 720)
MULTI_STORE_TESTS_PER_CASE = 2
MULTI_STORE_USE_MFDATASET = True
MULTI_STORE_CONCAT_DIM = "cycle"

# Metadata open config
METADATA_OPEN_ITERATIONS = 10

# COMMAND ----------

# Optional: generate weekly fixtures locally and sync to Volume
if CREATE_FIXTURES:
    print("Creating forecast-cycle Zarr fixtures...")
    generated = generate_forecast_cycle_zarrs_local_and_sync(
        local_base_path=LOCAL_FIXTURE_PATH,
        volume_target_path=ZARR_BASE_PATH,
        start_time=FIXTURE_START_TIME,
        num_days=FIXTURE_NUM_DAYS,
        cycle_hours=FIXTURE_CYCLE_HOURS,
        overwrite=FIXTURE_OVERWRITE,
    )
    print(f"Created {len(generated)} stores locally and synced to Volume")

# COMMAND ----------

# Discover Zarr stores
zarr_paths = sorted(str(p) for p in Path(ZARR_BASE_PATH).glob("*.zarr"))
if not zarr_paths:
    raise FileNotFoundError(f"No Zarr stores found under {ZARR_BASE_PATH}")

SINGLE_ZARR_PATH = zarr_paths[0]
print(f"Using single Zarr store: {SINGLE_ZARR_PATH}")
print(f"Found {len(zarr_paths)} Zarr stores for multi-store tests")

# COMMAND ----------

# Benchmark 1: metadata open time
print("=" * 60)
print("BENCHMARK 1: Metadata Open Time")
print("=" * 60)

metadata_result = benchmark_metadata_open(
    SINGLE_ZARR_PATH,
    iterations=METADATA_OPEN_ITERATIONS,
    consolidated=True,
)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
metadata_json = f"{RESULTS_PATH}/metadata_open_{timestamp}.json"
save_metadata_open_results(metadata_result, metadata_json)

metadata_df = pd.DataFrame([
    {"open_time_ms": r.open_time_ms, "success": r.success}
    for r in metadata_result.results
])
metadata_csv = f"{RESULTS_PATH}/metadata_open_{timestamp}.csv"
Path(metadata_csv).parent.mkdir(parents=True, exist_ok=True)
metadata_df.to_csv(metadata_csv, index=False)
print(f"Saved: {metadata_json}")
print(f"Saved: {metadata_csv}")

# COMMAND ----------

# Benchmark 2: slice scaling (single-chunk vs multi-chunk grids)
print("=" * 60)
print("BENCHMARK 2: Slice Scaling")
print("=" * 60)

slice_result = benchmark_slice_scaling(
    zarr_store_path=SINGLE_ZARR_PATH,
    step_counts=STEP_COUNTS,
    ensemble_counts=ENSEMBLE_COUNTS,
    spatial_sizes=SPATIAL_SIZES,
    tests_per_case=TESTS_PER_CASE,
    consolidated=True,
    align_to_chunks=True,
)

slice_json = f"{RESULTS_PATH}/slice_scaling_{timestamp}.json"
save_slice_scaling_results(slice_result, slice_json)

slice_summary_df = pd.DataFrame([r.__dict__ for r in slice_result.case_summaries])
slice_summary_csv = f"{RESULTS_PATH}/slice_scaling_summary_{timestamp}.csv"
slice_summary_df.to_csv(slice_summary_csv, index=False)
print(f"Saved: {slice_json}")
print(f"Saved: {slice_summary_csv}")

# COMMAND ----------

# Benchmark 3: multi-store scaling
print("=" * 60)
print("BENCHMARK 3: Multi-Store Scaling")
print("=" * 60)

store_counts = [c for c in STORE_COUNTS if c <= len(zarr_paths)]
multi_store_result = benchmark_multi_store_scaling(
    zarr_paths=zarr_paths,
    store_counts=store_counts,
    step_count=MULTI_STORE_STEP_COUNT,
    ensemble_count=MULTI_STORE_ENSEMBLE_COUNT,
    spatial_size=MULTI_STORE_SPATIAL_SIZE,
    tests_per_case=MULTI_STORE_TESTS_PER_CASE,
    consolidated=True,
    align_to_chunks=True,
    use_mfdataset=MULTI_STORE_USE_MFDATASET,
    concat_dim=MULTI_STORE_CONCAT_DIM,
)

multi_store_json = f"{RESULTS_PATH}/multi_store_scaling_{timestamp}.json"
save_multi_store_results(multi_store_result, multi_store_json)

multi_store_summary_df = pd.DataFrame([r.__dict__ for r in multi_store_result.case_summaries])
multi_store_summary_csv = f"{RESULTS_PATH}/multi_store_scaling_summary_{timestamp}.csv"
multi_store_summary_df.to_csv(multi_store_summary_csv, index=False)
print(f"Saved: {multi_store_json}")
print(f"Saved: {multi_store_summary_csv}")

# COMMAND ----------

# Summary
print("=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"Metadata open p50: {metadata_result.p50_open_time_ms:.1f} ms")
print(f"Metadata open p95: {metadata_result.p95_open_time_ms:.1f} ms")
print(f"Slice cases: {len(slice_result.case_summaries)}")
print(f"Multi-store cases: {len(multi_store_result.case_summaries)}")
print(f"Results path: {RESULTS_PATH}")
