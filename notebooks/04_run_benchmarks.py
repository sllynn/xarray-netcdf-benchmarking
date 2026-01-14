# Databricks notebook source
# MAGIC %md
# MAGIC # Comprehensive Pipeline Benchmarks
# MAGIC 
# MAGIC This notebook runs the complete benchmarking suite to validate
# MAGIC pipeline performance against the architecture requirements:
# MAGIC 
# MAGIC | Component | Target |
# MAGIC |-----------|--------|
# MAGIC | Zarr Initialization | < 30 seconds |
# MAGIC | Region Write | < 500ms per file |
# MAGIC | AzCopy Sync | < 5 seconds for 50 chunks |
# MAGIC | End-to-End | < 30 seconds total |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"

# Paths
LOCAL_ZARR_PATH = "/local_disk0/benchmark.zarr"
LANDING_ZONE = f"/Volumes/{CATALOG}/{SCHEMA}/bronze/grib/"
CLOUD_DESTINATION = f"/Volumes/{CATALOG}/{SCHEMA}/silver/"
RESULTS_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/benchmark_results/"

# Benchmark parameters
REGION_WRITE_TESTS = 50
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
SYNC_CHUNK_COUNTS = [1, 5, 10, 25, 50]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import sys
sys.path.insert(0, '/Workspace/Repos/your_user/raster-benchmarking')

import time
from datetime import datetime
from pathlib import Path

from src.zarr_init import initialize_zarr_store, generate_forecast_steps
from src.benchmarks.region_write import (
    benchmark_region_writes,
    create_test_zarr_store,
    save_results as save_region_results,
)
from src.benchmarks.azcopy_sync import (
    benchmark_azcopy_sync,
    create_test_files,
    save_results as save_sync_results,
)
from src.benchmarks.consumer_read import (
    benchmark_consumer_reads,
    save_results as save_consumer_results,
)
from src.benchmarks.e2e_latency import (
    benchmark_e2e_latency,
    BenchmarkConfig,
    save_results as save_e2e_results,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Zarr Initialization Benchmark

# COMMAND ----------

print("=" * 60)
print("BENCHMARK 1: Zarr Store Initialization")
print("=" * 60)
print()

# Test with full production dimensions
variables = ["t2m", "u10", "v10", "sp"]
ensemble_members = 50
lat_size = 361
lon_size = 720

steps = generate_forecast_steps()
print(f"Configuration:")
print(f"  Variables: {variables}")
print(f"  Steps: {len(steps)}")
print(f"  Ensemble: {ensemble_members}")
print(f"  Spatial: {lat_size} x {lon_size}")

# Clean up existing store
if Path(LOCAL_ZARR_PATH).exists():
    import shutil
    shutil.rmtree(LOCAL_ZARR_PATH)

# Run initialization
start_time = time.time()

store = initialize_zarr_store(
    output_path=LOCAL_ZARR_PATH,
    variables=variables,
    ensemble_members=ensemble_members,
    lat_size=lat_size,
    lon_size=lon_size,
    reference_time=datetime.utcnow(),
)

init_time = time.time() - start_time

print(f"\n✓ Initialization completed in {init_time:.2f} seconds")

# Validate
TARGET_INIT_TIME = 30.0
if init_time < TARGET_INIT_TIME:
    print(f"✓ PASS: {init_time:.2f}s < {TARGET_INIT_TIME}s target")
else:
    print(f"✗ FAIL: {init_time:.2f}s >= {TARGET_INIT_TIME}s target")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Region Write Performance Benchmark

# COMMAND ----------

print("=" * 60)
print("BENCHMARK 2: Region Write Performance")
print("=" * 60)
print()

print(f"Configuration:")
print(f"  Writes per concurrency level: {REGION_WRITE_TESTS}")
print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")

# Run benchmark
region_result = benchmark_region_writes(
    zarr_store_path=LOCAL_ZARR_PATH,
    num_writes=REGION_WRITE_TESTS,
    concurrency_levels=CONCURRENCY_LEVELS,
    variable='t2m',
)

print(f"\nResults:")
print(f"  Optimal concurrency: {region_result.optimal_concurrency}")
print(f"  Best throughput: {region_result.optimal_throughput_writes_per_sec:.1f} writes/s")
print(f"  Chunk size: {region_result.chunk_size_mb:.2f} MB")

print(f"\nBy concurrency level:")
for cr in region_result.concurrency_results:
    print(f"  {cr.concurrency:2d} workers: {cr.avg_write_time_ms:6.1f}ms avg, "
          f"{cr.writes_per_second:5.1f} writes/s")

# Validate
TARGET_WRITE_TIME = 500.0  # ms
best_avg_time = min(cr.avg_write_time_ms for cr in region_result.concurrency_results if cr.successful_writes > 0)
if best_avg_time < TARGET_WRITE_TIME:
    print(f"\n✓ PASS: Best avg write time {best_avg_time:.1f}ms < {TARGET_WRITE_TIME}ms target")
else:
    print(f"\n✗ FAIL: Best avg write time {best_avg_time:.1f}ms >= {TARGET_WRITE_TIME}ms target")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_region_results(
    region_result,
    f"/dbfs{RESULTS_PATH}/region_write_benchmark_{timestamp}.json"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. AzCopy Sync Benchmark

# COMMAND ----------

print("=" * 60)
print("BENCHMARK 3: AzCopy Sync Performance")
print("=" * 60)
print()

# Check if azcopy is available
from src.cloud_sync import find_azcopy

azcopy_path = find_azcopy()
use_mock = azcopy_path is None

if use_mock:
    print("⚠️  azcopy not found, using simulated results")
else:
    print(f"✓ Found azcopy at: {azcopy_path}")

print(f"\nConfiguration:")
print(f"  Chunk counts to test: {SYNC_CHUNK_COUNTS}")
print(f"  Chunk size: ~50 MB")

# Create test directory
import tempfile
test_dir = tempfile.mkdtemp(prefix="sync_benchmark_")

# Create test files
print("\nCreating test files...")
test_files = create_test_files(
    directory=test_dir,
    num_files=max(SYNC_CHUNK_COUNTS),
    file_size_mb=50.0,
)
print(f"  Created {len(test_files)} test files")

# Run benchmark (use mock if no Azure connection)
sync_result = benchmark_azcopy_sync(
    source_path=test_dir,
    dest_url=f"https://mock.blob.core.windows.net/container/benchmark",
    changed_chunk_counts=SYNC_CHUNK_COUNTS,
    chunk_size_mb=50.0,
    repetitions=2,
    use_mock=use_mock,
)

print(f"\nResults:")
print(f"  Average sync time: {sync_result.avg_sync_time_ms:.1f}ms")
print(f"  Average throughput: {sync_result.avg_chunks_per_second:.1f} chunks/s")
print(f"  Time per chunk: {sync_result.time_per_chunk_ms:.1f}ms")
print(f"  Base overhead: {sync_result.base_overhead_ms:.1f}ms")

# Validate for 50 chunks
TARGET_SYNC_TIME = 5000.0  # 5 seconds in ms
fifty_chunk_results = [r for r in sync_result.test_results if r.num_changed_chunks == 50 and r.success]
if fifty_chunk_results:
    avg_fifty_chunk_time = sum(r.sync_time_ms for r in fifty_chunk_results) / len(fifty_chunk_results)
    if avg_fifty_chunk_time < TARGET_SYNC_TIME:
        print(f"\n✓ PASS: 50-chunk sync {avg_fifty_chunk_time:.1f}ms < {TARGET_SYNC_TIME}ms target")
    else:
        print(f"\n✗ FAIL: 50-chunk sync {avg_fifty_chunk_time:.1f}ms >= {TARGET_SYNC_TIME}ms target")

# Clean up
import shutil
shutil.rmtree(test_dir)

# Save results
save_sync_results(
    sync_result,
    f"/dbfs{RESULTS_PATH}/azcopy_sync_benchmark_{timestamp}.json"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Consumer Read Benchmark

# COMMAND ----------

print("=" * 60)
print("BENCHMARK 4: Consumer Read Latency")
print("=" * 60)
print()

print(f"Configuration:")
print(f"  Access patterns: single_step, time_range, spatial_subset, ensemble_mean")
print(f"  Tests per pattern: 5")

# Run benchmark
consumer_result = benchmark_consumer_reads(
    zarr_store_path=LOCAL_ZARR_PATH,
    nan_percentage=0,  # Assume fully populated for read benchmark
    access_patterns=['single_step', 'time_range', 'spatial_subset', 'ensemble_mean'],
    tests_per_pattern=5,
)

print(f"\nResults:")
print(f"  Overall avg latency: {consumer_result.overall_avg_latency_ms:.1f}ms")
print(f"  Overall throughput: {consumer_result.overall_throughput_mb_per_sec:.1f} MB/s")

print(f"\nBy access pattern:")
for pr in consumer_result.pattern_results:
    print(f"  {pr.access_pattern:15s}: avg={pr.avg_total_time_ms:6.1f}ms, "
          f"p95={pr.p95_total_time_ms:6.1f}ms, "
          f"{pr.mb_per_second:.1f} MB/s")

# Save results
save_consumer_results(
    consumer_result,
    f"/dbfs{RESULTS_PATH}/consumer_read_benchmark_{timestamp}.json"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. End-to-End Latency Benchmark

# COMMAND ----------

print("=" * 60)
print("BENCHMARK 5: End-to-End Latency")
print("=" * 60)
print()

# Check for GRIB files
grib_files = dbutils.fs.ls(LANDING_ZONE) if LANDING_ZONE.startswith("/Volumes") else []
grib_paths = [f.path.replace("dbfs:", "/dbfs") for f in grib_files if f.name.endswith(('.grib', '.grib2'))]

if grib_paths:
    print(f"Found {len(grib_paths)} GRIB files for E2E benchmark")
    
    # Create E2E config
    e2e_config = BenchmarkConfig(
        zarr_store_path=LOCAL_ZARR_PATH,
        landing_zone=tempfile.mkdtemp(prefix="e2e_landing_"),
        cloud_destination=None,  # Skip cloud sync for local benchmark
        num_files=min(10, len(grib_paths)),
        arrival_pattern='burst',
    )
    
    print(f"\nConfiguration:")
    print(f"  Files: {e2e_config.num_files}")
    print(f"  Arrival pattern: {e2e_config.arrival_pattern}")
    
    # Run benchmark
    e2e_result = benchmark_e2e_latency(
        config=e2e_config,
        source_grib_files=grib_paths[:e2e_config.num_files],
    )
    
    print(f"\nResults:")
    print(f"  Total latency: {e2e_result.total_latency_ms:.1f}ms")
    print(f"  Breakdown:")
    print(f"    File arrival: {e2e_result.file_arrival_time_ms:.1f}ms")
    print(f"    Discovery: {e2e_result.discovery_time_ms:.1f}ms")
    print(f"    Processing: {e2e_result.processing_time_ms:.1f}ms")
    print(f"    Sync: {e2e_result.sync_time_ms:.1f}ms")
    print(f"    Consumer read: {e2e_result.consumer_read_time_ms:.1f}ms")
    print(f"  Throughput: {e2e_result.files_per_second:.1f} files/s")
    
    # Validate
    TARGET_E2E_TIME = 30000.0  # 30 seconds in ms
    if e2e_result.total_latency_ms < TARGET_E2E_TIME:
        print(f"\n✓ PASS: E2E latency {e2e_result.total_latency_ms:.1f}ms < {TARGET_E2E_TIME}ms target")
    else:
        print(f"\n✗ FAIL: E2E latency {e2e_result.total_latency_ms:.1f}ms >= {TARGET_E2E_TIME}ms target")
    
    # Save results
    save_e2e_results(
        [e2e_result],
        f"/dbfs{RESULTS_PATH}/e2e_latency_benchmark_{timestamp}.json"
    )
    
    # Clean up
    shutil.rmtree(e2e_config.landing_zone)
else:
    print("⚠️  No GRIB files available for E2E benchmark")
    print("   Upload GRIB files to the landing zone to run this benchmark")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print()

# Collect all results
summary = {
    "Zarr Initialization": {
        "result": f"{init_time:.2f}s",
        "target": f"<{TARGET_INIT_TIME}s",
        "pass": init_time < TARGET_INIT_TIME,
    },
    "Region Write": {
        "result": f"{best_avg_time:.1f}ms",
        "target": f"<{TARGET_WRITE_TIME}ms",
        "pass": best_avg_time < TARGET_WRITE_TIME,
    },
}

if fifty_chunk_results:
    summary["AzCopy Sync (50 chunks)"] = {
        "result": f"{avg_fifty_chunk_time:.1f}ms",
        "target": f"<{TARGET_SYNC_TIME}ms",
        "pass": avg_fifty_chunk_time < TARGET_SYNC_TIME,
    }

if grib_paths:
    summary["End-to-End"] = {
        "result": f"{e2e_result.total_latency_ms:.1f}ms",
        "target": f"<{TARGET_E2E_TIME}ms",
        "pass": e2e_result.total_latency_ms < TARGET_E2E_TIME,
    }

print(f"{'Component':<25} {'Result':>15} {'Target':>15} {'Status':>10}")
print("-" * 65)

all_pass = True
for component, data in summary.items():
    status = "✓ PASS" if data["pass"] else "✗ FAIL"
    all_pass = all_pass and data["pass"]
    print(f"{component:<25} {data['result']:>15} {data['target']:>15} {status:>10}")

print("-" * 65)
overall_status = "✓ ALL PASS" if all_pass else "✗ SOME FAILED"
print(f"{'Overall':<25} {'':>15} {'':>15} {overall_status:>10}")

print(f"\nResults saved to: {RESULTS_PATH}")
print(f"Timestamp: {timestamp}")

