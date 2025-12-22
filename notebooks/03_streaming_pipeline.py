# Databricks notebook source
# MAGIC %md
# MAGIC # Streaming Pipeline with AutoLoader
# MAGIC
# MAGIC This notebook implements the full streaming ingestion pipeline:
# MAGIC - AutoLoader monitors the landing zone for new GRIB files
# MAGIC - foreachBatch processes files in parallel using ThreadPoolExecutor
# MAGIC - Region writes update specific slots in the Zarr store
# MAGIC - AzCopy syncs changed chunks to cloud storage

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
LANDING_ZONE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/landing/"
LOCAL_ZARR_PATH = "/tmp/forecast.zarr"
CLOUD_DESTINATION = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/silver/"
CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/checkpoints/grib_pipeline"

# Processing parameters
MAX_FILES_PER_BATCH = 32
NUM_WORKERS = 32
TRIGGER_INTERVAL = "0 seconds"  # Process immediately (no buffering)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Logging
# MAGIC
# MAGIC Enable logging output so we can see timing information in the notebook.

# COMMAND ----------

import logging

# Configure logging to show INFO level messages from our modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Ensure our module loggers are at INFO level
logging.getLogger('src.region_writer').setLevel(logging.INFO)
logging.getLogger('src.streaming_pipeline').setLevel(logging.INFO)
logging.getLogger('src.cloud_sync').setLevel(logging.INFO)

# Reduce noise from Spark/Py4J
logging.getLogger('py4j').setLevel(logging.WARNING)
logging.getLogger('py4j.clientserver').setLevel(logging.WARNING)
logging.getLogger('py4j.java_gateway').setLevel(logging.WARNING)

print("✓ Logging configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import sys
# sys.path.insert(0, '/Workspace/Repos/your_user/raster-benchmarking')

from datetime import datetime
from src.streaming_pipeline import (
    PipelineConfig,
    PipelineManager,
    create_streaming_pipeline,
    run_batch_processing,
)
from src.zarr_init import initialize_zarr_store, generate_forecast_steps

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Zarr Store (if needed)
# MAGIC
# MAGIC The store should be initialized at the start of each forecast cycle
# MAGIC (00:00, 06:00, 12:00, 18:00 UTC).

# COMMAND ----------

# Check if store exists, create if needed
from pathlib import Path
from datetime import timezone

if not Path(LOCAL_ZARR_PATH).exists():
    print("Initializing Zarr store...")
    
    now = datetime.now(timezone.utc)
    cycle_hour = (now.hour // 6) * 6
    reference_time = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    
    store = initialize_zarr_store(
        output_path=LOCAL_ZARR_PATH,
        variables=["t2m", "u10", "v10", "sp"],
        ensemble_members=50,
        lat_size=361,
        lon_size=720,
        reference_time=reference_time,
    )
    print(f"✓ Store initialized for cycle {reference_time}")
else:
    print(f"✓ Using existing store at {LOCAL_ZARR_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clear Checkpoint (for testing)
# MAGIC
# MAGIC Clear the checkpoint directory to reprocess all files. 
# MAGIC Comment this out in production to enable incremental processing.

# COMMAND ----------

import shutil

# Clear checkpoint to reprocess all files
try:
    dbutils.fs.rm(CHECKPOINT_PATH, recurse=True)
    print(f"✓ Cleared checkpoint: {CHECKPOINT_PATH}")
except Exception as e:
    # Checkpoint may not exist yet on first run
    print(f"Note: Could not clear checkpoint (may not exist yet): {e}")

# Also clear local Zarr store if you want a fresh start
shutil.rmtree(LOCAL_ZARR_PATH, ignore_errors=True)
print(f"✓ Cleared local Zarr store: {LOCAL_ZARR_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Pipeline Configuration

# COMMAND ----------

config = PipelineConfig(
    landing_zone=LANDING_ZONE,
    zarr_store_path=LOCAL_ZARR_PATH,
    cloud_destination=CLOUD_DESTINATION,
    checkpoint_path=CHECKPOINT_PATH,
    max_files_per_batch=MAX_FILES_PER_BATCH,
    num_workers=NUM_WORKERS,
    trigger_interval=TRIGGER_INTERVAL,
    use_file_notification=True,  # Low-latency file discovery
    sync_after_each_batch=True,  # Sync to cloud after each batch
)

print("Pipeline Configuration:")
print(f"  Landing zone: {config.landing_zone}")
print(f"  Zarr store: {config.zarr_store_path}")
print(f"  Cloud destination: {config.cloud_destination}")
print(f"  Max files per batch: {config.max_files_per_batch}")
print(f"  Workers: {config.num_workers}")
print(f"  Trigger: {config.trigger_interval}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Start Streaming Pipeline
# MAGIC
# MAGIC The pipeline will run continuously, processing GRIB files as they arrive.

# COMMAND ----------

# Create pipeline manager
manager = PipelineManager(spark, config)

# Start the pipeline
manager.start()
  
print(f"✓ Pipeline started")
print(f"  Query ID: {manager.query.id}")
print(f"  Status: {manager.status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitor Pipeline Progress
# MAGIC
# MAGIC Use this cell to monitor the pipeline while it's running.

# COMMAND ----------

import time

# Monitor for a period of time
monitor_duration = 300  # seconds
check_interval = 5  # seconds

print(f"Monitoring pipeline for {monitor_duration} seconds...")
print()

start_time = time.time()
last_batch_count = 0

while time.time() - start_time < monitor_duration:
    status = manager.status
    metrics = manager.get_metrics()
    
    current_batch_count = metrics.get('batches_processed', 0)
    
    if current_batch_count > last_batch_count:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch processed!")
        print(f"  Total batches: {current_batch_count}")
        print(f"  Files processed: {metrics.get('total_files_processed', 0)}")
        print(f"  Success rate: {metrics.get('total_files_successful', 0)}/{metrics.get('total_files_processed', 0)}")
        if metrics.get('avg_processing_time_ms'):
            print(f"  Avg processing time: {metrics['avg_processing_time_ms']:.1f}ms")
        print()
        last_batch_count = current_batch_count
    
    time.sleep(check_interval)

print("Monitoring complete")
print(f"\nFinal metrics: {manager.get_metrics()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stop Pipeline (when done)

# COMMAND ----------

# Stop the pipeline when you're done
manager.stop()
print("✓ Pipeline stopped")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Batch Processing Mode
# MAGIC
# MAGIC For testing or one-time processing, you can use batch mode instead of streaming.

# COMMAND ----------

# List files for batch processing
from pyspark.sql.functions import col

# Get list of GRIB files (filter to exclude .idx sidecar files and other artifacts)
grib_files = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.{grib,grib2,grb,grb2}")
    .load(LANDING_ZONE)
)
file_paths = [row.path for row in grib_files.select("path").collect()]

print(f"Found {len(file_paths)} GRIB files for batch processing")

# COMMAND ----------

# Run batch processing (non-streaming)
if file_paths:
    # Convert to local paths
    local_paths = [p.replace("dbfs:", "/dbfs") for p in file_paths[:20]]  # Limit for testing
    
    print(f"Processing {len(local_paths)} files in batch mode...")
    
    batch_result = run_batch_processing(
        spark=spark,
        config=config,
        file_paths=local_paths,
    )
    
    print(f"\n✓ Batch complete")
    print(f"  Files processed: {batch_result.files_processed}")
    print(f"  Successful: {batch_result.files_successful}")
    print(f"  Failed: {batch_result.files_failed}")
    print(f"  Processing time: {batch_result.processing_time_ms:.1f}ms")
    print(f"  Sync time: {batch_result.sync_time_ms:.1f}ms")
    print(f"  Total time: {batch_result.total_time_ms:.1f}ms")
    
    if batch_result.errors:
        print(f"\nErrors:")
        for error in batch_result.errors[:5]:
            print(f"  {error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Success Criteria Validation
# MAGIC
# MAGIC Per the architecture document:
# MAGIC - End-to-end latency should be < 30 seconds

# COMMAND ----------

metrics = manager.get_metrics()

if metrics.get('avg_processing_time_ms') and metrics.get('avg_sync_time_ms'):
    avg_latency = metrics['avg_processing_time_ms'] + metrics['avg_sync_time_ms']
    target_latency = 30000  # 30 seconds in ms
    
    print(f"Average batch latency: {avg_latency:.1f}ms")
    print(f"Target: <{target_latency}ms")
    
    if avg_latency < target_latency:
        print(f"\n✓ PASS: Latency {avg_latency:.1f}ms < {target_latency}ms target")
    else:
        print(f"\n✗ FAIL: Latency {avg_latency:.1f}ms >= {target_latency}ms target")
else:
    print("No metrics available yet. Pipeline may not have processed any batches.")

