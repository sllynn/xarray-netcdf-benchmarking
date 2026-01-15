# **Architecture Proposal & Implementation**

## *Low-Latency Weather Data Pipeline*

**Target environment**: Azure Databricks (Single node cluster, external Volumes)

**Objective**: Ingest, process, and publish incremental GRIB weather data to a cloud-native Zarr archive with minimal end-to-end latency.

## **1\. Summary**

To achieve minimal latency for power production forecasting, we propose a **"**state-based, region-write**"** architecture driven by Spark Structured Streaming. This design decouples the ingestion of raw GRIB files from the processing logic, allowing for high-throughput bursts.

By utilising a pre-allocated Zarr store and the azcopy CLI tool for incremental syncing of outputs, the system eliminates the "read-modify-write" I/O penalty that might otherwise be incurred by incrementally concatenating GRIBs into a freshly created Zarr archive. This ensures that the inference model has access to new data immediately after processing, without requiring full file rewrites.

## **2\. Architecture**

The pipeline operates on a single node Databricks cluster. The cluster driver acts as the central coordinator, utilising its local SSD for high-IOPS temporary storage before syncing outputs to the cloud object storage service (an external Volume in the silver layer).

### **Core Components**

1. **Ingestion:** Databricks **AutoLoader** (cloudFiles) in file notification mode ([docs](https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/file-notification-mode)) monitors the landing zone for new GRIB files by listening for file events.  
2. **Orchestration:** A Spark Structured Streaming stream with a foreachBatch sink handles micro-batches of incoming files.  
3. **Processing:** A Python ThreadPoolExecutor (running on the driver) writes GRIB data directly into specific "slots" of a local Zarr archive ("region writes").  
4. **Egress:** The Azure Storage CLI tool, **AzCopy**, pushes only changed chunks to the silver layer Volume using ephemeral SAS tokens.  
5. **Security**: A **TokenManager** class proactively handles SAS token generation and refresh logic.  
6. **Consumption:** The Inference Model uses a validation logic pattern to query data, handling NaN (missing) values gracefully.

## **3\. Implementation Specification**

### **A. Storage Strategy: Pre-allocated Zarr Store**

* **Concept:** The Zarr store is treated as a static container rather than a growing dataset.  
* **Initialization (00:00, 06:00, 12:00 and 18:00):** A job creates the full directory structure for the forecast cycle (e.g., T+0 to T+360) on the local SSD. All chunks are initialized with NaN (nodata) values.  
* **Metadata:** The .zmetadata file is generated and consolidated **once** at initialization. It is not modified during incremental updates.  
* **Consistency:** The inference wrapper for forecast models always reads a valid schema, observing NaN values transitioning to float values as data arrives.

**Implementation:** `src/zarr_init.py`

The `initialize_zarr_store()` function creates the full Zarr structure with:
- NaN-filled data arrays for each variable
- Consolidated metadata (`.zmetadata`) for fast opens
- Configurable dimensions and chunking strategy

```python
store = initialize_zarr_store(
    output_path="/local_disk0/forecast.zarr",
    variables=["t2m", "u10", "v10", "sp"],
    ensemble_members=50,
    lat_size=361,
    lon_size=720,
    reference_time=datetime(2024, 1, 1, 0, 0, 0),
)
```

### **B. Forecast Step Configuration**

A key design decision was handling non-uniform forecast steps matching ECMWF ENS format:

| Range | Resolution | Steps |
|-------|------------|-------|
| Hours 0-90 | Hourly | 91 steps |
| Hours 93-144 | 3-hourly | 18 steps |
| Hours 150-360 | 6-hourly | 36 steps |

**Total: 145 steps per forecast cycle**

The `ForecastStepConfig` dataclass encapsulates this configuration, and `build_hour_to_index_map()` creates a lookup table for O(1) mapping from forecast hour to array index:

```python
forecast_steps = generate_forecast_steps()  # [0, 1, 2, ..., 90, 93, 96, ..., 360]
hour_to_index = build_hour_to_index_map(forecast_steps)
# hour_to_index[93] == 91 (first index after hourly section)
```

### **C. Chunking Strategy**

To ensure atomic updates and maximise sync performance, the Zarr chunking must align with the input data stream.

| Dimension | Strategy | Rationale |
| :---- | :---- | :---- |
| **Variable** | **1** | Ensures 1 GRIB file corresponds to 1 Zarr Chunk. Prevents overwriting future data during updates.  |
| **Validation Time (Step)** | **1** | Ensures 1 GRIB file corresponds to 1 Zarr Chunk. |
| **Ensemble member** | **50** | Faster sync to the object store. |
| **Longitude** | **Full** | Keeps the spatial field contiguous to optimise read speed for the inference model. |
| **Latitude** | **Full** | Keeps the spatial field contiguous. |

### **D. GRIB Reading and Variable Mapping**

**Implementation:** `src/region_writer.py`

The region writer handles GRIB parsing using eccodes (primary) or cfgrib (fallback):

1. **Variable Name Resolution:** GRIB files use different naming conventions (e.g., `2t` for 2-metre temperature) than CF conventions (`t2m`). The `get_cf_varname_from_grib()` function first tries the `cfVarName` attribute from eccodes, then falls back to a hardcoded mapping.

2. **Multi-message GRIB Files:** Each GRIB file may contain multiple messages (e.g., all 50 ensemble members). The reader stacks all messages along the ensemble dimension.

3. **Metadata Extraction:** The `GribMetadata` dataclass captures variable name, forecast hour, reference time, valid time, ensemble info, and spatial shape.

### **E. Region Write Strategy**

**Implementation:** `src/region_writer.py`

Two write strategies are available:

**1. xarray Region Writes (`write_grib_to_zarr_region`)**
- Uses xarray's `to_zarr(region=...)` for atomic slice updates
- Higher overhead due to Dataset creation and metadata parsing per write

**2. Direct Zarr Writes (`write_grib_to_zarr_direct`)** *(Recommended)*
- Bypasses xarray entirely
- Opens zarr arrays once per batch with `open_zarr_arrays()`
- Direct numpy array assignment: `zarr_array[step_index, :, :, :] = data`
- No file locking needed for non-overlapping regions
- ~10× faster than xarray region writes

```python
# Open arrays once per batch
zarr_arrays = open_zarr_arrays(zarr_store_path)

# Each worker writes directly (no contention for non-overlapping regions)
result = write_grib_to_zarr_direct(local_grib_path, zarr_arrays, hour_to_index)
```

### **F. File Staging Strategies**

Reading GRIB files directly from FUSE-mounted Volumes is slow due to metadata overhead. Multiple staging strategies were implemented:

| Method | Startup Overhead | Transfer Speed | Best For |
|--------|-----------------|----------------|----------|
| **FUSE copy** (`shutil.copy2`) | None | Slow | Small batches, testing |
| **azcopy** | ~15s | Very fast | Large batches (>10 files) |
| **Azure SDK (async)** | None | Fast | Small-medium batches |

**Implementation Details:**

1. **`stage_files_with_azcopy()`**: Uses `azcopy copy` with `--include-path` to download multiple files in a single command.

2. **`stage_files_with_azure_sdk()`**: Uses `azure.storage.blob.aio.ContainerClient` with asyncio for concurrent downloads without azcopy startup overhead.

3. **`download_single_file_azure_sdk()`**: For "stream" processing mode where each worker downloads its own file.

### **G. Processing Modes**

**Implementation:** `src/streaming_pipeline.py`

The `PipelineConfig` dataclass supports two processing modes:

**1. Batch Stage Mode (`processing_mode='batch_stage'`)**
- Phase 1: Download all files to local SSD (parallel using chosen staging method)
- Phase 2: Process all files in parallel (ThreadPoolExecutor)
- Best for: Large batches where download time can be amortized

**2. Stream Mode (`processing_mode='stream'`)**
- Each worker downloads AND processes its own file
- Download and processing overlap across workers
- Best for: Per-file latency optimization, smaller batches

```python
config = PipelineConfig(
    landing_zone=LANDING_ZONE,
    zarr_store_path=LOCAL_ZARR_PATH,
    cloud_destination=CLOUD_DESTINATION,
    checkpoint_path=CHECKPOINT_PATH,
    staging_method='azure_sdk',      # or 'azcopy'
    processing_mode='stream',        # or 'batch_stage'
    max_files_per_batch=4,
    num_workers=32,
)
```

### **H. Ingestion Pipeline**

**Configuration:**

* **Cluster:**  
  * Single node  
  * High core count (32+) Azure VM type with attached SSD storage and at least 2GB RAM per thread  
  * Dedicated security mode  
* **Trigger:** processingTime='0 seconds' (i.e. poll immediately, do not buffer)  
* **Source:** CloudFiles stream reader, BinaryFile format, `maxFilesPerTrigger` equal to concurrency of the read / stack ThreadPoolExecutor  
* **Sink**: `foreachBatch`

**File Discovery:**

In **production**, AutoLoader watches directly for GRIB files (`*.grib2`) in the landing zone. This is the simplest and most robust approach.

For **benchmarking only**, we use an optional manifest-based pattern (`.grib2.json` files) to enable precise end-to-end latency measurement:

1. AutoLoader watches for `*.grib2.json` manifest files instead of GRIBs
2. Each manifest contains the path to the actual GRIB file and producer timestamps
3. The producer controls exactly when the manifest is written (the "release" moment)
4. This enables accurate latency calculation: `consumer_visible_utc - producer_release_utc`

> **Note:** The manifest pattern adds complexity and is **not recommended for production**. It exists solely to provide a precise time authority for benchmarking without clock skew issues.

**The foreachBatch logic:**

1. **Collect paths:** Materialise the batch of manifest file paths to the driver, read each manifest to get GRIB paths.
2. **Parallel process:** Use ThreadPoolExecutor with direct zarr writes to saturate Local SSD I/O.
3. **Atomic sync:** Trigger azcopy sync once after the full batch is written.

### **I. Cloud Sync**

**Implementation:** `src/cloud_sync.py`

The `CloudSyncer` class provides a high-level interface for syncing Zarr stores:

```python
syncer = CloudSyncer.from_volume_path('/Volumes/catalog/schema/volume/forecast.zarr')
result = syncer.sync('/local_disk0/forecast.zarr')
```

**azcopy auto-installation:** The `ensure_azcopy()` function finds azcopy on the system or automatically downloads and installs it to `/tmp/azcopy/`.

### **J. Security & Token Management**

**Implementation:** `src/cloud_sync.py` - `TokenManager` class

* **Identity:** Uses the identity available in the notebook context via the Databricks SDK `WorkspaceClient`.  
* **Mechanism:** Append the SAS token to the destination URL string.  
* **Token Generation:** Uses `databricks.sdk.temporary_path_credentials.generate_temporary_path_credentials()` to generate short-lived SAS tokens.  
* **Refresh Strategy:** Thread-safe "Check-Before-Act" pattern with double-checked locking:
  * Store the token expiration timestamp
  * Before invoking azcopy, check if `now > (expiration - 5_mins)`
  * If true, acquire lock, double-check, and refresh token
* **Grants:** Ensure the identity has `EXTERNAL USE LOCATION` on the Unity Catalog Volume/Location.

```python
token_manager = TokenManager.from_volume_path('/Volumes/catalog/schema/volume/')
sas_url = token_manager.get_sas_url()  # Thread-safe, auto-refreshes if needed
```

## **4\. Code Structure**

```
src/
├── __init__.py
├── zarr_init.py          # Zarr store initialization
│   ├── ForecastStepConfig        # Non-uniform step configuration
│   ├── generate_forecast_steps() # Create step list (145 steps)
│   ├── build_hour_to_index_map() # Hour → index lookup
│   └── initialize_zarr_store()   # Create NaN-filled Zarr
│
├── region_writer.py      # GRIB reading and Zarr writing
│   ├── GribMetadata              # Extracted GRIB metadata
│   ├── WriteResult               # Write operation result
│   ├── read_grib_data()          # eccodes-based GRIB reading
│   ├── write_grib_to_zarr_region()  # xarray region writes
│   ├── write_grib_to_zarr_direct()  # Direct zarr writes (fast)
│   ├── open_zarr_arrays()        # Open arrays for batch processing
│   ├── stage_files_with_azcopy() # Bulk staging via azcopy
│   ├── stage_files_with_azure_sdk()  # Async SDK staging
│   └── download_single_file_azure_sdk()  # Per-file download
│
├── cloud_sync.py         # Cloud storage sync with token management
│   ├── TokenManager              # Thread-safe SAS token management
│   ├── CloudSyncer               # High-level sync interface
│   ├── sync_with_azcopy()        # azcopy wrapper with retry
│   └── ensure_azcopy()           # Auto-install azcopy
│
├── streaming_pipeline.py # Spark Structured Streaming pipeline
│   ├── PipelineConfig            # Pipeline configuration
│   ├── BatchResult               # Micro-batch result
│   ├── create_streaming_pipeline()  # Setup AutoLoader + foreachBatch
│   ├── PipelineManager           # Lifecycle management
│   └── run_batch_processing()    # Non-streaming batch mode
│
└── benchmarks/                   # Testing infrastructure (not production code)
    ├── streaming_harness.py  # E2E latency test harness
    │   ├── EmittedFile           # Producer-side file record
    │   ├── VisibilityEvent       # Consumer-side visibility record
    │   ├── emit_schedule()       # Emit files at controlled cadence
    │   ├── prepare_all_gribs_locally()  # Phase 1: generate to local disk
    │   ├── stage_gribs_to_landing()     # Phase 2: bulk copy to landing
    │   ├── release_staged_gribs()       # Phase 3: write manifests (benchmarking)
    │   ├── wait_for_visibility()        # Poll Zarr for non-NaN values
    │   └── follow_manifests_and_measure()  # Continuous latency measurement
    │
    └── (other benchmark modules)
```

## **5\. End-to-End Latency Testing (Benchmarking Infrastructure)**

**Implementation:** `src/benchmarks/streaming_harness.py`

> **Note:** This section describes the **benchmarking infrastructure** used to measure pipeline latency. The manifest-based approach described here is specifically for testing and is **not part of the production pipeline**.

The test harness measures true end-to-end latency without clock skew issues:

### **Design Principles**

1. **Single Time Authority:** Producer timestamps captured in the same environment as the write.
2. **Deterministic Correlation:** Each GRIB has a unique `file_id` encoded in filename and manifest.
3. **Cloud-Synced Visibility:** Consumer validates visibility in the *silver* Zarr (not local).

### **Three-Phase Producer Pattern (Benchmarking Only)**

To enable precise latency measurement in tests, GRIB emission is split into phases:

```
Phase 1: Generate GRIBs to local SSD (slow, CPU-intensive)
         ↓
Phase 2: Bulk copy to landing zone via azcopy (fast)
         ↓
Phase 3: Write manifests at controlled intervals (timing-critical)
         AutoLoader detects manifests → triggers processing
```

**Phase 3 - Manifest Release:**

Manifests are small JSON files that:
- Point to the already-staged GRIB file
- Capture the exact `producer_release_utc` timestamp
- Trigger AutoLoader file notifications

This separates the slow GRIB generation from the timing-critical release moment.

### **Consumer Pattern**

```python
events = follow_manifests_and_measure(
    landing_dir='/Volumes/.../landing/',
    silver_zarr_path='/Volumes/.../silver/forecast.zarr',
    poll_interval_ms=200,
    max_runtime_s=300,
)
```

The consumer:
1. Watches for new manifest files in the landing zone
2. Adds each file to a wait-set
3. Polls the silver Zarr store for NaN→value transitions
4. Records `VisibilityEvent` with latency calculated as: `consumer_visible_utc - producer_release_utc`

## **6\. Alternative Architectural Options Considered**

The following patterns were evaluated and rejected for this specific low-latency requirement.

### **Option 1: rsync over FUSE Mount**

* **Concept:** Use standard rsync to copy from /local\_disk0 to /Volumes/\<catalog\>/\<schema\>/silver.  
* **Verdict:** **Rejected.**  
* **Reason:** FUSE overhead is significant for metadata operations. Syncing thousands of small Zarr chunks would likely result in significantly lower throughput compared to azcopy.

### **Option 2: Sequential Processing**

* **Concept:** Loop through files one by one: Read GRIB → Write Zarr → Sync Cloud.  
* **Verdict:** **Rejected.**  
* **Reason:** Sync latency blocks processing. If a burst of files arrives, the pipeline accumulates a backlog. The decoupled "Batch Write → Batch Sync" model processes bursts more efficiently.

### **Option 3: Spark "Real-Time" Mode (UDFs)**

* **Concept:** Use Python UDFs inside the stream to process files row-by-row without foreachBatch.  
* **Verdict:** **Rejected.**  
* **Reason:** UDFs isolate logic per row. This forces separate azcopy calls for each file, adding substantial overhead compared to the single batch sync available in foreachBatch.

### **Option 4: Custom PySpark DataSource**

* **Concept:** Write a formal Spark Connector to handle the GRIB-to-Zarr logic.  
* **Verdict:** **Rejected.**  
* **Reason:** Increases code complexity (requires implementing State management and Commit protocols) and loses the native benefits of Auto Loader for file discovery.

### **Option 5: Batch Processing via xarray.open\_mfdataset**

* **Concept:** Use Xarray's Dask-backed open\_mfdataset to read the entire micro-batch of GRIB files as a single logical Dataset and write it to Zarr in one operation using the region parameter.  
* **Verdict:** **Rejected.**  
* **Reason:**  
  1. **Geometry Constraints:** The region parameter in to\_zarr requires a contiguous slice (e.g., indices 5, 6, 7). If a batch contains non-sequential files (e.g., T+5 and T+8), the write operation fails. Handling this requires complex logic to split the batch back into contiguous groups.  
  2. **Failure Isolation:** open\_mfdataset lazily scans all files in the batch. If a single file is corrupt, the entire operation usually fails. Independent file processing (via ThreadPool) ensures that valid files are processed even if one file in the batch is bad.  
  3. **Dask Overhead:** For simple "Read 1 → Write 1" I/O operations with no computation/aggregation, the Dask graph scheduler introduces unnecessary latency compared to a lightweight Python ThreadPoolExecutor.

### **Option 6: Databricks native utilities (dbutils.fs.cp)**

* **Concept:** Use the built-in dbutils.fs.cp(source, dest, recurse=True) command to copy the local Zarr archive to the cloud object store.  
* **Verdict:** **Rejected.**  
* **Reason:**  
  1. **No differential sync:** dbutils.fs.cp does not support "sync" logic (checking modification times or checksums). It performs a blind "copy all" operation.  
  2. **Serial execution:** The utility is effectively single-threaded and processes files sequentially.  
  3. **Latency penalty:** The combination of copying unchanged data and serial execution would increase the sync duration from seconds (with azcopy) to potentially minutes.

### **Option 7: xarray Region Writes (per-file)**

* **Concept:** Use `xarray.Dataset.to_zarr(region=...)` for each GRIB file.
* **Verdict:** **Initially used, then optimized.**
* **Reason:** The overhead of creating an xarray Dataset for each file (~100-200ms) dominated processing time. Direct zarr array writes reduced per-file overhead to ~10-20ms.

## **7\. Key Design Decisions**

### **Direct Zarr Writes vs xarray**

The single largest performance improvement came from bypassing xarray for writes:

| Method | Per-file Overhead | Notes |
|--------|------------------|-------|
| xarray `to_zarr(region=...)` | ~100-200ms | Dataset creation, metadata parsing |
| Direct zarr assignment | ~10-20ms | Just numpy array write |

### **Azure SDK vs azcopy for Staging**

For downloading files from cloud storage to local SSD:

- **azcopy**: ~15s startup overhead to plan transfers, but extremely fast bulk transfers. Best for large batches.
- **Azure SDK**: No startup overhead, uses async HTTP connections. Better for smaller batches or "stream" mode.

### **Manifest-Based File Discovery (Benchmarking Only)**

> **Important:** This pattern is used **only for E2E latency benchmarking**, not production.

In production, AutoLoader watches for GRIB files directly. For benchmarking, we introduced manifest files to enable:
- Precise latency measurement (manifest write time = release time)
- Single time authority (avoids clock skew between producer and consumer)
- Controlled release timing (GRIB staged first, manifest triggers processing at exact moment)

The added complexity is justified only when accurate latency measurement is required.

### **Pre-allocated NaN Store**

Instead of appending to a growing Zarr:
- Consumer always sees valid schema (never fails to open)
- NaN values gracefully handled by inference models
- No metadata updates during writes (consolidated once at init)
- Enables region writes without coordination

## **8\. Usage Examples**

### **Initialize Store (Start of Forecast Cycle)**

```python
from src.zarr_init import initialize_zarr_store
from datetime import datetime, timezone

# Round to nearest 6-hour cycle
now = datetime.now(timezone.utc)
cycle_hour = (now.hour // 6) * 6
reference_time = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

store = initialize_zarr_store(
    output_path="/local_disk0/forecast.zarr",
    variables=["t2m", "u10", "v10", "sp"],
    ensemble_members=50,
    lat_size=361,
    lon_size=720,
    reference_time=reference_time,
)
```

### **Start Streaming Pipeline**

```python
from src.streaming_pipeline import PipelineConfig, PipelineManager

config = PipelineConfig(
    landing_zone="/Volumes/catalog/schema/volume/landing/",
    zarr_store_path="/local_disk0/forecast.zarr",
    cloud_destination="/Volumes/catalog/schema/volume/silver/forecast.zarr",
    checkpoint_path="/Volumes/catalog/schema/volume/checkpoints/pipeline",
    max_files_per_batch=4,
    num_workers=32,
    staging_method='azure_sdk',
    processing_mode='stream',
)

manager = PipelineManager(spark, config)
manager.start()
```

### **Run E2E Latency Test (Benchmarking Only)**

This example uses the manifest-based test harness for precise latency measurement. This is **not** the production file discovery pattern.

```python
from src.benchmarks.streaming_harness import (
    prepare_all_gribs_locally,
    stage_gribs_to_landing,
    release_staged_gribs,
    follow_manifests_and_measure,
)

# Producer: prepare test files
plan = [("t2m", h) for h in [0, 1, 2, 3, 4, 5]]
prepared = prepare_all_gribs_locally(
    local_staging_dir="/local_disk0/test_gribs",
    plan=plan,
)
staged = stage_gribs_to_landing(
    local_staging_dir="/local_disk0/test_gribs",
    landing_dir="/Volumes/.../landing/",
    prepared=prepared,
)

# Consumer: measure latency (start before producer releases)
events = follow_manifests_and_measure(
    landing_dir="/Volumes/.../landing/",
    silver_zarr_path="/Volumes/.../silver/forecast.zarr",
    poll_interval_ms=200,
    max_runtime_s=120,
)

# Producer: release files at controlled cadence
emitted, timings = release_staged_gribs(
    staged_gribs=staged,
    mode="steady",
    steady_interval_s=1.0,
)
```
