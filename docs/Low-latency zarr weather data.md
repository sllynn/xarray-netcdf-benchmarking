# **Architecture Proposal**

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
3. **Processing:** A Python ThreadPoolExecutor (running on the driver) writes GRIB data directly into specific "slots" of a local Zarr archive (“region writes”).  
4. **Egress:** The Azure Storage CLIM tool, **AzCopy**, pushes only changed chunks to the silver layer Volume using ephemeral SAS tokens.  
5. **Security**: A **TokenManager** class proactively handles SAS token generation and refresh logic.  
6. **Consumption:** The Inference Model uses a validation logic pattern to query data, handling NaN (missing) values gracefully.

## **3\. Implementation specification**

### **A. Storage strategy: pre-allocated Zarr store**

* **Concept:** The Zarr store is treated as a static container rather than a growing dataset.  
* **Initialization (00:00, 06:00, 12:00 and 18:00):** A job creates the full directory structure for the forecast cycle (e.g., T+0 to T+360) on the local SSD. All chunks are initialized with NaN (nodata) values.  
* **Metadata:** The .zmetadata file is generated and consolidated **once** at initialization. It is not modified during incremental updates.  
* **Consistency:** The inference wrapper for forecast models always reads a valid schema, observing NaN values transitioning to float values as data arrives.

### **B. Chunking strategy**

To ensure atomic updates and maximise sync performance, the Zarr chunking must align with the input data stream.

| Dimension | Strategy | Rationale |
| :---- | :---- | :---- |
| **Variable** | **1** | Ensures 1 GRIB file corresponds to 1 Zarr Chunk. Prevents overwriting future data during updates.  |
| **Validation Time (Step)** | **1** | Ensures 1 GRIB file corresponds to 1 Zarr Chunk. |
| **Ensemble member** | **50** | Faster sync to the object store. |
| **Longitude** | **Full** | Keeps the spatial field contiguous to optimise read speed for the inference model. |
| **Latitude** | **Full** | Keeps the spatial field contiguous. |

### **C. Ingestion pipeline**

**Configuration:**

* **Cluster:**  
  * Single node  
  * High core count (32+) Azure VM type with attached SSD storage and at least 2GB RAM per thread  
  * Dedicated security mode  
* **Trigger:** processingTime='0 seconds' (i.e. poll immediately, do not buffer)  
* **Source:** CloudFiles stream reader**,** BinaryFile format, `maxFilesPerTrigger` equal to concurrency of the read / stack ThreadPoolExecutor  
* **Sink**: `foreachBatch`	

**The foreachBatch logic:**

1. **Collect paths:** Materialise the batch of new file paths to the driver.  
2. **Parallel process the input file read / stacking:** Use ThreadPoolExecutor to saturate Local SSD I/O. The logic reads the GRIB, identifies the time index, and performs a Zarr region write.  
3. **Atomic sync:** Trigger azcopy once after the full batch is written. This syncs all changed chunks in a single operation.

### **D. Security & permissions**

* **Identity:** Use the identity available in the notebook context (either the notebook user or job initiator depending on how this is executed) via the Databricks SDK WorkspaceClient.  
* **Mechanism:** Append the SAS token to the destination URL string.  
* **Token Generation:** Use databricks.sdk.temporary\_path\_credentials ([docs](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/catalog/temporary_path_credentials.html#databricks.sdk.service.catalog.TemporaryPathCredentialsAPI)) to generate short-lived SAS tokens for azcopy.  
* **Refresh Strategy:** Implement a "Check-Before-Act" pattern.  
  * Store the token expiration timestamp.  
  * Before invoking azcopy, check if now \> (expiration \- 5\_mins).  
  * If true, call the SDK to generate a new token before proceeding.  
* **Grants:** Ensure the identity has EXTERNAL USE LOCATION on the Unity Catalog Volume/Location.

## **4\. Alternative architectural options considered but not put forwards**

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

* **Concept:** Use the built-in dbutils.fs.cp(source, dest, recurse=True) command to copy the local Zarr archive to the cloud object store. This method abstracts away authentication by using the cluster's attached credentials automatically.  
* **Verdict:** **Rejected.**  
* **Reason:**  
  1. **No differential sync:** dbutils.fs.cp does not support "sync" logic (checking modification times or checksums). It performs a blind "copy all" operation. If the Zarr store contains 10,000 chunks and only 1 has changed, this method will needlessly re-upload the other 9,999 unchanged files.  
  2. **Serial execution:** The utility is effectively single-threaded and processes files sequentially. For Zarr stores composed of thousands of small files, the lack of parallelism results in extremely poor throughput compared to azcopy (which maximizes concurrency) or even a threaded Python loop.  
  3. **Latency penalty:** The combination of copying unchanged data and serial execution would increase the sync duration from seconds (with azcopy) to potentially minutes, violating the end-to-end latency requirement.

## **5\. Next steps**

Plan for building a working prototype:

1. **Develop a proof-of-concept:** for the GRIB read \-\> Zarr region write logic.  
2. **Environment Setup:** Deploy a single node Databricks cluster (Standard\_D32d\_v5 or similar instance).  
3. **Schema Definition:** Define the exact (Variable, Step, Member, Lon, Lat) dimensions and variables list.  
4. **Initialization logic:** Write the job that creates the NaN\-filled Zarr structure at the start of the cycle.  
5. **Pipeline code:** Implement the foreachBatch logic using the ThreadPoolExecutor pattern developed in the proof-of-concept.  
6. **Validation:** Run the validation read logic in a separate notebook to verify that NaNs are correctly replaced by data as the pipeline runs and that the read and transform test can be completed in a reasonable time.

