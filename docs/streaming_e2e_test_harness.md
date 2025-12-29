# Streaming E2E latency test harness (Databricks)

This adds a deterministic way to measure end-to-end latency from **producer GRIB write completion** → **consumer visibility in the cloud-synced silver Zarr**, without relying on blob mtimes (which can be skewed).

## What was added

- Producer/consumer helper module: [`src/benchmarks/streaming_harness.py`](src/benchmarks/streaming_harness.py:1)
- Producer notebook: [`notebooks/05_mock_producer.py`](notebooks/05_mock_producer.py:1)
- Consumer notebook: [`notebooks/06_silver_consumer_latency.py`](notebooks/06_silver_consumer_latency.py:1)

## Correlation + clock-skew avoidance

### Correlation ID
Each emitted GRIB is assigned a unique `file_id`:
- embedded in the GRIB filename
- recorded in a sidecar manifest `*.grib2.json`

Example:
- `t2m_step003_emit-20251223T140033123Z_id-<uuid>.grib2`
- `t2m_step003_emit-...grib2.json`

### Time authority
- Producer records timestamps using Databricks UTC at write time.
- Consumer records timestamps using its own Databricks UTC at the moment it first observes non-NaN.

The key measurement is:

`e2e_ms = consumer_visible_utc - producer_write_end_utc`

This avoids dependence on:
- object-store event timestamps
- volume/FUSE mtimes
- auto-loader discovery times

## Assumptions

- Your pipeline chunking rule holds: **one GRIB == one Zarr chunk** (Variable=1, Step=1).
- Each GRIB fully populates its chunk (so a single sample point can detect visibility).

## How to run (3 jobs)

### 0) Start the streaming pipeline
Run your existing streaming notebook and keep it running:
- [`notebooks/03_streaming_pipeline.py`](notebooks/03_streaming_pipeline.py:1)

Ensure:
- `LANDING_ZONE` points at the same landing directory the producer uses.
- `CLOUD_DESTINATION` points at the same silver directory the consumer reads.

### 1) Emit files (producer)
Run:
- [`notebooks/05_mock_producer.py`](notebooks/05_mock_producer.py:1)

Configure in the notebook:
- `MODE = "steady"` (1 file/sec)
- or `MODE = "burst"` (200 files back-to-back)

This writes GRIBs + manifests into the landing zone.

### 2) Measure visibility latency (consumer)
Run:
- [`notebooks/06_silver_consumer_latency.py`](notebooks/06_silver_consumer_latency.py:1)

This will:
- load manifests from the landing zone
- poll the silver Zarr store until each (variable, step) becomes non-NaN
- write JSONL results to `RESULTS_DIR/latency_events.jsonl`
- print p50/p90/p99/max and a simple backlog proxy (first/last visible time)

## Output format

The JSONL file contains one record per visible file_id:

```json
{"file_id":"...","variable":"t2m","forecast_hour":3,"step_index":3,"producer_write_end_utc":"...Z","consumer_visible_utc":"...Z","latency_ms":12345.6}
```

## Notes / gotchas

- If the Zarr store is initialized for a particular forecast cycle (00/06/12/18Z), the producer must use the same reference time as the pipeline expects. The helper currently documents this in [`emit_one_grib()`](src/benchmarks/streaming_harness.py:101).
- The producer writes into a `_tmp/` subdir then atomically renames into the landing zone, so AutoLoader doesn’t see partial files.
- The consumer re-opens Zarr on each poll to reduce caching/staleness effects.
