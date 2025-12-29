# Databricks notebook source
# MAGIC %md
# MAGIC # Consumer / Validator: Silver Zarr Visibility Latency
# MAGIC
# MAGIC Polls the cloud-synced Zarr store in the silver Volume and records when
# MAGIC each expected (variable, forecast_hour) becomes non-NaN.
# MAGIC
# MAGIC This notebook assumes the producer wrote `.json` sidecars next to GRIBs
# MAGIC in the landing zone using the correlation strategy from
# MAGIC [`src.benchmarks.streaming_harness`](src/benchmarks/streaming_harness.py:1).

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

import json
from pathlib import Path

from src.benchmarks.streaming_harness import wait_for_visibility, save_latency_jsonl, EmittedFile

CATALOG = "stuart"
SCHEMA = "lseg"
VOLUME_NAME = "netcdf"

LANDING_ZONE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/landing/"
SILVER_ZARR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/silver/forecast.zarr"

RESULTS_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}/benchmark_results/streaming_latency"

POLL_INTERVAL_MS = 200.0
TIMEOUT_S = 900.0

print(f"Landing zone: {LANDING_ZONE}")
print(f"Silver Zarr:   {SILVER_ZARR}")
print(f"Results dir:   {RESULTS_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load producer manifests

# COMMAND ----------

manifest_paths = sorted(Path(LANDING_ZONE).glob("*.grib2.json"))
print(f"Found {len(manifest_paths)} manifests")

emitted = []
for mp in manifest_paths:
    payload = json.loads(mp.read_text())
    emitted.append(
        EmittedFile(
            file_id=payload["file_id"],
            variable=payload["variable"],
            forecast_hour=int(payload["forecast_hour"]),
            landing_path=payload["landing_path"],
            manifest_path=str(mp),
            producer_write_start_utc=payload["producer_write_start_utc"],
            producer_write_end_utc=payload["producer_write_end_utc"],
        )
    )

print(f"Loaded {len(emitted)} emitted records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for visibility + save results

# COMMAND ----------

events = wait_for_visibility(
    silver_zarr_path=SILVER_ZARR,
    emitted=emitted,
    poll_interval_ms=POLL_INTERVAL_MS,
    timeout_s=TIMEOUT_S,
)

print(f"Visible: {len(events)}/{len(emitted)}")

# Save JSONL
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
output_path = str(Path(RESULTS_DIR) / "latency_events.jsonl")
save_latency_jsonl(output_path, events)
print(f"✓ Saved: {output_path}")

# Basic summary
if events:
    latencies = sorted(e.latency_ms for e in events)

    def pct(p: float) -> float:
        idx = int(round((p / 100.0) * (len(latencies) - 1)))
        return latencies[max(0, min(idx, len(latencies) - 1))]

    print("Latency ms:")
    print(f"  count: {len(latencies)}")
    print(f"  p50:   {pct(50):.0f}")
    print(f"  p90:   {pct(90):.0f}")
    print(f"  p99:   {pct(99):.0f}")
    print(f"  max:   {max(latencies):.0f}")

    # Very lightweight backlog proxy: sort by consumer visibility time and show spread.
    # (If a burst arrives, this tends to widen as the pipeline drains backlog.)
    consumer_times = sorted(e.consumer_visible_utc for e in events)
    print("Backlog proxy:")
    print(f"  first_visible_utc: {consumer_times[0]}")
    print(f"  last_visible_utc:  {consumer_times[-1]}")
