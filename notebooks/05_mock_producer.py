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

# Arrival config
MODE = "steady"  # 'steady' or 'burst'
STEADY_INTERVAL_S = 1.0

# What to emit
VARIABLES = ["t2m", "u10", "v10", "sp"]
FORECAST_HOURS = [0, 1, 2, 3, 4, 5, 6, 9, 12]  # keep small for quick smoke tests

# Burst config
BURST_COUNT = 200

print(f"Landing zone: {LANDING_ZONE}")
print(f"Mode: {MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Emit files

# COMMAND ----------

import itertools
from pathlib import Path

from src.benchmarks.streaming_harness import emit_schedule

Path(LANDING_ZONE).mkdir(parents=True, exist_ok=True)

if MODE == "burst":
    # Repeat a simple plan to reach BURST_COUNT
    base_plan = list(itertools.product(VARIABLES, FORECAST_HOURS))
    plan = (base_plan * ((BURST_COUNT // len(base_plan)) + 1))[:BURST_COUNT]
else:
    plan = list(itertools.product(VARIABLES, FORECAST_HOURS))

print(f"Emitting {len(plan)} files...")

emitted = emit_schedule(
    landing_dir=LANDING_ZONE,
    plan=plan,
    mode=MODE,
    steady_interval_s=STEADY_INTERVAL_S,
    workers=4,
)

print(f"✓ Emitted {len(emitted)} files")
print("Sample:")
for e in emitted[:3]:
    print(f"  {e.landing_path}")
