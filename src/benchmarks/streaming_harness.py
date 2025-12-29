#!/usr/bin/env python3
"""Streaming pipeline E2E test harness.

This module provides two primitives used to measure end-to-end latency in the
Databricks streaming pipeline, without relying on object-store mtimes or event
timestamps that may suffer from clock skew.

Design goals
------------
- Deterministic correlation between producer writes and consumer visibility.
- Single time authority: producer timestamps captured in the same environment
  as the write to the landing zone.
- Consumer validates *cloud-synced* Zarr visibility in the silver Volume.

Key idea
--------
Each emitted GRIB has a unique `file_id` encoded in the filename and persisted
in a sidecar JSON manifest. The consumer watches for corresponding non-NaN
values in the target Zarr slot and records the first time it becomes visible.

Files are assumed to follow the pipeline chunking rule:
- one GRIB == one chunk (Variable=1, Step=1, Ensemble=full)
so a single point sample is sufficient to detect visibility.

This integrates with existing step mapping utilities in
[`src.zarr_init.generate_forecast_steps()`](src/zarr_init.py:1).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import xarray as xr

from ..zarr_init import build_hour_to_index_map, generate_forecast_steps

ArrivalMode = Literal["steady", "burst"]


def utc_now_iso() -> str:
    """Return an RFC3339-ish UTC timestamp string with milliseconds."""
    # datetime.isoformat(timespec="milliseconds") available in py>=3.6
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class EmittedFile:
    """Represents a single emitted GRIB and its correlation metadata."""

    file_id: str
    variable: str
    forecast_hour: int
    landing_path: str
    manifest_path: str
    producer_write_start_utc: str
    producer_write_end_utc: str


@dataclass(frozen=True)
class VisibilityEvent:
    """Represents the first time a file's data becomes visible in the silver Zarr."""

    file_id: str
    variable: str
    forecast_hour: int
    step_index: int
    producer_write_end_utc: str
    consumer_visible_utc: str
    latency_ms: float


def build_emitted_filename(variable: str, forecast_hour: int, file_id: str, producer_end_utc: str) -> str:
    """Create a deterministic filename used for correlation."""
    # Keep predictable, parsable structure.
    # Example: t2m_step003_emit-2025-12-23T14:00:33.123Z_id-<uuid>.grib2
    safe_ts = producer_end_utc.replace(":", "").replace("-", "").replace(".", "")
    return f"{variable}_step{forecast_hour:03d}_emit-{safe_ts}_id-{file_id}.grib2"


def write_manifest_json(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_path, manifest_path)  # atomic rename


def emit_one_grib(
    *,
    landing_dir: str,
    variable: str,
    forecast_hour: int,
    reference_time_iso: Optional[str] = None,
) -> EmittedFile:
    """Emit a single GRIB file into the landing zone and write a sidecar manifest.

    Notes
    -----
    This function reuses the GRIB generation logic in
    [`generate_mock_data.generate_mock_file()`](generate_mock_data.py:341)
    but wraps it with correlation ID + timestamps.

    Parameters
    ----------
    landing_dir:
        Target directory (Databricks Volumes path) where files should be written.
    variable:
        One of the supported variables (t2m/u10/v10/sp).
    forecast_hour:
        Forecast lead hour to encode in the GRIB.
    reference_time_iso:
        Optional ISO reference time (UTC). If None, generator will round to cycle.

    Returns
    -------
    EmittedFile
        Metadata for correlation and later latency computation.
    """
    # Import locally so this module can be imported in environments that don't
    # necessarily have eccodes installed.
    from generate_mock_data import generate_mock_file

    landing_path = Path(landing_dir)
    landing_path.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex

    producer_write_start = utc_now_iso()

    # Use a temp name then atomic rename to avoid AutoLoader seeing partial writes.
    tmp_dir = landing_path / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Generate into temp dir with generator's default filename, then rename.
    reference_time = None
    if reference_time_iso is not None:
        reference_time = datetime.fromisoformat(reference_time_iso.replace("Z", "+00:00"))

    # IMPORTANT: reference_time affects how the pipeline maps into the pre-allocated
    # forecast cycle. If you are initializing the Zarr store for a specific cycle,
    # pass the same reference time here (e.g. cycle start).
    generated_path = generate_mock_file(
        output_dir=tmp_dir,
        variable=variable,
        forecast_hour=forecast_hour,
        reference_time=reference_time or datetime.now(timezone.utc),
        output_format="grib",
    )

    producer_write_end = utc_now_iso()

    final_name = build_emitted_filename(variable, forecast_hour, file_id, producer_write_end)
    final_path = landing_path / final_name
    os.replace(generated_path, final_path)

    manifest_path = final_path.with_suffix(final_path.suffix + ".json")
    write_manifest_json(
        manifest_path,
        {
            "file_id": file_id,
            "variable": variable,
            "forecast_hour": forecast_hour,
            "landing_path": str(final_path),
            "producer_write_start_utc": producer_write_start,
            "producer_write_end_utc": producer_write_end,
        },
    )

    return EmittedFile(
        file_id=file_id,
        variable=variable,
        forecast_hour=forecast_hour,
        landing_path=str(final_path),
        manifest_path=str(manifest_path),
        producer_write_start_utc=producer_write_start,
        producer_write_end_utc=producer_write_end,
    )


def emit_schedule(
    *,
    landing_dir: str,
    plan: Iterable[tuple[str, int]],
    mode: ArrivalMode,
    steady_interval_s: float = 1.0,
) -> list[EmittedFile]:
    """Emit a sequence of GRIBs according to a plan.

    Parameters
    ----------
    landing_dir:
        Landing zone directory.
    plan:
        Iterable of (variable, forecast_hour) tuples.
    mode:
        - steady: wait `steady_interval_s` between emits.
        - burst: emit with no delay.
    steady_interval_s:
        Delay for steady mode.

    Returns
    -------
    list[EmittedFile]
        Emission records.
    """
    emitted: list[EmittedFile] = []
    for i, (var, fh) in enumerate(plan):
        rec = emit_one_grib(landing_dir=landing_dir, variable=var, forecast_hour=fh)
        emitted.append(rec)
        if mode == "steady" and i < 10**12:
            time.sleep(steady_interval_s)
    return emitted


def wait_for_visibility(
    *,
    silver_zarr_path: str,
    emitted: list[EmittedFile],
    poll_interval_ms: float = 200.0,
    timeout_s: float = 600.0,
    sample: tuple[int, int, int] = (0, 180, 360),
) -> list[VisibilityEvent]:
    """Wait until each emitted file becomes visible in the *silver* Zarr store.

    This checks a single point per (variable, step) for NaN->value transition.

    Parameters
    ----------
    silver_zarr_path:
        Path to the Zarr store in the silver Volume.
    emitted:
        Emitted file records.
    poll_interval_ms:
        Poll interval.
    timeout_s:
        Max total wait time.
    sample:
        Tuple of (number_idx, latitude_idx, longitude_idx) sample point.

    Returns
    -------
    list[VisibilityEvent]
        One event per emitted file that became visible within the timeout.
    """
    hour_to_index = build_hour_to_index_map(generate_forecast_steps())

    # Pre-compute targets
    targets = {
        e.file_id: (e.variable, e.forecast_hour, hour_to_index[e.forecast_hour], e.producer_write_end_utc)
        for e in emitted
    }

    start = time.perf_counter()
    remaining = set(targets.keys())
    visible: list[VisibilityEvent] = []

    n_idx, lat_idx, lon_idx = sample

    while remaining:
        if (time.perf_counter() - start) > timeout_s:
            break

        # Open each poll to avoid stale caching.
        ds = xr.open_zarr(silver_zarr_path, consolidated=True)
        try:
            now_iso = utc_now_iso()
            now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))

            to_remove = []
            for file_id in list(remaining):
                var, fh, step_idx, producer_end_iso = targets[file_id]

                # Read one point
                val = (
                    ds[var]
                    .isel(step=step_idx, number=n_idx, latitude=lat_idx, longitude=lon_idx)
                    .values
                )
                if isinstance(val, np.ndarray):
                    val = val.item()

                if not (np.isnan(val)):
                    producer_end_dt = datetime.fromisoformat(producer_end_iso.replace("Z", "+00:00"))
                    latency_ms = (now_dt - producer_end_dt).total_seconds() * 1000

                    visible.append(
                        VisibilityEvent(
                            file_id=file_id,
                            variable=var,
                            forecast_hour=fh,
                            step_index=step_idx,
                            producer_write_end_utc=producer_end_iso,
                            consumer_visible_utc=now_iso,
                            latency_ms=latency_ms,
                        )
                    )
                    to_remove.append(file_id)

            for fid in to_remove:
                remaining.remove(fid)
        finally:
            ds.close()

        time.sleep(poll_interval_ms / 1000.0)

    return visible


def save_latency_jsonl(output_path: str, events: list[VisibilityEvent]) -> None:
    """Save visibility events as JSONL for easy downstream analysis."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for e in events:
            f.write(
                json.dumps(
                    {
                        "file_id": e.file_id,
                        "variable": e.variable,
                        "forecast_hour": e.forecast_hour,
                        "step_index": e.step_index,
                        "producer_write_end_utc": e.producer_write_end_utc,
                        "consumer_visible_utc": e.consumer_visible_utc,
                        "latency_ms": e.latency_ms,
                    }
                )
                + "\n"
            )
