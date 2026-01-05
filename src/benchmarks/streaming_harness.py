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
from concurrent.futures import Future, ThreadPoolExecutor
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


@dataclass(frozen=True, kw_only=True)
class EmittedFile:
    """Represents a single emitted GRIB and its correlation metadata."""

    file_id: str
    variable: str
    forecast_hour: int
    landing_path: str
    manifest_path: str
    producer_write_start_utc: str
    producer_release_utc: str


@dataclass(frozen=True, kw_only=True)
class VisibilityEvent:
    """Represents the first time a file's data becomes visible in the silver Zarr.

    Poll counters
    -------------
    `total_poll_count` counts how many times the consumer checked the target Zarr
    slot for this file_id.

    `na_poll_count` counts how many of those checks still observed NaN / missing
    data (i.e. the slot was not yet visible).

    These can help distinguish cases where latency is dominated by repeated
    "still NA" reads vs. cases where visibility arrives quickly.
    """

    file_id: str
    variable: str
    forecast_hour: int
    step_index: int
    producer_release_utc: str
    consumer_visible_utc: str
    latency_ms: float
    na_poll_count: int
    total_poll_count: int


def build_emitted_filename(variable: str, forecast_hour: int, file_id: str, producer_release_utc: str) -> str:
    """Create a deterministic filename used for correlation."""
    # Keep predictable, parsable structure.
    # Example: t2m_step003_emit-2025-12-23T14:00:33.123Z_id-<uuid>.grib2
    safe_ts = producer_release_utc.replace(":", "").replace("-", "").replace(".", "")
    return f"{variable}_step{forecast_hour:03d}_emit-{safe_ts}_id-{file_id}.grib2"


def write_manifest_json(manifest_path: Path, payload: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_path, manifest_path)  # atomic rename


@dataclass
class _PreparedGrib:
    """Internal: a GRIB that has been generated but not yet released."""
    file_id: str
    variable: str
    forecast_hour: int
    staged_payload: Path
    landing_path: Path
    producer_write_start_utc: str


@dataclass
class _StagedGrib:
    """A GRIB that's been synced to landing_zone/_tmp, ready for fast release."""
    file_id: str
    variable: str
    forecast_hour: int
    staged_path: Path      # Path in landing_zone/_tmp/
    landing_dir: Path      # Parent landing directory
    producer_write_start_utc: str


def _prepare_grib_locally(
    *,
    local_staging_dir: str,
    variable: str,
    forecast_hour: int,
    reference_time_iso: Optional[str] = None,
) -> tuple[str, str, int, Path, str]:
    """Generate a GRIB to local disk (fast). Returns metadata for later staging.
    
    Returns
    -------
    tuple of (file_id, variable, forecast_hour, local_path, producer_write_start_utc)
    """
    from generate_mock_data import generate_mock_file

    local_dir = Path(local_staging_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    producer_write_start = utc_now_iso()

    reference_time = None
    if reference_time_iso is not None:
        reference_time = datetime.fromisoformat(reference_time_iso.replace("Z", "+00:00"))

    generated_path = generate_mock_file(
        output_dir=local_dir,
        variable=variable,
        forecast_hour=forecast_hour,
        reference_time=reference_time or datetime.now(timezone.utc),
        output_format="grib",
    )

    # Rename to include file_id for tracking
    final_local_path = local_dir / f"{file_id}.grib2"
    os.replace(generated_path, final_local_path)

    return (file_id, variable, forecast_hour, final_local_path, producer_write_start)


def prepare_all_gribs_locally(
    *,
    local_staging_dir: str,
    plan: Iterable[tuple[str, int]],
    workers: int = 4,
    reference_time_iso: Optional[str] = None,
) -> list[tuple[str, str, int, Path, str]]:
    """Phase 1: Generate all GRIBs to local disk in parallel.
    
    This is fast because it writes to local SSD, not a Volume.
    
    Parameters
    ----------
    local_staging_dir:
        Local directory for staging (e.g., '/local_disk0/grib_staging').
    plan:
        Iterable of (variable, forecast_hour) tuples.
    workers:
        Number of parallel workers for GRIB generation.
    reference_time_iso:
        Optional reference time for GRIB metadata.
    
    Returns
    -------
    list of (file_id, variable, forecast_hour, local_path, producer_write_start_utc) tuples
    """
    plan_list = list(plan)
    
    def _gen(args: tuple[str, int]) -> tuple[str, str, int, Path, str]:
        var, fh = args
        return _prepare_grib_locally(
            local_staging_dir=local_staging_dir,
            variable=var,
            forecast_hour=fh,
            reference_time_iso=reference_time_iso,
        )
    
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_gen, args) for args in plan_list]
        for fut in futures:
            results.append(fut.result())
    
    return results


def stage_gribs_to_landing(
    *,
    local_staging_dir: str,
    landing_dir: str,
    prepared: list[tuple[str, str, int, Path, str]],
) -> list[_StagedGrib]:
    """Phase 2: Copy all prepared GRIBs from local disk to landing_zone/_tmp.
    
    This can be slow (Volume I/O) but happens once before the timed test.
    
    Parameters
    ----------
    local_staging_dir:
        Local directory containing prepared GRIBs.
    landing_dir:
        Landing zone directory (Volume path).
    prepared:
        Output from prepare_all_gribs_locally().
    
    Returns
    -------
    list of _StagedGrib ready for fast release
    """
    import shutil
    
    landing_path = Path(landing_dir)
    tmp_dir = landing_path / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    staged = []
    for file_id, variable, forecast_hour, local_path, producer_write_start in prepared:
        # Copy to landing_zone/_tmp/
        staged_path = tmp_dir / f"{file_id}.grib2"
        shutil.copy2(local_path, staged_path)
        
        staged.append(_StagedGrib(
            file_id=file_id,
            variable=variable,
            forecast_hour=forecast_hour,
            staged_path=staged_path,
            landing_dir=landing_path,
            producer_write_start_utc=producer_write_start,
        ))
    
    return staged


def release_staged_gribs(
    *,
    staged_gribs: list[_StagedGrib],
    mode: ArrivalMode,
    steady_interval_s: float = 1.0,
) -> list[EmittedFile]:
    """Phase 3: Release pre-staged GRIBs at scheduled intervals (fast renames only).
    
    Since files are already on the Volume, this is just atomic renames.
    
    Parameters
    ----------
    staged_gribs:
        Output from stage_gribs_to_landing().
    mode:
        'steady' for timed releases, 'burst' for immediate.
    steady_interval_s:
        Interval between releases in steady mode.
    
    Returns
    -------
    list of EmittedFile records
    """
    emitted: list[EmittedFile] = []
    
    if mode == "burst":
        for sg in staged_gribs:
            release_utc = utc_now_iso()
            rec = _release_staged_grib(sg, release_utc)
            emitted.append(rec)
        return emitted
    
    # Steady mode: release at fixed cadence
    start = time.perf_counter()
    
    for i, sg in enumerate(staged_gribs):
        # Wait until target release time
        target_t = start + i * steady_interval_s
        sleep_s = target_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        
        # Fast atomic rename
        release_utc = utc_now_iso()
        rec = _release_staged_grib(sg, release_utc)
        emitted.append(rec)
    
    return emitted


def _release_staged_grib(staged: _StagedGrib, producer_release_utc: str) -> EmittedFile:
    """Release a staged GRIB via atomic rename (fast)."""
    # Build final filename with release timestamp
    final_name = build_emitted_filename(
        staged.variable,
        staged.forecast_hour,
        staged.file_id,
        producer_release_utc,
    )
    final_path = staged.landing_dir / final_name
    final_manifest = final_path.with_suffix(final_path.suffix + ".json")
    
    # Write manifest (small JSON file)
    staged_manifest = staged.staged_path.with_suffix(".json")
    write_manifest_json(
        staged_manifest,
        {
            "file_id": staged.file_id,
            "variable": staged.variable,
            "forecast_hour": staged.forecast_hour,
            "landing_path": str(final_path),
            "producer_write_start_utc": staged.producer_write_start_utc,
            "producer_release_utc": producer_release_utc,
        },
    )
    
    # Atomic release: rename staged -> final (fast, same filesystem)
    os.replace(staged.staged_path, final_path)
    os.replace(staged_manifest, final_manifest)
    
    return EmittedFile(
        file_id=staged.file_id,
        variable=staged.variable,
        forecast_hour=staged.forecast_hour,
        landing_path=str(final_path),
        manifest_path=str(final_manifest),
        producer_write_start_utc=staged.producer_write_start_utc,
        producer_release_utc=producer_release_utc,
    )


def _prepare_grib(
    *,
    landing_dir: str,
    variable: str,
    forecast_hour: int,
    reference_time_iso: Optional[str] = None,
) -> _PreparedGrib:
    """Generate a GRIB into staging (slow). Does NOT release to landing zone yet.
    
    This is the CPU-intensive part that can be parallelized in background workers.
    """
    from generate_mock_data import generate_mock_file

    landing_path = Path(landing_dir)
    landing_path.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    producer_write_start = utc_now_iso()

    tmp_dir = landing_path / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    reference_time = None
    if reference_time_iso is not None:
        reference_time = datetime.fromisoformat(reference_time_iso.replace("Z", "+00:00"))

    generated_path = generate_mock_file(
        output_dir=tmp_dir,
        variable=variable,
        forecast_hour=forecast_hour,
        reference_time=reference_time or datetime.now(timezone.utc),
        output_format="grib",
    )

    # Move to staged location (still in _tmp, but with our naming)
    staged_payload = tmp_dir / f"{file_id}.grib2.staged"
    os.replace(generated_path, staged_payload)

    # Compute final landing path (will be renamed to this at release time)
    # We don't know the release timestamp yet, so use a placeholder in the filename
    final_path = landing_path / f"{variable}_step{forecast_hour:03d}_{file_id}.grib2"

    return _PreparedGrib(
        file_id=file_id,
        variable=variable,
        forecast_hour=forecast_hour,
        staged_payload=staged_payload,
        landing_path=final_path,
        producer_write_start_utc=producer_write_start,
    )


def _release_prepared_grib(
    prepared: _PreparedGrib,
    producer_release_utc: str,
) -> EmittedFile:
    """Release a prepared GRIB to the landing zone (fast atomic rename).
    
    This is the time-critical part that should happen at the exact release moment.
    """
    # Build the final filename with the actual release timestamp
    final_name = build_emitted_filename(
        prepared.variable,
        prepared.forecast_hour,
        prepared.file_id,
        producer_release_utc,
    )
    final_path = prepared.landing_path.parent / final_name
    final_manifest = final_path.with_suffix(final_path.suffix + ".json")

    # Write manifest (small, fast)
    staged_manifest = prepared.staged_payload.with_suffix(".json")
    write_manifest_json(
        staged_manifest,
        {
            "file_id": prepared.file_id,
            "variable": prepared.variable,
            "forecast_hour": prepared.forecast_hour,
            "landing_path": str(final_path),
            "producer_write_start_utc": prepared.producer_write_start_utc,
            "producer_release_utc": producer_release_utc,
        },
    )

    # Atomic release: rename staged -> final
    os.replace(prepared.staged_payload, final_path)
    os.replace(staged_manifest, final_manifest)

    return EmittedFile(
        file_id=prepared.file_id,
        variable=prepared.variable,
        forecast_hour=prepared.forecast_hour,
        landing_path=str(final_path),
        manifest_path=str(final_manifest),
        producer_write_start_utc=prepared.producer_write_start_utc,
        producer_release_utc=producer_release_utc,
    )


def emit_one_grib(
    *,
    landing_dir: str,
    variable: str,
    forecast_hour: int,
    reference_time_iso: Optional[str] = None,
    producer_release_utc: Optional[str] = None,
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
    producer_release_utc:
        Producer-side release timestamp (UTC ISO). Used for correlation and for
        the emitted filename.

        This function will always stage payload+manifest under non-matching
        `.partial` filenames and then atomically rename them into place.

    Returns
    -------
    EmittedFile
        Metadata for correlation and later latency computation.
    """
    prepared = _prepare_grib(
        landing_dir=landing_dir,
        variable=variable,
        forecast_hour=forecast_hour,
        reference_time_iso=reference_time_iso,
    )
    
    if producer_release_utc is None:
        producer_release_utc = utc_now_iso()
    
    return _release_prepared_grib(prepared, producer_release_utc)


def emit_schedule(
    *,
    landing_dir: str,
    plan: Iterable[tuple[str, int]],
    mode: ArrivalMode,
    steady_interval_s: float = 1.0,
    workers: int = 1,
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
    workers:
        Number of background workers used to *prepare* files.

        In steady mode, the harness will still *release* files at the requested
        cadence (1/`steady_interval_s`), but will overlap the expensive GRIB
        generation work.

    Returns
    -------
    list[EmittedFile]
        Emission records.
    """
    # Materialize the plan so we can safely index it and schedule background work.
    plan_list = list(plan)

    def _emit(args: tuple[str, int], *, release_utc: Optional[str] = None) -> EmittedFile:
        var, fh = args
        return emit_one_grib(
            landing_dir=landing_dir,
            variable=var,
            forecast_hour=fh,
            producer_release_utc=release_utc,
        )

    emitted: list[EmittedFile] = []

    # Burst mode: emit as fast as possible. Optional worker pool can still help by
    # overlapping generation across tasks.
    if mode == "burst":
        if workers <= 1:
            for var, fh in plan_list:
                emitted.append(_emit((var, fh)))
            return emitted

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_emit, args) for args in plan_list]
            for fut in futures:
                emitted.append(fut.result())
        return emitted

    # Steady mode: release at a fixed cadence, but overlap GRIB generation in background.
    # Workers pre-generate GRIBs into staging; main thread releases at scheduled times.
    cadence_s = steady_interval_s
    start = time.perf_counter()

    def _prepare(args: tuple[str, int]) -> _PreparedGrib:
        """Background worker: generate GRIB into staging (slow)."""
        var, fh = args
        return _prepare_grib(
            landing_dir=landing_dir,
            variable=var,
            forecast_hour=fh,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        next_to_schedule = 0
        in_flight: dict[int, Future[_PreparedGrib]] = {}

        def _schedule_one(i: int) -> None:
            # Schedule background GRIB generation (CPU-heavy work).
            in_flight[i] = ex.submit(_prepare, plan_list[i])

        # Prime the pipeline: start preparing several GRIBs ahead of time
        while next_to_schedule < len(plan_list) and len(in_flight) < max(1, workers):
            _schedule_one(next_to_schedule)
            next_to_schedule += 1

        for i in range(len(plan_list)):
            # Wait until the target wall-clock release time for item i.
            target_t = start + i * cadence_s
            sleep_s = target_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

            # Ensure the i-th item is scheduled (defensive).
            if i not in in_flight:
                _schedule_one(i)
                next_to_schedule = max(next_to_schedule, i + 1)

            # Wait for GRIB preparation to complete (should already be done if workers > 0)
            prepared = in_flight.pop(i).result()

            # Release time: atomic rename into landing zone (fast)
            release_utc = utc_now_iso()
            rec = _release_prepared_grib(prepared, release_utc)
            emitted.append(rec)

            # Keep the backlog full: schedule more background preparations
            while next_to_schedule < len(plan_list) and len(in_flight) < max(1, workers):
                _schedule_one(next_to_schedule)
                next_to_schedule += 1

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

    Notes
    -----
    This function assumes you already have the full list of `emitted` files.
    If you want to measure latency while the producer is still running
    (recommended to avoid bias), use:
    [`follow_manifests_and_measure()`](src/benchmarks/streaming_harness.py:1)

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
        e.file_id: (e.variable, e.forecast_hour, hour_to_index[e.forecast_hour], e.producer_release_utc)
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
        #
        # Some stores in Volumes/silver may not have consolidated metadata
        # (no .zmetadata). In that case, fall back to non-consolidated opens.
        try:
            ds = xr.open_zarr(silver_zarr_path, consolidated=True)
        except KeyError as e:
            if str(e).strip("\"") in {".zmetadata", "zarr.json"}:
                ds = xr.open_zarr(silver_zarr_path, consolidated=False)
            else:
                raise
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
                    producer_release_dt = datetime.fromisoformat(producer_end_iso.replace("Z", "+00:00"))
                    latency_ms = (now_dt - producer_release_dt).total_seconds() * 1000

                    visible.append(
                        VisibilityEvent(
                            file_id=file_id,
                            variable=var,
                            forecast_hour=fh,
                            step_index=step_idx,
                            producer_release_utc=producer_end_iso,
                            consumer_visible_utc=now_iso,
                            latency_ms=latency_ms,
                            na_poll_count=0,
                            total_poll_count=0,
                        )
                    )
                    to_remove.append(file_id)

            for fid in to_remove:
                remaining.remove(fid)
        finally:
            ds.close()

        time.sleep(poll_interval_ms / 1000.0)

    return visible


def follow_manifests_and_measure(
    *,
    landing_dir: str,
    silver_zarr_path: str,
    poll_interval_ms: float = 200.0,
    sample: tuple[int, int, int] = (0, 180, 360),
    max_runtime_s: Optional[float] = None,
) -> list[VisibilityEvent]:
    """Continuously discover new manifests and record first-visibility latency.

    This is the unbiased mode for a streaming test: start the consumer first,
    then start the producer. The consumer will:
    1) watch `landing_dir` for new `*.grib2.json` manifests
    2) add them to the wait-set
    3) record a `VisibilityEvent` as soon as each corresponding Zarr slot becomes non-NaN

    Termination
    -----------
    - If `max_runtime_s` is provided, it will stop after that wall-clock time.
    - Otherwise it runs until interrupted by the user (e.g. stop cell execution).

    Parameters
    ----------
    landing_dir:
        Directory containing producer manifest files.
    silver_zarr_path:
        Path to the Zarr store in the silver Volume.
    poll_interval_ms:
        Poll interval for both manifest discovery and Zarr reads.
    sample:
        Tuple of (number_idx, latitude_idx, longitude_idx) sample point.
    max_runtime_s:
        Optional maximum runtime.

    Returns
    -------
    list[VisibilityEvent]
        All visibility events captured during the run.
    """
    landing = Path(landing_dir)
    seen_manifests: set[str] = set()
    emitted_by_id: dict[str, EmittedFile] = {}
    events_by_id: dict[str, VisibilityEvent] = {}

    # Poll counters keyed by file_id.
    # - total_poll_count: number of times we've checked this file's target Zarr slot
    # - na_poll_count: number of those checks that were still NaN
    poll_counts: dict[str, dict[str, int]] = {}

    start = time.perf_counter()
    hour_to_index = build_hour_to_index_map(generate_forecast_steps())
    n_idx, lat_idx, lon_idx = sample

    while True:
        if max_runtime_s is not None and (time.perf_counter() - start) > max_runtime_s:
            break

        # 1) Discover new manifests
        for mp in landing.glob("*.grib2.json"):
            mp_str = str(mp)
            if mp_str in seen_manifests:
                continue
            try:
                payload = json.loads(mp.read_text())
                ef = EmittedFile(
                    file_id=payload["file_id"],
                    variable=payload["variable"],
                    forecast_hour=int(payload["forecast_hour"]),
                    landing_path=payload["landing_path"],
                    manifest_path=mp_str,
                    producer_write_start_utc=payload["producer_write_start_utc"],
                    producer_release_utc=payload["producer_release_utc"],
                )
                emitted_by_id[ef.file_id] = ef
                seen_manifests.add(mp_str)
            except Exception:
                # If a manifest is mid-write or transiently unreadable, just retry next poll.
                continue

        # 2) Check visibility for all not-yet-visible file_ids
        pending = [fid for fid in emitted_by_id.keys() if fid not in events_by_id]
        for fid in pending:
            poll_counts.setdefault(fid, {"na_poll_count": 0, "total_poll_count": 0})

        if pending:
            try:
                ds = xr.open_zarr(silver_zarr_path, consolidated=True)
            except KeyError as e:
                if str(e).strip("\"") in {".zmetadata", "zarr.json"}:
                    ds = xr.open_zarr(silver_zarr_path, consolidated=False)
                else:
                    raise

            try:
                now_iso = utc_now_iso()
                now_dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))

                for fid in pending:
                    ef = emitted_by_id[fid]
                    step_idx = hour_to_index.get(ef.forecast_hour)
                    if step_idx is None:
                        continue

                    val = (
                        ds[ef.variable]
                        .isel(step=step_idx, number=n_idx, latitude=lat_idx, longitude=lon_idx)
                        .values
                    )
                    if isinstance(val, np.ndarray):
                        val = val.item()

                    poll_counts[fid]["total_poll_count"] += 1

                    if not np.isnan(val):
                        producer_release_dt = datetime.fromisoformat(
                            ef.producer_release_utc.replace("Z", "+00:00")
                        )
                        latency_ms = (now_dt - producer_release_dt).total_seconds() * 1000
                        events_by_id[fid] = VisibilityEvent(
                            file_id=fid,
                            variable=ef.variable,
                            forecast_hour=ef.forecast_hour,
                            step_index=step_idx,
                            producer_release_utc=ef.producer_release_utc,
                            consumer_visible_utc=now_iso,
                            latency_ms=latency_ms,
                            na_poll_count=poll_counts[fid]["na_poll_count"],
                            total_poll_count=poll_counts[fid]["total_poll_count"],
                        )
                    else:
                        poll_counts[fid]["na_poll_count"] += 1
            finally:
                ds.close()

        time.sleep(poll_interval_ms / 1000.0)

    return list(events_by_id.values())


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
                        "producer_release_utc": e.producer_release_utc,
                        "consumer_visible_utc": e.consumer_visible_utc,
                        "latency_ms": e.latency_ms,
                        "na_poll_count": e.na_poll_count,
                        "total_poll_count": e.total_poll_count,
                    }
                )
                + "\n"
            )
