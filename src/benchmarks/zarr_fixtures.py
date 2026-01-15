#!/usr/bin/env python3
"""
Utilities for generating forecast-cycle Zarr stores for benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence
import logging
import shutil

import numpy as np

from src.zarr_init import initialize_zarr_store, generate_forecast_steps

logger = logging.getLogger(__name__)


@dataclass
class FixtureConfig:
    base_path: str
    start_time: datetime
    num_days: int = 7
    cycle_hours: int = 6
    variables: Sequence[str] = ("t2m", "u10", "v10", "sp")
    ensemble_members: int = 50
    lat_size: int = 361
    lon_size: int = 720
    fill_value: float = 0.0
    consolidate_metadata: bool = True
    prefix: str = "forecast_"
    overwrite: bool = False


def _parse_start_time(start_time: str | datetime) -> datetime:
    if isinstance(start_time, datetime):
        return start_time
    parsed = datetime.fromisoformat(start_time)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def generate_forecast_cycle_zarrs(
    base_path: str,
    start_time: str | datetime,
    num_days: int = 7,
    cycle_hours: int = 6,
    variables: Sequence[str] = ("t2m", "u10", "v10", "sp"),
    ensemble_members: int = 50,
    lat_size: int = 361,
    lon_size: int = 720,
    fill_value: float = 0.0,
    consolidate_metadata: bool = True,
    prefix: str = "forecast_",
    overwrite: bool = False,
) -> list[str]:
    """Generate multiple forecast-cycle Zarr stores for benchmarking."""
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    start_dt = _parse_start_time(start_time)
    total_hours = num_days * 24
    steps = generate_forecast_steps()

    paths: list[str] = []
    for offset in range(0, total_hours, cycle_hours):
        cycle_time = start_dt + timedelta(hours=offset)
        name = f"{prefix}{cycle_time:%Y%m%d_%H}.zarr"
        output_path = base / name

        if output_path.exists():
            if overwrite:
                shutil.rmtree(output_path)
            else:
                logger.info(f"Skipping existing store: {output_path}")
                paths.append(str(output_path))
                continue

        logger.info(f"Creating store: {output_path}")
        initialize_zarr_store(
            output_path=str(output_path),
            variables=list(variables),
            ensemble_members=ensemble_members,
            lat_size=lat_size,
            lon_size=lon_size,
            reference_time=cycle_time,
            forecast_steps=steps,
            fill_value=float(fill_value),
            dtype=np.float32,
            consolidate_metadata=consolidate_metadata,
        )
        paths.append(str(output_path))

    return paths


def create_weekly_fixture(config: FixtureConfig) -> list[str]:
    """Create a week of forecast-cycle Zarr stores."""
    return generate_forecast_cycle_zarrs(
        base_path=config.base_path,
        start_time=config.start_time,
        num_days=config.num_days,
        cycle_hours=config.cycle_hours,
        variables=config.variables,
        ensemble_members=config.ensemble_members,
        lat_size=config.lat_size,
        lon_size=config.lon_size,
        fill_value=config.fill_value,
        consolidate_metadata=config.consolidate_metadata,
        prefix=config.prefix,
        overwrite=config.overwrite,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate forecast-cycle Zarr stores for benchmarking.")
    parser.add_argument("--base-path", required=True, help="Base path for Zarr stores (e.g., /Volumes/.../zarrs)")
    parser.add_argument("--start-time", required=True, help="Start time (ISO8601, e.g., 2025-12-29T00:00:00)")
    parser.add_argument("--num-days", type=int, default=7)
    parser.add_argument("--cycle-hours", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_forecast_cycle_zarrs(
        base_path=args.base_path,
        start_time=args.start_time,
        num_days=args.num_days,
        cycle_hours=args.cycle_hours,
        overwrite=args.overwrite,
    )
