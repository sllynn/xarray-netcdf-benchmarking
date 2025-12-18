#!/usr/bin/env python3
"""
End-to-End Latency Benchmark.

Measures time from file arrival to consumer-readable data.
Tests the complete pipeline including file discovery, GRIB processing,
Zarr region writes, cloud sync, and consumer reads.
"""

import os
import time
import shutil
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class E2ELatencyResult:
    """Result of end-to-end latency benchmark."""
    test_name: str
    arrival_pattern: str
    num_files: int
    
    # Timing breakdown (all in milliseconds)
    file_arrival_time_ms: float
    discovery_time_ms: float
    processing_time_ms: float
    sync_time_ms: float
    consumer_read_time_ms: float
    total_latency_ms: float
    
    # Throughput metrics
    files_per_second: float
    mb_per_second: float
    
    # Success metrics
    files_successful: int
    files_failed: int
    data_validated: bool
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    errors: list = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for E2E benchmark."""
    zarr_store_path: str
    landing_zone: str
    cloud_destination: Optional[str] = None
    
    # Test parameters
    num_files: int = 10
    file_size_mb: float = 50.0
    arrival_pattern: str = 'burst'  # 'burst', 'steady', 'random'
    arrival_interval_ms: float = 100.0  # For steady pattern
    
    # Timeouts
    discovery_timeout_ms: float = 5000.0
    processing_timeout_ms: float = 30000.0
    sync_timeout_ms: float = 60000.0


def simulate_file_arrival(
    source_files: list[str],
    landing_zone: str,
    pattern: str = 'burst',
    interval_ms: float = 100.0,
) -> list[tuple[str, float]]:
    """Simulate file arrival with different patterns.
    
    Parameters
    ----------
    source_files : list[str]
        List of source GRIB files to copy.
    landing_zone : str
        Destination directory (landing zone).
    pattern : str
        Arrival pattern: 'burst', 'steady', or 'random'.
    interval_ms : float
        Interval between files for 'steady' pattern.
    
    Returns
    -------
    list[tuple[str, float]]
        List of (destination_path, arrival_timestamp_ms) tuples.
    """
    arrivals = []
    landing_path = Path(landing_zone)
    landing_path.mkdir(parents=True, exist_ok=True)
    
    start_time = time.perf_counter() * 1000
    
    for i, source_file in enumerate(source_files):
        source_path = Path(source_file)
        dest_path = landing_path / f"arrival_{i:04d}_{source_path.name}"
        
        if pattern == 'steady':
            # Wait for interval before each file (except first)
            if i > 0:
                time.sleep(interval_ms / 1000.0)
        elif pattern == 'random':
            # Random delay between 0 and 2x interval
            delay = np.random.uniform(0, 2 * interval_ms / 1000.0)
            time.sleep(delay)
        # 'burst' pattern: no delay
        
        # Copy file to landing zone
        shutil.copy2(source_file, dest_path)
        arrival_time = time.perf_counter() * 1000
        arrivals.append((str(dest_path), arrival_time - start_time))
    
    return arrivals


def wait_for_data_available(
    zarr_store_path: str,
    step_indices: list[int],
    variable: str,
    timeout_ms: float = 30000.0,
    poll_interval_ms: float = 100.0,
) -> tuple[bool, float]:
    """Wait for data to become available (non-NaN) in Zarr store.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to Zarr store.
    step_indices : list[int]
        Step indices to check.
    variable : str
        Variable name to check.
    timeout_ms : float
        Maximum wait time in milliseconds.
    poll_interval_ms : float
        Polling interval in milliseconds.
    
    Returns
    -------
    tuple[bool, float]
        (all_available, time_to_available_ms)
    """
    start_time = time.perf_counter() * 1000
    
    while True:
        try:
            ds = xr.open_zarr(zarr_store_path, consolidated=True)
            
            all_available = True
            for step_idx in step_indices:
                # Check if data at this step is non-NaN
                sample = ds[variable].isel(step=step_idx, number=0, latitude=180, longitude=360).values
                if np.isnan(sample):
                    all_available = False
                    break
            
            ds.close()
            
            if all_available:
                elapsed = time.perf_counter() * 1000 - start_time
                return True, elapsed
                
        except Exception as e:
            logger.debug(f"Error checking data availability: {e}")
        
        elapsed = time.perf_counter() * 1000 - start_time
        if elapsed > timeout_ms:
            return False, elapsed
        
        time.sleep(poll_interval_ms / 1000.0)


def measure_consumer_read_latency(
    zarr_store_path: str,
    step_indices: list[int],
    variable: str,
    access_pattern: str = 'single_step',
) -> tuple[float, dict]:
    """Measure consumer read latency for different access patterns.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to Zarr store.
    step_indices : list[int]
        Step indices to read.
    variable : str
        Variable name to read.
    access_pattern : str
        Access pattern: 'single_step', 'time_range', 'spatial_subset'.
    
    Returns
    -------
    tuple[float, dict]
        (read_time_ms, data_stats)
    """
    start_time = time.perf_counter()
    
    ds = xr.open_zarr(zarr_store_path, consolidated=True)
    
    if access_pattern == 'single_step':
        # Read a single time step (common for point-in-time forecasts)
        data = ds[variable].isel(step=step_indices[0]).compute()
    elif access_pattern == 'time_range':
        # Read a range of time steps (common for trajectory analysis)
        data = ds[variable].isel(step=step_indices).compute()
    elif access_pattern == 'spatial_subset':
        # Read a spatial subset across all times
        data = ds[variable].isel(
            step=step_indices,
            latitude=slice(150, 200),
            longitude=slice(300, 400),
        ).compute()
    else:
        raise ValueError(f"Unknown access pattern: {access_pattern}")
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Compute statistics
    stats = {
        'shape': data.shape,
        'nan_percentage': float(np.isnan(data).mean() * 100),
        'min': float(np.nanmin(data)) if not np.all(np.isnan(data)) else None,
        'max': float(np.nanmax(data)) if not np.all(np.isnan(data)) else None,
        'mean': float(np.nanmean(data)) if not np.all(np.isnan(data)) else None,
    }
    
    ds.close()
    
    return elapsed_ms, stats


def benchmark_e2e_latency(
    config: BenchmarkConfig,
    source_grib_files: list[str],
    process_batch_fn: Optional[Callable] = None,
) -> E2ELatencyResult:
    """Run end-to-end latency benchmark.
    
    Simulates the complete data flow:
    1. Files arrive in landing zone
    2. Pipeline discovers and processes files
    3. Data is written to Zarr
    4. (Optional) Data is synced to cloud
    5. Consumer reads data
    
    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.
    source_grib_files : list[str]
        List of source GRIB files to use.
    process_batch_fn : callable, optional
        Custom batch processing function. If None, uses default region writer.
    
    Returns
    -------
    E2ELatencyResult
        Benchmark results including timing breakdown.
    """
    from ..zarr_init import generate_forecast_steps, build_hour_to_index_map
    from ..region_writer import write_grib_batch_parallel
    
    logger.info(f"Starting E2E latency benchmark")
    logger.info(f"  Files: {config.num_files}")
    logger.info(f"  Pattern: {config.arrival_pattern}")
    
    errors = []
    
    # Limit to available files
    files_to_use = source_grib_files[:config.num_files]
    if len(files_to_use) < config.num_files:
        logger.warning(
            f"Only {len(files_to_use)} source files available, "
            f"requested {config.num_files}"
        )
    
    # Build hour-to-index mapping
    forecast_steps = generate_forecast_steps()
    hour_to_index = build_hour_to_index_map(forecast_steps)
    
    # === Phase 1: File Arrival ===
    arrival_start = time.perf_counter() * 1000
    
    arrivals = simulate_file_arrival(
        files_to_use,
        config.landing_zone,
        pattern=config.arrival_pattern,
        interval_ms=config.arrival_interval_ms,
    )
    
    file_arrival_time = time.perf_counter() * 1000 - arrival_start
    logger.info(f"File arrival complete: {len(arrivals)} files in {file_arrival_time:.1f}ms")
    
    # === Phase 2: Discovery ===
    # In a real streaming scenario, AutoLoader handles this
    # Here we simulate discovery time
    discovery_start = time.perf_counter() * 1000
    
    landed_files = [a[0] for a in arrivals]
    # Verify all files exist
    for f in landed_files:
        if not Path(f).exists():
            errors.append(f"File not found: {f}")
    
    discovery_time = time.perf_counter() * 1000 - discovery_start
    logger.info(f"Discovery complete: {discovery_time:.1f}ms")
    
    # === Phase 3: Processing ===
    processing_start = time.perf_counter() * 1000
    
    if process_batch_fn:
        results = process_batch_fn(landed_files, config.zarr_store_path)
    else:
        results = write_grib_batch_parallel(
            landed_files,
            config.zarr_store_path,
            hour_to_index=hour_to_index,
            max_workers=32,
        )
    
    processing_time = time.perf_counter() * 1000 - processing_start
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    for r in results:
        if not r.success:
            errors.append(f"Processing failed: {r.grib_path}: {r.error}")
    
    logger.info(f"Processing complete: {successful}/{len(results)} files in {processing_time:.1f}ms")
    
    # === Phase 4: Sync (optional) ===
    sync_time = 0.0
    if config.cloud_destination:
        from ..cloud_sync import CloudSyncer
        
        sync_start = time.perf_counter() * 1000
        syncer = CloudSyncer(config.cloud_destination)
        sync_result = syncer.sync_zarr_chunks(config.zarr_store_path)
        sync_time = time.perf_counter() * 1000 - sync_start
        
        if not sync_result.success:
            errors.append(f"Sync failed: {sync_result.error}")
        
        logger.info(f"Sync complete: {sync_time:.1f}ms")
    
    # === Phase 5: Consumer Read ===
    read_start = time.perf_counter() * 1000
    
    # Get step indices that were written
    step_indices = [r.step_index for r in results if r.success and r.step_index >= 0]
    variable = results[0].variable if results else 't2m'
    
    if step_indices:
        read_time, data_stats = measure_consumer_read_latency(
            config.zarr_store_path,
            step_indices[:5],  # Limit to first 5 for speed
            variable,
            access_pattern='single_step',
        )
        data_validated = data_stats['nan_percentage'] < 100
    else:
        read_time = 0
        data_validated = False
    
    consumer_read_time = time.perf_counter() * 1000 - read_start
    logger.info(f"Consumer read complete: {consumer_read_time:.1f}ms")
    
    # === Calculate totals ===
    total_latency = file_arrival_time + discovery_time + processing_time + sync_time + consumer_read_time
    
    # Calculate throughput
    total_time_seconds = total_latency / 1000.0
    files_per_second = len(arrivals) / total_time_seconds if total_time_seconds > 0 else 0
    total_mb = len(arrivals) * config.file_size_mb
    mb_per_second = total_mb / total_time_seconds if total_time_seconds > 0 else 0
    
    return E2ELatencyResult(
        test_name='e2e_latency',
        arrival_pattern=config.arrival_pattern,
        num_files=len(arrivals),
        file_arrival_time_ms=file_arrival_time,
        discovery_time_ms=discovery_time,
        processing_time_ms=processing_time,
        sync_time_ms=sync_time,
        consumer_read_time_ms=consumer_read_time,
        total_latency_ms=total_latency,
        files_per_second=files_per_second,
        mb_per_second=mb_per_second,
        files_successful=successful,
        files_failed=failed,
        data_validated=data_validated,
        errors=errors,
    )


def run_benchmark_suite(
    config: BenchmarkConfig,
    source_grib_files: list[str],
    patterns: list[str] = None,
    file_counts: list[int] = None,
) -> list[E2ELatencyResult]:
    """Run a suite of E2E latency benchmarks with different configurations.
    
    Parameters
    ----------
    config : BenchmarkConfig
        Base configuration.
    source_grib_files : list[str]
        List of source GRIB files.
    patterns : list[str], optional
        Arrival patterns to test (default: ['burst', 'steady']).
    file_counts : list[int], optional
        File counts to test (default: [1, 5, 10, 20]).
    
    Returns
    -------
    list[E2ELatencyResult]
        Results for each configuration tested.
    """
    if patterns is None:
        patterns = ['burst', 'steady']
    
    if file_counts is None:
        file_counts = [1, 5, 10, 20]
    
    results = []
    
    for pattern in patterns:
        for count in file_counts:
            if count > len(source_grib_files):
                logger.warning(f"Skipping count={count}, only {len(source_grib_files)} files available")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Running benchmark: pattern={pattern}, files={count}")
            logger.info(f"{'='*60}")
            
            # Update config for this run
            test_config = BenchmarkConfig(
                zarr_store_path=config.zarr_store_path,
                landing_zone=config.landing_zone,
                cloud_destination=config.cloud_destination,
                num_files=count,
                arrival_pattern=pattern,
            )
            
            result = benchmark_e2e_latency(test_config, source_grib_files)
            results.append(result)
            
            logger.info(f"Result: {result.total_latency_ms:.1f}ms total, "
                       f"{result.files_per_second:.1f} files/s")
    
    return results


def save_results(results: list[E2ELatencyResult], output_path: str) -> None:
    """Save benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = [
        {
            'test_name': r.test_name,
            'arrival_pattern': r.arrival_pattern,
            'num_files': r.num_files,
            'file_arrival_time_ms': r.file_arrival_time_ms,
            'discovery_time_ms': r.discovery_time_ms,
            'processing_time_ms': r.processing_time_ms,
            'sync_time_ms': r.sync_time_ms,
            'consumer_read_time_ms': r.consumer_read_time_ms,
            'total_latency_ms': r.total_latency_ms,
            'files_per_second': r.files_per_second,
            'mb_per_second': r.mb_per_second,
            'files_successful': r.files_successful,
            'files_failed': r.files_failed,
            'data_validated': r.data_validated,
            'timestamp': r.timestamp,
            'errors': r.errors,
        }
        for r in results
    ]
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("E2E Latency Benchmark")
    print("=" * 60)
    print()
    print("This benchmark measures end-to-end latency from file arrival")
    print("to consumer-readable data. Run from Databricks or with GRIB files.")

