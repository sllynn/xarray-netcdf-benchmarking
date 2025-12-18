#!/usr/bin/env python3
"""
Consumer Read Latency Benchmark.

Measures read performance including NaN handling to validate the
architecture's assumptions about consumer-side read latency.
"""

import time
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class ReadTestResult:
    """Result of a single read test."""
    test_id: str
    access_pattern: str
    
    # Query parameters
    steps_requested: int
    ensemble_members: int
    spatial_subset: Optional[tuple] = None
    
    # Timing (milliseconds)
    open_time_ms: float
    read_time_ms: float
    compute_time_ms: float
    total_time_ms: float
    
    # Data statistics
    data_shape: tuple
    data_size_mb: float
    nan_percentage: float
    
    # Computed values (for validation)
    data_min: Optional[float] = None
    data_max: Optional[float] = None
    data_mean: Optional[float] = None
    
    success: bool = True
    error: Optional[str] = None
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AccessPatternResult:
    """Aggregated results for an access pattern."""
    access_pattern: str
    num_tests: int
    
    # Timing statistics (milliseconds)
    avg_total_time_ms: float
    min_total_time_ms: float
    max_total_time_ms: float
    p50_total_time_ms: float
    p95_total_time_ms: float
    
    # Breakdown
    avg_open_time_ms: float
    avg_read_time_ms: float
    avg_compute_time_ms: float
    
    # Data statistics
    avg_nan_percentage: float
    avg_data_size_mb: float
    
    # Throughput
    mb_per_second: float


@dataclass
class BenchmarkResult:
    """Complete consumer read benchmark result."""
    test_name: str
    zarr_store_path: str
    zarr_shape: dict
    
    pattern_results: list  # list[AccessPatternResult]
    all_test_results: list  # list[ReadTestResult]
    
    # Overall summary
    overall_avg_latency_ms: float
    overall_throughput_mb_per_sec: float
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def read_single_step(
    ds: xr.Dataset,
    variable: str,
    step_idx: int,
) -> tuple[xr.DataArray, float, float]:
    """Read a single time step.
    
    Returns
    -------
    tuple
        (data, read_time_ms, compute_time_ms)
    """
    read_start = time.perf_counter()
    data = ds[variable].isel(step=step_idx)
    read_time = (time.perf_counter() - read_start) * 1000
    
    compute_start = time.perf_counter()
    data_computed = data.compute()
    compute_time = (time.perf_counter() - compute_start) * 1000
    
    return data_computed, read_time, compute_time


def read_time_range(
    ds: xr.Dataset,
    variable: str,
    step_slice: slice,
) -> tuple[xr.DataArray, float, float]:
    """Read a range of time steps.
    
    Returns
    -------
    tuple
        (data, read_time_ms, compute_time_ms)
    """
    read_start = time.perf_counter()
    data = ds[variable].isel(step=step_slice)
    read_time = (time.perf_counter() - read_start) * 1000
    
    compute_start = time.perf_counter()
    data_computed = data.compute()
    compute_time = (time.perf_counter() - compute_start) * 1000
    
    return data_computed, read_time, compute_time


def read_spatial_subset(
    ds: xr.Dataset,
    variable: str,
    step_idx: int,
    lat_slice: slice,
    lon_slice: slice,
) -> tuple[xr.DataArray, float, float]:
    """Read a spatial subset.
    
    Returns
    -------
    tuple
        (data, read_time_ms, compute_time_ms)
    """
    read_start = time.perf_counter()
    data = ds[variable].isel(
        step=step_idx,
        latitude=lat_slice,
        longitude=lon_slice,
    )
    read_time = (time.perf_counter() - read_start) * 1000
    
    compute_start = time.perf_counter()
    data_computed = data.compute()
    compute_time = (time.perf_counter() - compute_start) * 1000
    
    return data_computed, read_time, compute_time


def read_ensemble_mean(
    ds: xr.Dataset,
    variable: str,
    step_idx: int,
) -> tuple[xr.DataArray, float, float]:
    """Read and compute ensemble mean for a single step.
    
    Returns
    -------
    tuple
        (data, read_time_ms, compute_time_ms)
    """
    read_start = time.perf_counter()
    data = ds[variable].isel(step=step_idx)
    read_time = (time.perf_counter() - read_start) * 1000
    
    compute_start = time.perf_counter()
    data_computed = data.mean(dim='number').compute()
    compute_time = (time.perf_counter() - compute_start) * 1000
    
    return data_computed, read_time, compute_time


def benchmark_access_pattern(
    zarr_store_path: str,
    access_pattern: str,
    variable: str = 't2m',
    num_tests: int = 5,
    nan_percentage: float = 0.0,
) -> list[ReadTestResult]:
    """Benchmark a specific access pattern.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to Zarr store.
    access_pattern : str
        Access pattern: 'single_step', 'time_range', 'spatial_subset', 'ensemble_mean'.
    variable : str
        Variable name to read.
    num_tests : int
        Number of test iterations.
    nan_percentage : float
        Expected NaN percentage (for validation).
    
    Returns
    -------
    list[ReadTestResult]
        List of test results.
    """
    results = []
    
    for i in range(num_tests):
        test_id = f"{access_pattern}_{i:03d}"
        
        try:
            # Open store
            open_start = time.perf_counter()
            ds = xr.open_zarr(zarr_store_path, consolidated=True)
            open_time = (time.perf_counter() - open_start) * 1000
            
            n_steps = ds.sizes['step']
            n_ensemble = ds.sizes['number']
            n_lat = ds.sizes['latitude']
            n_lon = ds.sizes['longitude']
            
            # Select random step for each test
            step_idx = np.random.randint(0, n_steps)
            
            if access_pattern == 'single_step':
                data, read_time, compute_time = read_single_step(ds, variable, step_idx)
                steps_requested = 1
                spatial_subset = None
                
            elif access_pattern == 'time_range':
                # Read 10 consecutive steps
                start_idx = np.random.randint(0, max(1, n_steps - 10))
                step_slice = slice(start_idx, min(start_idx + 10, n_steps))
                data, read_time, compute_time = read_time_range(ds, variable, step_slice)
                steps_requested = min(10, n_steps - start_idx)
                spatial_subset = None
                
            elif access_pattern == 'spatial_subset':
                # Read a regional subset (e.g., Europe-sized)
                lat_start = np.random.randint(0, max(1, n_lat - 50))
                lon_start = np.random.randint(0, max(1, n_lon - 100))
                lat_slice = slice(lat_start, lat_start + 50)
                lon_slice = slice(lon_start, lon_start + 100)
                data, read_time, compute_time = read_spatial_subset(
                    ds, variable, step_idx, lat_slice, lon_slice
                )
                steps_requested = 1
                spatial_subset = (lat_slice.start, lat_slice.stop, lon_slice.start, lon_slice.stop)
                
            elif access_pattern == 'ensemble_mean':
                data, read_time, compute_time = read_ensemble_mean(ds, variable, step_idx)
                steps_requested = 1
                spatial_subset = None
                
            else:
                raise ValueError(f"Unknown access pattern: {access_pattern}")
            
            ds.close()
            
            total_time = open_time + read_time + compute_time
            
            # Calculate statistics
            data_values = data.values
            nan_pct = float(np.isnan(data_values).mean() * 100)
            data_size_mb = data_values.nbytes / 1024 / 1024
            
            # Compute min/max/mean (excluding NaN)
            valid_data = data_values[~np.isnan(data_values)]
            if len(valid_data) > 0:
                data_min = float(np.min(valid_data))
                data_max = float(np.max(valid_data))
                data_mean = float(np.mean(valid_data))
            else:
                data_min = data_max = data_mean = None
            
            results.append(ReadTestResult(
                test_id=test_id,
                access_pattern=access_pattern,
                steps_requested=steps_requested,
                ensemble_members=n_ensemble,
                spatial_subset=spatial_subset,
                open_time_ms=open_time,
                read_time_ms=read_time,
                compute_time_ms=compute_time,
                total_time_ms=total_time,
                data_shape=data_values.shape,
                data_size_mb=data_size_mb,
                nan_percentage=nan_pct,
                data_min=data_min,
                data_max=data_max,
                data_mean=data_mean,
                success=True,
            ))
            
        except Exception as e:
            logger.error(f"Error in test {test_id}: {e}")
            results.append(ReadTestResult(
                test_id=test_id,
                access_pattern=access_pattern,
                steps_requested=0,
                ensemble_members=0,
                open_time_ms=0,
                read_time_ms=0,
                compute_time_ms=0,
                total_time_ms=0,
                data_shape=(),
                data_size_mb=0,
                nan_percentage=0,
                success=False,
                error=str(e),
            ))
    
    return results


def benchmark_consumer_reads(
    zarr_store_path: str,
    nan_percentage: float = 0.0,
    access_patterns: list[str] = None,
    tests_per_pattern: int = 5,
) -> BenchmarkResult:
    """Run comprehensive consumer read benchmark.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to Zarr store.
    nan_percentage : float
        Expected NaN percentage in data (simulates partial fill).
    access_patterns : list[str], optional
        Access patterns to test. Default: all patterns.
    tests_per_pattern : int
        Number of tests per pattern (default: 5).
    
    Returns
    -------
    BenchmarkResult
        Complete benchmark results.
    """
    if access_patterns is None:
        access_patterns = ['single_step', 'time_range', 'spatial_subset', 'ensemble_mean']
    
    logger.info(f"Starting consumer read benchmark")
    logger.info(f"  Store: {zarr_store_path}")
    logger.info(f"  Patterns: {access_patterns}")
    logger.info(f"  Tests per pattern: {tests_per_pattern}")
    
    # Get store info
    ds = xr.open_zarr(zarr_store_path, consolidated=True)
    zarr_shape = dict(ds.sizes)
    ds.close()
    
    all_results = []
    pattern_results = []
    
    for pattern in access_patterns:
        logger.info(f"Testing pattern: {pattern}")
        
        results = benchmark_access_pattern(
            zarr_store_path,
            pattern,
            num_tests=tests_per_pattern,
            nan_percentage=nan_percentage,
        )
        
        all_results.extend(results)
        
        # Aggregate results for this pattern
        successful = [r for r in results if r.success]
        
        if successful:
            times = [r.total_time_ms for r in successful]
            pattern_result = AccessPatternResult(
                access_pattern=pattern,
                num_tests=len(successful),
                avg_total_time_ms=np.mean(times),
                min_total_time_ms=np.min(times),
                max_total_time_ms=np.max(times),
                p50_total_time_ms=np.percentile(times, 50),
                p95_total_time_ms=np.percentile(times, 95),
                avg_open_time_ms=np.mean([r.open_time_ms for r in successful]),
                avg_read_time_ms=np.mean([r.read_time_ms for r in successful]),
                avg_compute_time_ms=np.mean([r.compute_time_ms for r in successful]),
                avg_nan_percentage=np.mean([r.nan_percentage for r in successful]),
                avg_data_size_mb=np.mean([r.data_size_mb for r in successful]),
                mb_per_second=np.sum([r.data_size_mb for r in successful]) / (np.sum(times) / 1000) if np.sum(times) > 0 else 0,
            )
        else:
            pattern_result = AccessPatternResult(
                access_pattern=pattern,
                num_tests=0,
                avg_total_time_ms=0,
                min_total_time_ms=0,
                max_total_time_ms=0,
                p50_total_time_ms=0,
                p95_total_time_ms=0,
                avg_open_time_ms=0,
                avg_read_time_ms=0,
                avg_compute_time_ms=0,
                avg_nan_percentage=0,
                avg_data_size_mb=0,
                mb_per_second=0,
            )
        
        pattern_results.append(pattern_result)
        
        logger.info(
            f"  {pattern}: avg={pattern_result.avg_total_time_ms:.1f}ms, "
            f"p95={pattern_result.p95_total_time_ms:.1f}ms"
        )
    
    # Calculate overall statistics
    all_successful = [r for r in all_results if r.success]
    if all_successful:
        overall_avg_latency = np.mean([r.total_time_ms for r in all_successful])
        total_mb = sum(r.data_size_mb for r in all_successful)
        total_time_sec = sum(r.total_time_ms for r in all_successful) / 1000
        overall_throughput = total_mb / total_time_sec if total_time_sec > 0 else 0
    else:
        overall_avg_latency = 0
        overall_throughput = 0
    
    return BenchmarkResult(
        test_name='consumer_read',
        zarr_store_path=zarr_store_path,
        zarr_shape=zarr_shape,
        pattern_results=pattern_results,
        all_test_results=all_results,
        overall_avg_latency_ms=overall_avg_latency,
        overall_throughput_mb_per_sec=overall_throughput,
    )


def save_results(result: BenchmarkResult, output_path: str) -> None:
    """Save benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'test_name': result.test_name,
        'zarr_store_path': result.zarr_store_path,
        'zarr_shape': result.zarr_shape,
        'overall_avg_latency_ms': result.overall_avg_latency_ms,
        'overall_throughput_mb_per_sec': result.overall_throughput_mb_per_sec,
        'timestamp': result.timestamp,
        'pattern_results': [
            {
                'access_pattern': r.access_pattern,
                'num_tests': r.num_tests,
                'avg_total_time_ms': r.avg_total_time_ms,
                'min_total_time_ms': r.min_total_time_ms,
                'max_total_time_ms': r.max_total_time_ms,
                'p50_total_time_ms': r.p50_total_time_ms,
                'p95_total_time_ms': r.p95_total_time_ms,
                'avg_open_time_ms': r.avg_open_time_ms,
                'avg_read_time_ms': r.avg_read_time_ms,
                'avg_compute_time_ms': r.avg_compute_time_ms,
                'avg_nan_percentage': r.avg_nan_percentage,
                'avg_data_size_mb': r.avg_data_size_mb,
                'mb_per_second': r.mb_per_second,
            }
            for r in result.pattern_results
        ],
        'test_results': [
            {
                'test_id': r.test_id,
                'access_pattern': r.access_pattern,
                'steps_requested': r.steps_requested,
                'total_time_ms': r.total_time_ms,
                'data_size_mb': r.data_size_mb,
                'nan_percentage': r.nan_percentage,
                'success': r.success,
                'error': r.error,
            }
            for r in result.all_test_results
        ],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def run_benchmark_with_test_store() -> BenchmarkResult:
    """Run benchmark with a temporary test Zarr store.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results.
    """
    from .region_write import create_test_zarr_store
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "test.zarr"
        
        logger.info("Creating test Zarr store...")
        create_test_zarr_store(
            str(zarr_path),
            n_steps=50,
            n_ensemble=50,
            n_lat=361,
            n_lon=720,
        )
        
        logger.info("Running consumer read benchmark...")
        result = benchmark_consumer_reads(
            str(zarr_path),
            nan_percentage=0,
            tests_per_pattern=3,
        )
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Consumer Read Latency Benchmark")
    print("=" * 60)
    print()
    
    result = run_benchmark_with_test_store()
    
    print()
    print("Summary:")
    print(f"  Overall avg latency: {result.overall_avg_latency_ms:.1f} ms")
    print(f"  Overall throughput: {result.overall_throughput_mb_per_sec:.1f} MB/s")
    print()
    print("By access pattern:")
    for pr in result.pattern_results:
        print(f"  {pr.access_pattern}: avg={pr.avg_total_time_ms:.1f}ms, p95={pr.p95_total_time_ms:.1f}ms")

