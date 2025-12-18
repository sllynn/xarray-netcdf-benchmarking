#!/usr/bin/env python3
"""
Region Write Performance Benchmark.

Measures Zarr region write throughput with various concurrency levels
and data sizes to validate the architecture's write performance assumptions.
"""

import os
import time
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import numpy as np
import xarray as xr
import zarr

logger = logging.getLogger(__name__)


@dataclass
class RegionWriteResult:
    """Result of a single region write operation."""
    step_index: int
    write_time_ms: float
    data_size_bytes: int
    success: bool
    error: Optional[str] = None


@dataclass
class ConcurrencyTestResult:
    """Result of testing a specific concurrency level."""
    concurrency: int
    num_writes: int
    successful_writes: int
    failed_writes: int
    
    # Timing metrics (milliseconds)
    total_time_ms: float
    avg_write_time_ms: float
    min_write_time_ms: float
    max_write_time_ms: float
    p50_write_time_ms: float
    p95_write_time_ms: float
    p99_write_time_ms: float
    
    # Throughput metrics
    writes_per_second: float
    mb_per_second: float
    
    # Per-write results
    write_times_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Complete benchmark result across all concurrency levels."""
    test_name: str
    zarr_shape: tuple
    chunk_size_mb: float
    concurrency_results: list  # list[ConcurrencyTestResult]
    optimal_concurrency: int
    optimal_throughput_writes_per_sec: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def create_test_zarr_store(
    output_path: str,
    n_steps: int = 145,
    n_ensemble: int = 50,
    n_lat: int = 361,
    n_lon: int = 720,
    variables: list[str] = None,
) -> str:
    """Create a test Zarr store for benchmarking.
    
    Parameters
    ----------
    output_path : str
        Path for the Zarr store.
    n_steps : int
        Number of time steps (default: 145).
    n_ensemble : int
        Number of ensemble members (default: 50).
    n_lat : int
        Number of latitude points (default: 361).
    n_lon : int
        Number of longitude points (default: 720).
    variables : list[str], optional
        Variable names (default: ['t2m']).
    
    Returns
    -------
    str
        Path to created Zarr store.
    """
    from ..zarr_init import initialize_zarr_store, generate_forecast_steps
    
    if variables is None:
        variables = ['t2m']
    
    forecast_steps = generate_forecast_steps()
    
    store = initialize_zarr_store(
        output_path=output_path,
        variables=variables,
        ensemble_members=n_ensemble,
        lat_size=n_lat,
        lon_size=n_lon,
        forecast_steps=forecast_steps[:n_steps],
    )
    
    return output_path


def generate_test_data(
    n_ensemble: int,
    n_lat: int,
    n_lon: int,
    dtype=np.float32,
) -> np.ndarray:
    """Generate random test data for a single time step.
    
    Parameters
    ----------
    n_ensemble : int
        Number of ensemble members.
    n_lat : int
        Number of latitude points.
    n_lon : int
        Number of longitude points.
    dtype : numpy.dtype
        Data type (default: float32).
    
    Returns
    -------
    np.ndarray
        Random data array with shape (1, n_ensemble, n_lat, n_lon).
    """
    # Use realistic temperature range (220K to 320K)
    return np.random.uniform(220, 320, (1, n_ensemble, n_lat, n_lon)).astype(dtype)


def write_region(
    zarr_store_path: str,
    step_index: int,
    variable: str,
    data: np.ndarray,
) -> RegionWriteResult:
    """Write data to a specific region of the Zarr store.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to Zarr store.
    step_index : int
        Time step index to write.
    variable : str
        Variable name.
    data : np.ndarray
        Data array with shape (1, n_ensemble, n_lat, n_lon).
    
    Returns
    -------
    RegionWriteResult
        Result of the write operation.
    """
    start_time = time.perf_counter()
    
    try:
        # Create minimal dataset for region write
        write_ds = xr.Dataset({
            variable: (['step', 'number', 'latitude', 'longitude'], data)
        })
        
        # Define region slice
        region = {
            'step': slice(step_index, step_index + 1),
            'number': slice(None),
            'latitude': slice(None),
            'longitude': slice(None),
        }
        
        # Perform region write
        write_ds.to_zarr(
            zarr_store_path,
            mode='r+',
            region=region,
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return RegionWriteResult(
            step_index=step_index,
            write_time_ms=elapsed_ms,
            data_size_bytes=data.nbytes,
            success=True,
        )
        
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return RegionWriteResult(
            step_index=step_index,
            write_time_ms=elapsed_ms,
            data_size_bytes=data.nbytes,
            success=False,
            error=str(e),
        )


def benchmark_concurrency_level(
    zarr_store_path: str,
    concurrency: int,
    num_writes: int,
    variable: str = 't2m',
    data_shape: tuple = None,
) -> ConcurrencyTestResult:
    """Benchmark a specific concurrency level.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to Zarr store.
    concurrency : int
        Number of parallel workers.
    num_writes : int
        Total number of writes to perform.
    variable : str
        Variable name to write.
    data_shape : tuple, optional
        Shape of data (n_ensemble, n_lat, n_lon). Inferred from store if None.
    
    Returns
    -------
    ConcurrencyTestResult
        Results for this concurrency level.
    """
    logger.info(f"Testing concurrency={concurrency}, writes={num_writes}")
    
    # Get shape from store if not provided
    if data_shape is None:
        ds = xr.open_zarr(zarr_store_path, consolidated=True)
        data_shape = (ds.sizes['number'], ds.sizes['latitude'], ds.sizes['longitude'])
        ds.close()
    
    n_ensemble, n_lat, n_lon = data_shape
    
    # Generate test data for each write
    test_data = [
        generate_test_data(n_ensemble, n_lat, n_lon)
        for _ in range(num_writes)
    ]
    
    results = []
    errors = []
    
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                write_region,
                zarr_store_path,
                i,
                variable,
                test_data[i],
            ): i
            for i in range(num_writes)
        }
        
        for future in as_completed(futures):
            step_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                if not result.success:
                    errors.append(f"Step {step_idx}: {result.error}")
            except Exception as e:
                logger.error(f"Error writing step {step_idx}: {e}")
                errors.append(f"Step {step_idx}: {str(e)}")
    
    total_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Calculate statistics
    successful = [r for r in results if r.success]
    write_times = [r.write_time_ms for r in successful]
    
    if write_times:
        avg_write_time = np.mean(write_times)
        min_write_time = np.min(write_times)
        max_write_time = np.max(write_times)
        p50_write_time = np.percentile(write_times, 50)
        p95_write_time = np.percentile(write_times, 95)
        p99_write_time = np.percentile(write_times, 99)
    else:
        avg_write_time = min_write_time = max_write_time = 0
        p50_write_time = p95_write_time = p99_write_time = 0
    
    # Calculate throughput
    total_time_sec = total_time_ms / 1000.0
    writes_per_second = len(successful) / total_time_sec if total_time_sec > 0 else 0
    
    # Calculate data throughput
    chunk_size_bytes = test_data[0].nbytes if test_data else 0
    total_bytes = len(successful) * chunk_size_bytes
    mb_per_second = (total_bytes / 1024 / 1024) / total_time_sec if total_time_sec > 0 else 0
    
    return ConcurrencyTestResult(
        concurrency=concurrency,
        num_writes=num_writes,
        successful_writes=len(successful),
        failed_writes=len(results) - len(successful),
        total_time_ms=total_time_ms,
        avg_write_time_ms=avg_write_time,
        min_write_time_ms=min_write_time,
        max_write_time_ms=max_write_time,
        p50_write_time_ms=p50_write_time,
        p95_write_time_ms=p95_write_time,
        p99_write_time_ms=p99_write_time,
        writes_per_second=writes_per_second,
        mb_per_second=mb_per_second,
        write_times_ms=write_times,
        errors=errors,
    )


def benchmark_region_writes(
    zarr_store_path: str,
    num_writes: int = 50,
    concurrency_levels: list[int] = None,
    variable: str = 't2m',
) -> BenchmarkResult:
    """Run comprehensive region write benchmark.
    
    Tests various concurrency levels to find optimal parallelism.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to Zarr store.
    num_writes : int
        Number of writes to perform per concurrency level (default: 50).
    concurrency_levels : list[int], optional
        Concurrency levels to test (default: [1, 2, 4, 8, 16, 32]).
    variable : str
        Variable name to write.
    
    Returns
    -------
    BenchmarkResult
        Complete benchmark results.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16, 32]
    
    # Get store shape
    ds = xr.open_zarr(zarr_store_path, consolidated=True)
    zarr_shape = dict(ds.sizes)
    data_shape = (ds.sizes['number'], ds.sizes['latitude'], ds.sizes['longitude'])
    ds.close()
    
    # Calculate chunk size
    chunk_size_bytes = data_shape[0] * data_shape[1] * data_shape[2] * 4  # float32
    chunk_size_mb = chunk_size_bytes / 1024 / 1024
    
    logger.info(f"Starting region write benchmark")
    logger.info(f"  Zarr shape: {zarr_shape}")
    logger.info(f"  Chunk size: {chunk_size_mb:.2f} MB")
    logger.info(f"  Writes per level: {num_writes}")
    logger.info(f"  Concurrency levels: {concurrency_levels}")
    
    concurrency_results = []
    
    for concurrency in concurrency_levels:
        # Re-initialize store for each test to ensure clean state
        result = benchmark_concurrency_level(
            zarr_store_path,
            concurrency,
            min(num_writes, zarr_shape.get('step', 145)),  # Don't exceed steps
            variable,
            data_shape,
        )
        concurrency_results.append(result)
        
        logger.info(
            f"  Concurrency {concurrency:2d}: "
            f"{result.writes_per_second:.1f} writes/s, "
            f"avg {result.avg_write_time_ms:.1f}ms"
        )
    
    # Find optimal concurrency
    successful_results = [r for r in concurrency_results if r.successful_writes > 0]
    if successful_results:
        best = max(successful_results, key=lambda r: r.writes_per_second)
        optimal_concurrency = best.concurrency
        optimal_throughput = best.writes_per_second
    else:
        optimal_concurrency = 1
        optimal_throughput = 0
    
    logger.info(f"\nOptimal concurrency: {optimal_concurrency} ({optimal_throughput:.1f} writes/s)")
    
    return BenchmarkResult(
        test_name='region_write',
        zarr_shape=tuple(zarr_shape.items()),
        chunk_size_mb=chunk_size_mb,
        concurrency_results=concurrency_results,
        optimal_concurrency=optimal_concurrency,
        optimal_throughput_writes_per_sec=optimal_throughput,
    )


def save_results(result: BenchmarkResult, output_path: str) -> None:
    """Save benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'test_name': result.test_name,
        'zarr_shape': dict(result.zarr_shape),
        'chunk_size_mb': result.chunk_size_mb,
        'optimal_concurrency': result.optimal_concurrency,
        'optimal_throughput_writes_per_sec': result.optimal_throughput_writes_per_sec,
        'timestamp': result.timestamp,
        'concurrency_results': [
            {
                'concurrency': r.concurrency,
                'num_writes': r.num_writes,
                'successful_writes': r.successful_writes,
                'failed_writes': r.failed_writes,
                'total_time_ms': r.total_time_ms,
                'avg_write_time_ms': r.avg_write_time_ms,
                'min_write_time_ms': r.min_write_time_ms,
                'max_write_time_ms': r.max_write_time_ms,
                'p50_write_time_ms': r.p50_write_time_ms,
                'p95_write_time_ms': r.p95_write_time_ms,
                'p99_write_time_ms': r.p99_write_time_ms,
                'writes_per_second': r.writes_per_second,
                'mb_per_second': r.mb_per_second,
                'errors': r.errors,
            }
            for r in result.concurrency_results
        ],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def run_quick_benchmark(output_dir: str = None) -> BenchmarkResult:
    """Run a quick benchmark with temporary Zarr store.
    
    Useful for quick validation of write performance.
    
    Parameters
    ----------
    output_dir : str, optional
        Directory for results. Uses temp directory if None.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results.
    """
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "benchmark.zarr"
        
        logger.info("Creating test Zarr store...")
        create_test_zarr_store(
            str(zarr_path),
            n_steps=50,  # Smaller for quick test
            n_ensemble=50,
            n_lat=361,
            n_lon=720,
        )
        
        logger.info("Running benchmark...")
        result = benchmark_region_writes(
            str(zarr_path),
            num_writes=20,  # Fewer writes for quick test
            concurrency_levels=[1, 4, 8, 16],
        )
        
        if output_dir:
            save_results(
                result,
                Path(output_dir) / f"region_write_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Region Write Performance Benchmark")
    print("=" * 60)
    print()
    
    # Run quick benchmark
    result = run_quick_benchmark()
    
    print()
    print("Summary:")
    print(f"  Optimal concurrency: {result.optimal_concurrency}")
    print(f"  Best throughput: {result.optimal_throughput_writes_per_sec:.1f} writes/s")
    print(f"  Chunk size: {result.chunk_size_mb:.2f} MB")

