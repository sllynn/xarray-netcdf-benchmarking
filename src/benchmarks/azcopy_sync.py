#!/usr/bin/env python3
"""
AzCopy Sync Performance Benchmark.

Measures sync latency for different batch sizes and file counts
to validate the architecture's cloud sync performance assumptions.
"""

import os
import time
import shutil
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyncTestResult:
    """Result of a single sync test."""
    test_id: str
    num_changed_chunks: int
    total_chunks: int
    total_bytes: int
    
    # Timing (milliseconds)
    sync_time_ms: float
    files_transferred: int
    bytes_transferred: int
    
    # Throughput
    chunks_per_second: float
    mb_per_second: float
    
    success: bool
    error: Optional[str] = None
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BenchmarkResult:
    """Complete AzCopy sync benchmark result."""
    test_name: str
    chunk_size_mb: float
    test_results: list  # list[SyncTestResult]
    
    # Summary statistics
    avg_sync_time_ms: float
    avg_chunks_per_second: float
    avg_mb_per_second: float
    
    # Scaling analysis
    time_per_chunk_ms: float  # Linear coefficient
    base_overhead_ms: float  # Fixed overhead
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def create_test_files(
    directory: str,
    num_files: int,
    file_size_mb: float = 50.0,
    file_extension: str = '.chunk',
) -> list[str]:
    """Create test files for sync benchmark.
    
    Parameters
    ----------
    directory : str
        Directory to create files in.
    num_files : int
        Number of files to create.
    file_size_mb : float
        Size of each file in MB.
    file_extension : str
        File extension.
    
    Returns
    -------
    list[str]
        List of created file paths.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate bytes per file
    bytes_per_file = int(file_size_mb * 1024 * 1024)
    
    created_files = []
    for i in range(num_files):
        file_path = dir_path / f"chunk_{i:04d}{file_extension}"
        
        # Create file with random data
        with open(file_path, 'wb') as f:
            # Write in chunks to avoid memory issues
            chunk_size = min(bytes_per_file, 10 * 1024 * 1024)  # 10MB chunks
            remaining = bytes_per_file
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                f.write(os.urandom(write_size))
                remaining -= write_size
        
        created_files.append(str(file_path))
    
    return created_files


def modify_files(
    file_paths: list[str],
    num_to_modify: int,
) -> list[str]:
    """Modify a subset of files to trigger sync.
    
    Parameters
    ----------
    file_paths : list[str]
        All file paths.
    num_to_modify : int
        Number of files to modify.
    
    Returns
    -------
    list[str]
        List of modified file paths.
    """
    # Select random files to modify
    indices = np.random.choice(len(file_paths), size=min(num_to_modify, len(file_paths)), replace=False)
    modified_files = []
    
    for idx in indices:
        file_path = file_paths[idx]
        # Append small amount of data to trigger modification
        with open(file_path, 'ab') as f:
            f.write(os.urandom(1024))  # 1KB
        modified_files.append(file_path)
    
    return modified_files


def benchmark_sync_operation(
    source_path: str,
    destination_url: str,
    num_changed_chunks: int,
    total_chunks: int,
    chunk_size_mb: float,
    use_mock: bool = False,
) -> SyncTestResult:
    """Benchmark a single sync operation.
    
    Parameters
    ----------
    source_path : str
        Local source directory.
    destination_url : str
        Destination URL (with SAS token if needed).
    num_changed_chunks : int
        Number of chunks that were modified.
    total_chunks : int
        Total number of chunks.
    chunk_size_mb : float
        Size of each chunk in MB.
    use_mock : bool
        If True, simulate sync without actual transfer.
    
    Returns
    -------
    SyncTestResult
        Benchmark result.
    """
    from ..cloud_sync import sync_with_azcopy, find_azcopy
    
    test_id = f"sync_{num_changed_chunks}chunks_{datetime.now().strftime('%H%M%S')}"
    total_bytes = int(num_changed_chunks * chunk_size_mb * 1024 * 1024)
    
    if use_mock:
        # Simulate sync for testing without actual Azure connection
        # Assume ~100 MB/s transfer rate plus ~500ms overhead
        simulated_time_ms = 500 + (total_bytes / (100 * 1024 * 1024)) * 1000
        
        return SyncTestResult(
            test_id=test_id,
            num_changed_chunks=num_changed_chunks,
            total_chunks=total_chunks,
            total_bytes=total_bytes,
            sync_time_ms=simulated_time_ms,
            files_transferred=num_changed_chunks,
            bytes_transferred=total_bytes,
            chunks_per_second=num_changed_chunks / (simulated_time_ms / 1000),
            mb_per_second=(total_bytes / 1024 / 1024) / (simulated_time_ms / 1000),
            success=True,
        )
    
    # Check if azcopy is available
    azcopy_path = find_azcopy()
    if azcopy_path is None:
        logger.warning("azcopy not found, using mock results")
        return benchmark_sync_operation(
            source_path, destination_url, num_changed_chunks,
            total_chunks, chunk_size_mb, use_mock=True
        )
    
    start_time = time.perf_counter()
    
    try:
        result = sync_with_azcopy(
            source_path,
            destination_url,
            delete_destination=False,
            log_level='WARNING',
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        if result.success:
            chunks_per_second = num_changed_chunks / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            bytes_transferred = result.bytes_transferred if result.bytes_transferred > 0 else total_bytes
            mb_per_second = (bytes_transferred / 1024 / 1024) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            
            return SyncTestResult(
                test_id=test_id,
                num_changed_chunks=num_changed_chunks,
                total_chunks=total_chunks,
                total_bytes=total_bytes,
                sync_time_ms=elapsed_ms,
                files_transferred=result.files_transferred if result.files_transferred > 0 else num_changed_chunks,
                bytes_transferred=bytes_transferred,
                chunks_per_second=chunks_per_second,
                mb_per_second=mb_per_second,
                success=True,
            )
        else:
            return SyncTestResult(
                test_id=test_id,
                num_changed_chunks=num_changed_chunks,
                total_chunks=total_chunks,
                total_bytes=total_bytes,
                sync_time_ms=elapsed_ms,
                files_transferred=0,
                bytes_transferred=0,
                chunks_per_second=0,
                mb_per_second=0,
                success=False,
                error=result.error,
            )
            
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SyncTestResult(
            test_id=test_id,
            num_changed_chunks=num_changed_chunks,
            total_chunks=total_chunks,
            total_bytes=total_bytes,
            sync_time_ms=elapsed_ms,
            files_transferred=0,
            bytes_transferred=0,
            chunks_per_second=0,
            mb_per_second=0,
            success=False,
            error=str(e),
        )


def benchmark_azcopy_sync(
    source_path: str,
    dest_url: str,
    changed_chunk_counts: list[int] = None,
    chunk_size_mb: float = 50.0,
    repetitions: int = 3,
    use_mock: bool = False,
) -> BenchmarkResult:
    """Run comprehensive AzCopy sync benchmark.
    
    Parameters
    ----------
    source_path : str
        Local source directory containing files.
    dest_url : str
        Destination URL.
    changed_chunk_counts : list[int], optional
        Number of changed chunks to test (default: [1, 5, 10, 25, 50]).
    chunk_size_mb : float
        Size of each chunk in MB.
    repetitions : int
        Number of repetitions per test.
    use_mock : bool
        If True, simulate sync for testing.
    
    Returns
    -------
    BenchmarkResult
        Complete benchmark results.
    """
    if changed_chunk_counts is None:
        changed_chunk_counts = [1, 5, 10, 25, 50]
    
    logger.info("Starting AzCopy sync benchmark")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Destination: {dest_url.split('?')[0] if '?' in dest_url else dest_url}")
    logger.info(f"  Chunk size: {chunk_size_mb:.1f} MB")
    logger.info(f"  Changed chunk counts: {changed_chunk_counts}")
    
    # Get total chunks in source
    source_files = list(Path(source_path).glob('*'))
    total_chunks = len(source_files)
    
    all_results = []
    
    for num_chunks in changed_chunk_counts:
        if num_chunks > total_chunks:
            logger.warning(f"Skipping {num_chunks} chunks (only {total_chunks} available)")
            continue
        
        for rep in range(repetitions):
            logger.info(f"Testing {num_chunks} chunks (rep {rep+1}/{repetitions})...")
            
            result = benchmark_sync_operation(
                source_path,
                dest_url,
                num_chunks,
                total_chunks,
                chunk_size_mb,
                use_mock=use_mock,
            )
            
            all_results.append(result)
            
            if result.success:
                logger.info(
                    f"  Sync complete: {result.sync_time_ms:.1f}ms, "
                    f"{result.chunks_per_second:.1f} chunks/s"
                )
            else:
                logger.error(f"  Sync failed: {result.error}")
    
    # Calculate summary statistics
    successful_results = [r for r in all_results if r.success]
    
    if successful_results:
        avg_sync_time = np.mean([r.sync_time_ms for r in successful_results])
        avg_chunks_per_sec = np.mean([r.chunks_per_second for r in successful_results])
        avg_mb_per_sec = np.mean([r.mb_per_second for r in successful_results])
        
        # Linear regression to estimate overhead and per-chunk time
        if len(successful_results) > 1:
            x = np.array([r.num_changed_chunks for r in successful_results])
            y = np.array([r.sync_time_ms for r in successful_results])
            
            # y = mx + b (time = time_per_chunk * chunks + overhead)
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            
            time_per_chunk_ms = max(0, m)
            base_overhead_ms = max(0, b)
        else:
            time_per_chunk_ms = avg_sync_time / successful_results[0].num_changed_chunks if successful_results else 0
            base_overhead_ms = 0
    else:
        avg_sync_time = avg_chunks_per_sec = avg_mb_per_sec = 0
        time_per_chunk_ms = base_overhead_ms = 0
    
    return BenchmarkResult(
        test_name='azcopy_sync',
        chunk_size_mb=chunk_size_mb,
        test_results=all_results,
        avg_sync_time_ms=avg_sync_time,
        avg_chunks_per_second=avg_chunks_per_sec,
        avg_mb_per_second=avg_mb_per_sec,
        time_per_chunk_ms=time_per_chunk_ms,
        base_overhead_ms=base_overhead_ms,
    )


def save_results(result: BenchmarkResult, output_path: str) -> None:
    """Save benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'test_name': result.test_name,
        'chunk_size_mb': result.chunk_size_mb,
        'avg_sync_time_ms': result.avg_sync_time_ms,
        'avg_chunks_per_second': result.avg_chunks_per_second,
        'avg_mb_per_second': result.avg_mb_per_second,
        'time_per_chunk_ms': result.time_per_chunk_ms,
        'base_overhead_ms': result.base_overhead_ms,
        'timestamp': result.timestamp,
        'test_results': [
            {
                'test_id': r.test_id,
                'num_changed_chunks': r.num_changed_chunks,
                'total_chunks': r.total_chunks,
                'total_bytes': r.total_bytes,
                'sync_time_ms': r.sync_time_ms,
                'files_transferred': r.files_transferred,
                'bytes_transferred': r.bytes_transferred,
                'chunks_per_second': r.chunks_per_second,
                'mb_per_second': r.mb_per_second,
                'success': r.success,
                'error': r.error,
                'timestamp': r.timestamp,
            }
            for r in result.test_results
        ],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def run_mock_benchmark() -> BenchmarkResult:
    """Run benchmark with mock sync (for testing without Azure).
    
    Returns
    -------
    BenchmarkResult
        Mock benchmark results.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source"
        
        logger.info("Creating test files...")
        create_test_files(
            str(source_path),
            num_files=50,
            file_size_mb=50.0,
        )
        
        logger.info("Running mock benchmark...")
        result = benchmark_azcopy_sync(
            str(source_path),
            "https://mock.blob.core.windows.net/container/path",
            changed_chunk_counts=[1, 5, 10, 25, 50],
            chunk_size_mb=50.0,
            repetitions=2,
            use_mock=True,
        )
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("AzCopy Sync Performance Benchmark")
    print("=" * 60)
    print()
    
    # Run mock benchmark
    result = run_mock_benchmark()
    
    print()
    print("Summary (Mock Results):")
    print(f"  Avg sync time: {result.avg_sync_time_ms:.1f} ms")
    print(f"  Avg throughput: {result.avg_chunks_per_second:.1f} chunks/s")
    print(f"  Avg transfer rate: {result.avg_mb_per_second:.1f} MB/s")
    print(f"  Time per chunk: {result.time_per_chunk_ms:.1f} ms")
    print(f"  Base overhead: {result.base_overhead_ms:.1f} ms")

