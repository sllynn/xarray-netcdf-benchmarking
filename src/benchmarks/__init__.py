"""
Benchmarking framework for the weather data pipeline.

Provides comprehensive performance measurement for:
- End-to-end latency
- Zarr region write performance
- AzCopy sync performance
- Consumer read latency
"""

from .e2e_latency import benchmark_e2e_latency
from .region_write import benchmark_region_writes
from .azcopy_sync import benchmark_azcopy_sync
from .consumer_read import benchmark_consumer_reads

__all__ = [
    "benchmark_e2e_latency",
    "benchmark_region_writes",
    "benchmark_azcopy_sync",
    "benchmark_consumer_reads",
]

