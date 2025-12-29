"""Benchmarking framework for the weather data pipeline.

IMPORTANT
---------
Keep imports in this package *lazy*.

Databricks notebooks often import a single helper (for example
[`src.benchmarks.streaming_harness`](src/benchmarks/streaming_harness.py:1)).
If we eagerly import every benchmark module here, a failure in any benchmark
(e.g. environment-specific dependency or dataclass init issue) breaks unrelated
workflows like the mock producer.

Callers should import the specific benchmark they need, e.g.:
- `from src.benchmarks.e2e_latency import benchmark_e2e_latency`
- `from src.benchmarks.consumer_read import benchmark_consumer_reads`
"""

__all__: list[str] = []

