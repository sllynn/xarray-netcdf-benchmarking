#!/usr/bin/env python3
"""
Streaming Pipeline Module.

Implements Spark Structured Streaming pipeline with AutoLoader and foreachBatch
for low-latency weather data ingestion.

Key features:
- AutoLoader with file notification mode for efficient file discovery
- foreachBatch sink with ThreadPoolExecutor for parallel processing
- Batch processing: collect paths, parallel GRIB processing, single AzCopy sync
- Error handling and logging
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the streaming pipeline.
    
    Parameters
    ----------
    landing_zone : str
        Path to the landing zone where GRIB files arrive.
    zarr_store_path : str
        Path to the local Zarr store (on SSD).
    cloud_destination : str
        Unity Catalog Volume path for cloud storage.
    checkpoint_path : str
        Path for Spark checkpoint location.
    max_files_per_batch : int
        Maximum files to process per micro-batch (default: 32).
    num_workers : int
        Number of parallel workers for GRIB processing (default: 32).
    trigger_interval : str
        Spark trigger interval (default: '0 seconds' for immediate).
    use_file_notification : bool
        Use file notification mode for AutoLoader (default: True).
    sync_after_each_batch : bool
        Sync to cloud after each batch (default: True).
    """
    landing_zone: str
    zarr_store_path: str
    cloud_destination: str
    checkpoint_path: str
    max_files_per_batch: int = 32
    num_workers: int = 32
    trigger_interval: str = '0 seconds'
    use_file_notification: bool = True
    sync_after_each_batch: bool = True


@dataclass
class BatchResult:
    """Result of processing a micro-batch."""
    batch_id: int
    files_processed: int
    files_successful: int
    files_failed: int
    processing_time_ms: float
    sync_time_ms: float
    total_time_ms: float
    errors: list = field(default_factory=list)


def create_streaming_pipeline(
    spark,
    config: PipelineConfig,
    zarr_store=None,
    hour_to_index: Optional[dict] = None,
    cloud_syncer=None,
    on_batch_complete: Optional[Callable[[BatchResult], None]] = None,
):
    """Create a Spark Structured Streaming pipeline for GRIB ingestion.
    
    This function sets up the complete streaming pipeline:
    1. AutoLoader monitors landing_zone for new GRIB files
    2. foreachBatch collects file paths and processes them in parallel
    3. ThreadPoolExecutor writes GRIB data to Zarr using region writes
    4. AzCopy syncs changed chunks to cloud storage
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    config : PipelineConfig
        Pipeline configuration.
    zarr_store : zarr.Group, optional
        Pre-initialized Zarr store. If None, opens from config.zarr_store_path.
    hour_to_index : dict, optional
        Mapping from forecast hour to step index.
    cloud_syncer : CloudSyncer, optional
        Syncer for cloud upload. If None, creates from config.cloud_destination.
    on_batch_complete : callable, optional
        Callback function called after each batch completes.
    
    Returns
    -------
    StreamingQuery
        The running streaming query.
    
    Examples
    --------
    >>> config = PipelineConfig(
    ...     landing_zone='/Volumes/catalog/schema/bronze/grib/',
    ...     zarr_store_path='/local_disk0/forecast.zarr',
    ...     cloud_destination='/Volumes/catalog/schema/silver/',
    ...     checkpoint_path='/Volumes/catalog/schema/checkpoints/grib_pipeline',
    ... )
    >>> query = create_streaming_pipeline(spark, config)
    >>> query.awaitTermination()
    """
    from .zarr_init import generate_forecast_steps, build_hour_to_index_map
    from .region_writer import write_grib_to_zarr_region
    from .cloud_sync import CloudSyncer
    
    # Initialize hour-to-index mapping if not provided
    if hour_to_index is None:
        forecast_steps = generate_forecast_steps()
        hour_to_index = build_hour_to_index_map(forecast_steps)
    
    # Initialize cloud syncer if not provided
    if cloud_syncer is None and config.sync_after_each_batch:
        cloud_syncer = CloudSyncer.from_volume_path(config.cloud_destination)
    
    logger.info(f"Creating streaming pipeline:")
    logger.info(f"  Landing zone: {config.landing_zone}")
    logger.info(f"  Zarr store: {config.zarr_store_path}")
    logger.info(f"  Cloud destination: {config.cloud_destination}")
    logger.info(f"  Max files per batch: {config.max_files_per_batch}")
    logger.info(f"  Num workers: {config.num_workers}")
    
    def process_batch(batch_df, batch_id):
        """Process a micro-batch of GRIB files.
        
        This function:
        1. Collects file paths from the batch DataFrame
        2. Processes files in parallel using ThreadPoolExecutor
        3. Syncs to cloud storage after processing
        """
        batch_start = time.perf_counter()
        
        # Collect file paths to driver
        # AutoLoader provides 'path' column with binary file format
        file_paths = [row.path.replace("dbfs:", "") for row in batch_df.select('path').collect()]
        
        if not file_paths:
            logger.info(f"Batch {batch_id}: No files to process")
            return
        
        logger.info(f"Batch {batch_id}: Processing {len(file_paths)} files")
        
        # Process files in parallel
        process_start = time.perf_counter()
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(
                    write_grib_to_zarr_region,
                    file_path,
                    config.zarr_store_path,
                    hour_to_index,
                ): file_path
                for file_path in file_paths
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if not result.success:
                        errors.append(f"{file_path}: {result.error}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    errors.append(f"{file_path}: {str(e)}")
        
        processing_time = (time.perf_counter() - process_start) * 1000
        successful = sum(1 for r in results if r.success)
        
        logger.info(
            f"Batch {batch_id}: Processed {successful}/{len(file_paths)} files "
            f"in {processing_time:.1f}ms"
        )
        
        # Sync to cloud storage
        sync_time = 0
        if config.sync_after_each_batch and cloud_syncer and successful > 0:
            sync_start = time.perf_counter()
            sync_result = cloud_syncer.sync_zarr_chunks(config.zarr_store_path)
            sync_time = (time.perf_counter() - sync_start) * 1000
            
            if sync_result.success:
                logger.info(
                    f"Batch {batch_id}: Synced to cloud in {sync_time:.1f}ms"
                )
            else:
                logger.error(
                    f"Batch {batch_id}: Sync failed: {sync_result.error}"
                )
                errors.append(f"Sync failed: {sync_result.error}")
        
        total_time = (time.perf_counter() - batch_start) * 1000
        
        # Create batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            files_processed=len(file_paths),
            files_successful=successful,
            files_failed=len(file_paths) - successful,
            processing_time_ms=processing_time,
            sync_time_ms=sync_time,
            total_time_ms=total_time,
            errors=errors,
        )
        
        logger.info(
            f"Batch {batch_id} complete: "
            f"{batch_result.files_successful}/{batch_result.files_processed} files, "
            f"total time: {total_time:.1f}ms"
        )
        
        # Call completion callback if provided
        if on_batch_complete:
            try:
                on_batch_complete(batch_result)
            except Exception as e:
                logger.error(f"Batch callback error: {e}")
    
    # Configure AutoLoader
    reader_options = {
        'cloudFiles.format': 'binaryFile',
        'cloudFiles.maxFilesPerTrigger': str(config.max_files_per_batch),
        'cloudFiles.includeExistingFiles': 'true',
    }
    
    # Use file notification mode if enabled (recommended for low latency)
    if config.use_file_notification:
        # reader_options['cloudFiles.useNotifications'] = 'true'
        reader_options['cloudFiles.useManagedFileEvents'] = 'true'
    
    # Create the streaming DataFrame
    stream_df = (
        spark.readStream
        .format('cloudFiles')
        .options(**reader_options)
        .option('pathGlobFilter', '*.grib*')  # Match .grib and .grib2
        .load(config.landing_zone)
        .drop("content")
    )
    
    # Start the streaming query
    query = (
        stream_df.writeStream
        .foreachBatch(process_batch)
        .option('checkpointLocation', config.checkpoint_path)
        .trigger(processingTime=config.trigger_interval)
        .start()
    )
    
    logger.info(f"Streaming query started: {query.id}")
    
    return query


def run_batch_processing(
    spark,
    config: PipelineConfig,
    file_paths: list[str],
    hour_to_index: Optional[dict] = None,
) -> BatchResult:
    """Run a single batch of GRIB files (non-streaming mode).
    
    Useful for testing or one-time processing.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    config : PipelineConfig
        Pipeline configuration.
    file_paths : list[str]
        List of GRIB file paths to process.
    hour_to_index : dict, optional
        Mapping from forecast hour to step index.
    
    Returns
    -------
    BatchResult
        Result of batch processing.
    """
    from .zarr_init import generate_forecast_steps, build_hour_to_index_map
    from .region_writer import write_grib_to_zarr_region
    from .cloud_sync import CloudSyncer
    
    if hour_to_index is None:
        forecast_steps = generate_forecast_steps()
        hour_to_index = build_hour_to_index_map(forecast_steps)
    
    batch_start = time.perf_counter()
    
    # Process files in parallel
    process_start = time.perf_counter()
    results = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(
                write_grib_to_zarr_region,
                file_path,
                config.zarr_store_path,
                hour_to_index,
            ): file_path
            for file_path in file_paths
        }
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                if not result.success:
                    errors.append(f"{file_path}: {result.error}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                errors.append(f"{file_path}: {str(e)}")
    
    processing_time = (time.perf_counter() - process_start) * 1000
    successful = sum(1 for r in results if r.success)
    
    # Sync to cloud storage
    sync_time = 0
    if config.sync_after_each_batch and successful > 0:
        syncer = CloudSyncer.from_volume_path(config.cloud_destination)
        sync_start = time.perf_counter()
        sync_result = syncer.sync_zarr_chunks(config.zarr_store_path)
        sync_time = (time.perf_counter() - sync_start) * 1000
        
        if not sync_result.success:
            errors.append(f"Sync failed: {sync_result.error}")
    
    total_time = (time.perf_counter() - batch_start) * 1000
    
    return BatchResult(
        batch_id=0,
        files_processed=len(file_paths),
        files_successful=successful,
        files_failed=len(file_paths) - successful,
        processing_time_ms=processing_time,
        sync_time_ms=sync_time,
        total_time_ms=total_time,
        errors=errors,
    )


# Pipeline lifecycle management
class PipelineManager:
    """Manages the lifecycle of the streaming pipeline.
    
    Provides methods for starting, stopping, and monitoring the pipeline.
    
    Parameters
    ----------
    spark : SparkSession
        Active Spark session.
    config : PipelineConfig
        Pipeline configuration.
    """
    
    def __init__(self, spark, config: PipelineConfig):
        self.spark = spark
        self.config = config
        self.query = None
        self.batch_results = []
        self._hour_to_index = None
    
    def _on_batch_complete(self, result: BatchResult):
        """Callback for batch completion."""
        self.batch_results.append(result)
    
    def start(self) -> 'PipelineManager':
        """Start the streaming pipeline.
        
        Returns
        -------
        PipelineManager
            Self for method chaining.
        """
        from .zarr_init import generate_forecast_steps, build_hour_to_index_map
        
        if self._hour_to_index is None:
            forecast_steps = generate_forecast_steps()
            self._hour_to_index = build_hour_to_index_map(forecast_steps)
        
        self.query = create_streaming_pipeline(
            self.spark,
            self.config,
            hour_to_index=self._hour_to_index,
            on_batch_complete=self._on_batch_complete,
        )
        
        return self
    
    def stop(self) -> None:
        """Stop the streaming pipeline."""
        if self.query is not None:
            self.query.stop()
            logger.info("Pipeline stopped")
    
    def await_termination(self, timeout: Optional[float] = None) -> bool:
        """Wait for the pipeline to terminate.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds.
        
        Returns
        -------
        bool
            True if terminated, False if timeout.
        """
        if self.query is not None:
            return self.query.awaitTermination(timeout)
        return True
    
    @property
    def is_active(self) -> bool:
        """Check if the pipeline is running."""
        return self.query is not None and self.query.isActive
    
    @property
    def status(self) -> dict:
        """Get current pipeline status."""
        if self.query is None:
            return {'state': 'not_started'}
        
        return {
            'state': 'active' if self.query.isActive else 'stopped',
            'id': str(self.query.id),
            'batches_processed': len(self.batch_results),
            'recent_progress': self.query.recentProgress,
        }
    
    def get_metrics(self) -> dict:
        """Get aggregated pipeline metrics."""
        if not self.batch_results:
            return {}
        
        total_files = sum(r.files_processed for r in self.batch_results)
        total_successful = sum(r.files_successful for r in self.batch_results)
        total_processing_ms = sum(r.processing_time_ms for r in self.batch_results)
        total_sync_ms = sum(r.sync_time_ms for r in self.batch_results)
        
        return {
            'batches_processed': len(self.batch_results),
            'total_files_processed': total_files,
            'total_files_successful': total_successful,
            'total_files_failed': total_files - total_successful,
            'avg_processing_time_ms': total_processing_ms / len(self.batch_results),
            'avg_sync_time_ms': total_sync_ms / len(self.batch_results),
            'avg_files_per_batch': total_files / len(self.batch_results),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Streaming Pipeline Module")
    print("=" * 60)
    print()
    print("This module requires a Spark session to run.")
    print("See notebooks/03_streaming_pipeline.py for usage examples.")
    print()
    print("Example configuration:")
    print()
    
    config = PipelineConfig(
        landing_zone='/Volumes/catalog/schema/bronze/grib/',
        zarr_store_path='/local_disk0/forecast.zarr',
        cloud_destination='/Volumes/catalog/schema/silver/',
        checkpoint_path='/Volumes/catalog/schema/checkpoints/grib_pipeline',
        max_files_per_batch=32,
        num_workers=32,
    )
    
    print(f"  landing_zone: {config.landing_zone}")
    print(f"  zarr_store_path: {config.zarr_store_path}")
    print(f"  cloud_destination: {config.cloud_destination}")
    print(f"  checkpoint_path: {config.checkpoint_path}")
    print(f"  max_files_per_batch: {config.max_files_per_batch}")
    print(f"  num_workers: {config.num_workers}")

