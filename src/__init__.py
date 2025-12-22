"""
Low-latency weather data pipeline for Databricks.

This package provides tools for:
- Pre-allocated Zarr store initialization
- GRIB to Zarr region writes
- Cloud sync with SAS token management
- Streaming ingestion pipeline
- Comprehensive benchmarking
"""

from .zarr_init import (
    generate_forecast_steps,
    build_hour_to_index_map,
    initialize_zarr_store,
    get_zarr_store_info,
    ForecastStepConfig,
)

from .region_writer import (
    write_grib_to_zarr_region,
    write_grib_batch_parallel,
    extract_grib_metadata,
    read_grib_data,
    stage_file_locally,
    stage_files_batch,
    stage_files_with_azcopy,
    stage_files_with_azure_sdk,
    download_single_file_azure_sdk,
    download_single_file_azcopy,
    cleanup_staging_dir,
    GribMetadata,
    WriteResult,
    DEFAULT_STAGING_DIR,
)

from .cloud_sync import (
    TokenManager,
    CloudSyncer,
    sync_with_azcopy,
    find_azcopy,
    SyncResult,
)

from .streaming_pipeline import (
    PipelineConfig,
    PipelineManager,
    create_streaming_pipeline,
    run_batch_processing,
    BatchResult,
)

__all__ = [
    # Zarr initialization
    "generate_forecast_steps",
    "build_hour_to_index_map", 
    "initialize_zarr_store",
    "get_zarr_store_info",
    "ForecastStepConfig",
    # Region writer
    "write_grib_to_zarr_region",
    "write_grib_batch_parallel",
    "extract_grib_metadata",
    "read_grib_data",
    "stage_file_locally",
    "stage_files_batch",
    "stage_files_with_azcopy",
    "stage_files_with_azure_sdk",
    "download_single_file_azure_sdk",
    "download_single_file_azcopy",
    "cleanup_staging_dir",
    "GribMetadata",
    "WriteResult",
    "DEFAULT_STAGING_DIR",
    # Cloud sync
    "TokenManager",
    "CloudSyncer",
    "sync_with_azcopy",
    "find_azcopy",
    "SyncResult",
    # Streaming pipeline
    "PipelineConfig",
    "PipelineManager",
    "create_streaming_pipeline",
    "run_batch_processing",
    "BatchResult",
]

