#!/usr/bin/env python3
"""
Generate mock NetCDF files based on a template from Copernicus Climate Data Store.
Creates 100 files with 50 ensemble members each, filled with random noise.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import uuid

def generate_mock_file(template_path, output_path, num_ensemble_members=50):
    """
    Generate a mock NetCDF file based on a template.
    
    Parameters:
    -----------
    template_path : str or Path
        Path to the template NetCDF file
    output_path : str or Path
        Path where the mock file will be saved
    num_ensemble_members : int
        Number of ensemble members (default: 50)
    """
    # Open the template file
    template_ds = xr.open_dataset(template_path)
    
    # Get dimensions from template
    n_valid_time = template_ds.dims['valid_time']
    n_latitude = template_ds.dims['latitude']
    n_longitude = template_ds.dims['longitude']
    
    # Create new dimensions
    new_number = np.arange(num_ensemble_members)
    
    # Generate random temperature data
    # Using realistic temperature range: 220K to 320K (approximately -53°C to 47°C)
    random_t2m = np.random.uniform(220, 320, 
                                   size=(num_ensemble_members, n_valid_time, 
                                        n_latitude, n_longitude)).astype(np.float32)
    
    # Create new dataset with modified dimensions
    new_ds = xr.Dataset(
        data_vars={
            't2m': (
                ['number', 'valid_time', 'latitude', 'longitude'],
                random_t2m,
                template_ds['t2m'].attrs.copy()
            ),
            'expver': template_ds['expver'] if 'expver' in template_ds else None
        },
        coords={
            'number': (
                ['number'],
                new_number,
                template_ds['number'].attrs.copy()
            ),
            'valid_time': template_ds['valid_time'],
            'latitude': template_ds['latitude'],
            'longitude': template_ds['longitude']
        },
        attrs=template_ds.attrs.copy()
    )
    
    # Update the GRIB_totalNumber attribute to reflect new ensemble size
    if 'GRIB_totalNumber' in new_ds['t2m'].attrs:
        new_ds['t2m'].attrs['GRIB_totalNumber'] = num_ensemble_members
    
    # Save to NetCDF file
    new_ds.to_netcdf(output_path, engine='netcdf4')
    
    # Close datasets
    template_ds.close()
    new_ds.close()
    
    return output_path


def main():
    """Main function to generate 100 mock files."""
    # Paths
    template_file = Path('/Users/stuart.lynn/Customers/LSEG/raster-benchmarking/data/example/652f73a7818c431a469c7ed3e9054e0a.nc')
    output_dir = Path('/Users/stuart.lynn/Customers/LSEG/raster-benchmarking/data')
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating 100 mock NetCDF files with 50 ensemble members each...")
    print(f"Template: {template_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate 100 mock files
    for i in range(100):
        # Generate unique filename using UUID (similar to the template)
        filename = f"{uuid.uuid4().hex}.nc"
        output_path = output_dir / filename
        
        # Generate the mock file
        generate_mock_file(template_file, output_path, num_ensemble_members=50)
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/100 files...")
    
    print()
    print(f"✓ Successfully generated 100 mock files in {output_dir}")
    print(f"  Each file contains 50 ensemble members with random temperature data.")


if __name__ == '__main__':
    main()

