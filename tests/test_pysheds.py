import os
import numpy as np
from pysheds.grid import Grid

def test_flowdir_shape_matches_dem():
    test_dem_path = "tests/data/sample_dem.tif"
    if not os.path.exists(test_dem_path):
        import rasterio
        from rasterio.transform import from_origin
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        transform = from_origin(0, 3, 1, 1)
        with rasterio.open(
            test_dem_path, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs='+proj=latlong',
            transform=transform
        ) as dst:
            dst.write(data, 1)

    grid = Grid.from_raster(test_dem_path)
    dem = grid.read_raster(test_dem_path)

    pit_filled = grid.fill_pits(dem)
    depression_filled = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(depression_filled)

    fdir = grid.flowdir(inflated)
    assert fdir.shape == dem.shape, "Flow direction grid shape mismatch"
