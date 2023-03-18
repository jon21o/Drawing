import numpy as np
import xarray as xr

matrix = xr.DataArray(
    np.ones((3, 5)),
    coords={"dim": ["x", "y", "z"], "pt": ["center", "pt1", "pt2", "pt3", "pt4"]},
    dims=["dim", "pt"],
)

print(matrix)

matrix.loc[dict(dim="x", pt="pt1")] = 5

print(matrix.sizes["dim"])
