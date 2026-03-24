#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def fill_dem_simple(in_dem: str | Path, out_dem: str | Path) -> Path:
	"""Fill DEM nodata with a simple inpainting.

	Uses `rasterio.fill.fillnodata` when nodata pixels exist.
	If nodata is missing or fill is unavailable, the DEM is copied as-is.

	Returns the output path.
	"""

	in_path = Path(in_dem).expanduser().resolve()
	out_path = Path(out_dem).expanduser().resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)

	import rasterio

	with rasterio.open(in_path) as src:
		profile = src.profile.copy()
		band1 = src.read(1, masked=True)
		nodata = src.nodata

	# If there is no explicit nodata or no missing pixels, just copy.
	if nodata is None:
		with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
			dst.write(src.read())
		return out_path

	mask = getattr(band1, "mask", False)
	if mask is False:
		with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
			dst.write(src.read())
		return out_path
	try:
		if hasattr(mask, "any") and not mask.any():
			with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
				dst.write(src.read())
			return out_path
	except Exception:
		# If mask behavior is unexpected, do a safe copy.
		with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
			dst.write(src.read())
		return out_path

	# We have nodata; attempt fill.
	arr = band1.filled(nodata).astype("float32")
	valid_mask = ~band1.mask  # True where valid

	try:
		from rasterio.fill import fillnodata

		filled = fillnodata(
			arr,
			mask=valid_mask,
			max_search_distance=100.0,
			smoothing_iterations=0,
		)
		out_arr = filled.astype("float32")
	except Exception:
		out_arr = band1.filled(nodata)

	profile.update(count=1, compress="deflate", dtype=str(out_arr.dtype), nodata=nodata)
	with rasterio.open(out_path, "w", **profile) as dst:
		dst.write(out_arr, 1)

	return out_path

