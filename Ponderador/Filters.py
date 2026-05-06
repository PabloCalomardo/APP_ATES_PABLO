"""Raster smoothing filters for ponderador outputs.

The functions in this module operate on classified rasters and preserve the
original metadata while writing either a new raster or an overwritten one.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from scipy import ndimage as ndi
from skimage.morphology import closing, disk, opening

import importlib

# Optional dependency: shapely. Import dynamically to avoid hard dependency.
shapely_mapping = None
shapely_shape = None
if importlib.util.find_spec("shapely") is not None:
	shp = importlib.import_module("shapely.geometry")
	shapely_mapping = getattr(shp, "mapping", None)
	shapely_shape = getattr(shp, "shape", None)


def _output_path(raster_path: Path, method: str, overwrite: bool) -> Path:
	if overwrite:
		return raster_path
	return raster_path.with_name(f"{raster_path.stem}_{method}.tif")


def _read_raster(raster_path: Path) -> tuple[np.ndarray, dict, object]:
	with rasterio.open(raster_path) as src:
		array = src.read(1)
		profile = src.profile.copy()
		nodata = src.nodata
	return array, profile, nodata


def _write_raster(output_path: Path, array: np.ndarray, profile: dict) -> Path:
	profile = profile.copy()
	profile.update(dtype="int16", compress="deflate")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with rasterio.open(output_path, "w", **profile) as dst:
		dst.write(array.astype("int16", copy=False), 1)
	return output_path


def _valid_mask(array: np.ndarray, nodata_value: object) -> np.ndarray:
	if nodata_value is None:
		return np.isfinite(array)
	return array != nodata_value


def _mode_ignore_nodata(window: np.ndarray, nodata_value: object) -> int:
	valid = window[window != nodata_value] if nodata_value is not None else window[np.isfinite(window)]
	if valid.size == 0:
		return int(nodata_value) if nodata_value is not None else 0
	values, counts = np.unique(valid.astype(np.int64, copy=False), return_counts=True)
	return int(values[np.argmax(counts)])


def _neighbor_class_profile(
	array: np.ndarray,
	component_mask: np.ndarray,
	nodata_value: object,
	center_class: int,
) -> tuple[int | None, bool, bool]:
	"""Return the dominant neighbor class and whether lower/higher classes exist."""
	neighbors = ndi.binary_dilation(component_mask, structure=np.ones((3, 3), dtype=bool)) & (~component_mask)
	neighbor_values = array[neighbors]
	if nodata_value is not None:
		neighbor_values = neighbor_values[neighbor_values != nodata_value]
	if neighbor_values.size == 0:
		return None, False, False
	values, counts = np.unique(neighbor_values.astype(np.int64, copy=False), return_counts=True)
	majority_class = int(values[np.argmax(counts)])
	component_meaningful = neighbor_values.astype(np.int64, copy=False)
	has_lower = bool(np.any(component_meaningful < int(center_class)))
	has_higher = bool(np.any(component_meaningful > int(center_class)))
	return majority_class, has_lower, has_higher


def remove_small_class_islands(
	raster_path: Path,
	min_size: int = 15,
	overwrite: bool = False,
) -> Path:
	"""Replace 8-connected class islands smaller than min_size when the boundary is one-sided.

	If the island touches both lower and higher neighboring classes, it is preserved.
	"""
	if min_size < 1:
		raise ValueError("min_size must be >= 1")

	array, profile, nodata = _read_raster(raster_path)
	output_path = _output_path(raster_path, "islands", overwrite)
	valid = _valid_mask(array, nodata)
	if not valid.any():
		return _write_raster(output_path, array, profile)

	structure = np.ones((3, 3), dtype=bool)
	result = array.copy().astype(np.int32, copy=False)
	classes = np.unique(array[valid].astype(np.int64, copy=False))

	for class_value in classes:
		if nodata is not None and int(class_value) == int(nodata):
			continue
		mask = array == class_value
		labels, num_labels = ndi.label(mask, structure=structure)
		for label_id in range(1, num_labels + 1):
			component_mask = labels == label_id
			component_size = int(component_mask.sum())
			if component_size >= min_size:
				continue
			majority_class, has_lower, has_higher = _neighbor_class_profile(array, component_mask, nodata, int(class_value))
			if majority_class is None:
				continue
			if has_lower and has_higher:
				continue
			replacement = majority_class
			if replacement == int(class_value):
				continue
			result[component_mask] = int(replacement)

	if nodata is not None:
		result[~valid] = int(nodata)
	return _write_raster(output_path, result, profile)


def modal_filter_3x3(raster_path: Path, overwrite: bool = False) -> Path:
	"""Apply a 3x3 focal mode filter to the raster."""
	array, profile, nodata = _read_raster(raster_path)
	output_path = _output_path(raster_path, "smoothed", overwrite)
	valid = _valid_mask(array, nodata)
	if not valid.any():
		return _write_raster(output_path, array, profile)

	filled = array.copy().astype(np.int32, copy=False)
	if nodata is not None:
		filled[~valid] = int(nodata)

	filtered = ndi.generic_filter(
		filled,
		function=lambda values: _mode_ignore_nodata(values, nodata),
		size=3,
		mode="nearest",
		output=np.int32,
	)
	if nodata is not None:
		filtered[~valid] = int(nodata)
	return _write_raster(output_path, filtered, profile)


def morphological_per_class_filter(
	raster_path: Path,
	radius: int = 1,
	iterations: int = 1,
	overwrite: bool = False,
) -> Path:
	"""Smooth class boundaries using per-class binary opening + closing."""
	if radius < 1:
		raise ValueError("radius must be >= 1")
	if iterations < 1:
		raise ValueError("iterations must be >= 1")

	array, profile, nodata = _read_raster(raster_path)
	output_path = _output_path(raster_path, "smoothed", overwrite)
	valid = _valid_mask(array, nodata)
	if not valid.any():
		return _write_raster(output_path, array, profile)

	classes = np.unique(array[valid].astype(np.int64, copy=False))
	result = np.full_like(array, fill_value=int(nodata) if nodata is not None else 0, dtype=np.int32)
	footprint = disk(radius)

	for class_value in classes:
		if nodata is not None and int(class_value) == int(nodata):
			continue
		mask = array == class_value
		refined = mask.copy()
		for _ in range(iterations):
			refined = opening(refined, footprint=footprint)
			refined = closing(refined, footprint=footprint)
		result[refined] = int(class_value)

	if nodata is not None:
		result[~valid] = int(nodata)
	return _write_raster(output_path, result, profile)


def _chaikin_coords(coords: Sequence[Sequence[float]], iterations: int) -> List[Tuple[float, float]]:
	points = [(float(x), float(y)) for x, y in coords]
	if len(points) < 4 or iterations < 1:
		return points
	if points[0] != points[-1]:
		points = points + [points[0]]

	for _ in range(iterations):
		if len(points) < 4:
			break
		new_points: List[Tuple[float, float]] = [points[0]]
		for idx in range(len(points) - 1):
			x1, y1 = points[idx]
			x2, y2 = points[idx + 1]
			q = (0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2)
			r = (0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2)
			new_points.extend([q, r])
		new_points.append(new_points[0])
		points = new_points

	if points[0] != points[-1]:
		points.append(points[0])
	return points


def _smooth_geometry(geometry: dict, chaikin_iterations: int) -> dict:
	geom_type = geometry.get("type")
	coordinates = geometry.get("coordinates")
	if geom_type == "Polygon":
		return {
			"type": "Polygon",
			"coordinates": [
				_chaikin_coords(ring, chaikin_iterations)
				for ring in coordinates
			],
		}
	if geom_type == "MultiPolygon":
		return {
			"type": "MultiPolygon",
			"coordinates": [
				[
					_chaikin_coords(ring, chaikin_iterations)
					for ring in polygon
				]
				for polygon in coordinates
			],
		}
	return geometry


def vectorize_smooth_rasterize_filter(
	raster_path: Path,
	chaikin_iterations: int = 2,
	simplify_tolerance: float = 0.0,
	overwrite: bool = False,
) -> Path:
	"""Vectorize class regions, smooth their geometry, and rasterize back."""
	if chaikin_iterations < 0:
		raise ValueError("chaikin_iterations must be >= 0")
	if simplify_tolerance < 0:
		raise ValueError("simplify_tolerance must be >= 0")

	array, profile, nodata = _read_raster(raster_path)
	output_path = _output_path(raster_path, "smoothed", overwrite)
	valid = _valid_mask(array, nodata)
	if not valid.any():
		return _write_raster(output_path, array, profile)

	shape_geometries: List[tuple[dict, int]] = []
	for geometry, value in shapes(array.astype(np.int32, copy=False), mask=valid, transform=profile["transform"]):
		class_value = int(value)
		if nodata is not None and class_value == int(nodata):
			continue
		if simplify_tolerance > 0 and shapely_shape is not None and shapely_mapping is not None:
			shaped = shapely_shape(geometry).simplify(simplify_tolerance, preserve_topology=True)
			geometry = shapely_mapping(shaped)
		geometry = _smooth_geometry(geometry, chaikin_iterations)
		shape_geometries.append((geometry, class_value))

	if not shape_geometries:
		return _write_raster(output_path, array, profile)

	result = rasterize(
		shape_geometries,
		out_shape=array.shape,
		fill=int(nodata) if nodata is not None else 0,
		transform=profile["transform"],
		dtype="int32",
	)
	if nodata is not None:
		result[~valid] = int(nodata)
	return _write_raster(output_path, result.astype(np.int32, copy=False), profile)


def apply_filter(
	raster_path: Path,
	method: str,
	overwrite: bool = False,
	class_island_min_size: int = 15,
	**kwargs,
) -> Path:
	"""Dispatch the requested smoothing method."""
	method_norm = str(method).strip().lower()
	if method_norm in {"none", "off", "false"}:
		return raster_path
	if class_island_min_size > 0:
		raster_path = remove_small_class_islands(
			raster_path,
			min_size=class_island_min_size,
			overwrite=overwrite,
		)
	if method_norm in {"modal", "mode", "focal"}:
		result = modal_filter_3x3(raster_path, overwrite=overwrite)
	elif method_norm in {"morph", "morphological"}:
		result = morphological_per_class_filter(
			raster_path,
			radius=int(kwargs.get("radius", 1)),
			iterations=int(kwargs.get("iterations", 1)),
			overwrite=overwrite,
		)
	elif method_norm in {"vectorize", "vectorise"}:
		result = vectorize_smooth_rasterize_filter(
			raster_path,
			chaikin_iterations=int(kwargs.get("chaikin_iterations", 2)),
			simplify_tolerance=float(kwargs.get("simplify_tolerance", 0.0)),
			overwrite=overwrite,
		)
	else:
		raise ValueError(f"Unsupported smoothing method: {method}")
	if class_island_min_size > 0:
		result = remove_small_class_islands(
			result,
			min_size=class_island_min_size,
			overwrite=overwrite,
		)
		return result
	return result


def _build_arg_parser():
	import argparse

	parser = argparse.ArgumentParser(
		description="Apply a smoothing filter directly to an ATES raster."
	)
	parser.add_argument("input_raster", help="Path to the input ATES GeoTIFF")
	parser.add_argument(
		"--method",
		required=False,
		choices=["modal", "morph", "vectorize"],
		default="modal",
		help="Smoothing method to apply (default: modal; omit when using --testing)",
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Optional output path. If omitted, writes <input>_smoothed.tif",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite the input raster instead of creating a new file",
	)
	parser.add_argument(
		"--radius",
		type=int,
		default=1,
		help="Radius for the morphological filter (default: 1)",
	)
	parser.add_argument(
		"--iterations",
		type=int,
		default=1,
		help="Iterations for the morphological filter (default: 1)",
	)
	parser.add_argument(
		"--simplify-tolerance",
		type=float,
		default=0.0,
		help="Optional simplify tolerance for the vectorize filter",
	)
	parser.add_argument(
		"--chaikin-iterations",
		type=int,
		default=2,
		help="Chaikin iterations for the vectorize filter",
	)
	parser.add_argument(
		"--class-island-min-size",
		type=int,
		default=15,
		help="Minimum 8-connected island size to keep; smaller components are replaced by neighboring majority classes",
	)
	parser.add_argument(
		"--testing",
		action="store_true",
		help="Run all smoothing methods (modal,morph,vectorize) and write separate outputs for comparison",
	)
	return parser


def main() -> int:
	parser = _build_arg_parser()
	args = parser.parse_args()
	raster_path = Path(args.input_raster)
	if not raster_path.exists():
		parser.error(f"Input raster not found: {raster_path}")
	if args.radius < 1:
		parser.error("--radius must be >= 1")
	if args.iterations < 1:
		parser.error("--iterations must be >= 1")
	if args.simplify_tolerance < 0.0:
		parser.error("--simplify-tolerance must be >= 0")
	if args.chaikin_iterations < 0:
		parser.error("--chaikin-iterations must be >= 0")
	if args.class_island_min_size < 1:
		parser.error("--class-island-min-size must be >= 1")

	if args.testing:
		methods = ["modal", "morph", "vectorize"]
		produced: list[Path] = []
		for method in methods:
			res = apply_filter(
				raster_path=raster_path,
				method=method,
				overwrite=args.overwrite,
				class_island_min_size=args.class_island_min_size,
				radius=args.radius,
				iterations=args.iterations,
				simplify_tolerance=args.simplify_tolerance,
				chaikin_iterations=args.chaikin_iterations,
			)
			# compute distinct output path for this method
			if args.output:
				outp = Path(args.output)
				out_dir = outp.parent
				base = outp.stem
				ext = outp.suffix or ".tif"
				dest = out_dir / f"{base}_{method}{ext}"
			else:
				dest = raster_path.with_name(f"{raster_path.stem}_{method}.tif")

			# move/rename resulting file to dest (handle overwrite)
			try:
				if dest.exists():
					dest.unlink()
				if res != dest:
					res.replace(dest)
			except Exception:
				# fallback: copy bytes
				dest.write_bytes(res.read_bytes())
			produced.append(dest)

		for p in produced:
			print(p)
		return 0

	# single method mode
	result = apply_filter(
		raster_path=raster_path,
		method=args.method,
		overwrite=args.overwrite,
		class_island_min_size=args.class_island_min_size,
		radius=args.radius,
		iterations=args.iterations,
		simplify_tolerance=args.simplify_tolerance,
		chaikin_iterations=args.chaikin_iterations,
	)
	if args.output is not None:
		output_path = Path(args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		if output_path != result:
			output_path.write_bytes(result.read_bytes())
			result = output_path
	print(result)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())