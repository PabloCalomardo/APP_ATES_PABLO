#!/usr/bin/env python3
"""Post-process Flow-Py bitmasks to one GeoJSON with full overlap information.

Reads all source_ids_bitmask.tif files under outputs/Flow-Py/*/res_* and writes
one single GeoJSON containing:

1) avalanches: one (multi)polygon feature per avalanche id and run
2) overlap_zones: polygons of exact overlap combinations (bitmask zones)

This preserves all overlap information in one artifact without extra dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.features import shapes


def _find_flowpy_result_dirs(flowpy_root: Path) -> List[Path]:
	"""Return all result folders matching outputs/Flow-Py/*/res_*."""
	result_dirs: List[Path] = []
	if not flowpy_root.exists():
		return result_dirs

	for basin_dir in sorted([p for p in flowpy_root.iterdir() if p.is_dir()]):
		for res_dir in sorted([p for p in basin_dir.iterdir() if p.is_dir() and p.name.startswith("res_")]):
			result_dirs.append(res_dir)
	return result_dirs


def _decode_avalanche_ids(mask_value: int, max_bits: int = 64) -> List[int]:
	ids: List[int] = []
	value_u64 = np.uint64(mask_value)
	for bit_idx in range(max_bits):
		bit = np.uint64(1) << np.uint64(bit_idx)
		if (value_u64 & bit) != 0:
			ids.append(bit_idx + 1)
	return ids


def _run_context(res_dir: Path) -> Dict[str, str]:
	basin_dir = res_dir.parent.name
	run_id = res_dir.name
	return {
		"basin_id": basin_dir,
		"run_id": run_id,
	}


def _write_geojson(flowpy_root: Path, output_geojson: Path) -> None:
	output_geojson.parent.mkdir(parents=True, exist_ok=True)

	result_dirs = _find_flowpy_result_dirs(flowpy_root)
	if not result_dirs:
		raise RuntimeError(f"No Flow-Py result folders found under: {flowpy_root}")

	# Use first available CRS as reference metadata for the GeoJSON.
	first_bitmask = result_dirs[0] / "source_ids_bitmask.tif"
	if not first_bitmask.exists():
		raise RuntimeError(f"Missing source_ids_bitmask.tif in: {result_dirs[0]}")
	with rasterio.open(first_bitmask) as src_ref:
		crs_wkt = None if src_ref.crs is None else src_ref.crs.to_wkt()

	features: List[Dict] = []

	for res_dir in result_dirs:
		bitmask_path = res_dir / "source_ids_bitmask.tif"
		if not bitmask_path.exists():
			print(f"[skip] No source_ids_bitmask.tif in {res_dir}")
			continue

		with rasterio.open(bitmask_path) as src:
			bitmask = src.read(1).astype(np.uint64, copy=False)
			transform = src.transform

		ctx = _run_context(res_dir)

		# Layer-like feature group 1: avalanche polygons.
		n_avalanches = 0
		for avalanche_id in range(1, 65):
			bit = np.uint64(1) << np.uint64(avalanche_id - 1)
			mask = (bitmask & bit) > 0
			if not np.any(mask):
				continue
			n_avalanches += 1

			for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
				if int(value) != 1:
					continue
				features.append(
					{
						"type": "Feature",
						"geometry": geom,
						"properties": {
							"feature_type": "avalanche",
							"basin_id": ctx["basin_id"],
							"run_id": ctx["run_id"],
							"avalanche_id": avalanche_id,
						},
					}
				)

		# Layer-like feature group 2: exact overlap zones by bitmask value (>0).
		unique_values = np.unique(bitmask)
		unique_values = [int(v) for v in unique_values.tolist() if int(v) > 0]
		for value in unique_values:
			zone_mask = bitmask == np.uint64(value)
			if not np.any(zone_mask):
				continue
			av_ids = _decode_avalanche_ids(value)
			av_ids_str = ",".join(str(v) for v in av_ids)

			for geom, geom_val in shapes(zone_mask.astype(np.uint8), mask=zone_mask, transform=transform):
				if int(geom_val) != 1:
					continue
				features.append(
					{
						"type": "Feature",
						"geometry": geom,
						"properties": {
							"feature_type": "overlap_zone",
							"basin_id": ctx["basin_id"],
							"run_id": ctx["run_id"],
							"bitmask": str(value),
							"n_aval": len(av_ids),
							"aval_ids": av_ids_str,
						},
					}
				)

		print(f"[ok] {res_dir}: {n_avalanches} allaus processades")

	collection = {
		"type": "FeatureCollection",
		"name": "avalanche_shapes",
		"crs_wkt": crs_wkt,
		"features": features,
	}
	output_geojson.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export Flow-Py avalanche polygons and overlap zones to one GeoJSON"
	)
	parser.add_argument(
		"--flowpy-root",
		default="outputs/Flow-Py",
		help="Root folder containing Flow-Py basin folders (default: outputs/Flow-Py)",
	)
	parser.add_argument(
		"--output-geojson",
		default="outputs/Avalanche_Shapes/avalanche_shapes.geojson",
		help="Single GeoJSON output path (default: outputs/Avalanche_Shapes/avalanche_shapes.geojson)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	app_root = Path(__file__).resolve().parents[1]
	flowpy_root = Path(args.flowpy_root).expanduser()
	if not flowpy_root.is_absolute():
		flowpy_root = (app_root / flowpy_root).resolve()

	output_geojson = Path(args.output_geojson).expanduser()
	if not output_geojson.is_absolute():
		output_geojson = (app_root / output_geojson).resolve()

	print(f"Found {len(_find_flowpy_result_dirs(flowpy_root))} Flow-Py result folders")
	_write_geojson(flowpy_root=flowpy_root, output_geojson=output_geojson)
	print(f"Done. Output: {output_geojson}")


if __name__ == "__main__":
	main()

