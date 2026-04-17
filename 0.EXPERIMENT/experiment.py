from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


CLASSES_OF_INTEREST = (2, 3, 4)


@dataclass
class DatasetSpec:
	key: str
	dem_path: Path
	forest_path: Path
	validated_path: Path


@dataclass
class ParamSweep:
	name: str
	values: list[Any]


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run parameter experiments for APP_ATES_PABLO and compare output ATES "
			"rasters against validated references using classes 2/3/4."
		)
	)
	parser.add_argument(
		"--mode",
		choices=("quick", "extensive", "ultrafast"),
		default="quick",
		help=(
			"quick: key parameters, extensive: all parameters, "
			"ultrafast: precompute up to Flow-Py once and sweep only post-Flow-Py params"
		),
	)
	parser.add_argument(
		"--dems",
		default="all",
		help=(
			"Dataset keys or DEM filenames separated by commas (example: BOWSUMMIT,CONNAUGHT). "
			"Use 'all' to run all datasets detected in inputs/."
		),
	)
	parser.add_argument(
		"--inputs-dir",
		default="inputs",
		help="Directory containing DEM_* and FOREST_* rasters",
	)
	parser.add_argument(
		"--validated-dir",
		default="0.EXPERIMENT",
		help="Directory containing ATES_COMPROVAT_*.tif references",
	)
	parser.add_argument(
		"--main-script",
		default="main.py",
		help="Path to pipeline entrypoint script",
	)
	parser.add_argument(
		"--runs-root",
		default="outputs/experiments",
		help="Base output directory for experiment runs",
	)
	parser.add_argument(
		"--values-per-param",
		type=int,
		default=None,
		help="Optional cap for number of values per parameter (must be >= 3 when used)",
	)
	parser.add_argument(
		"--limit-params",
		type=int,
		default=None,
		help="Optional cap for number of parameters to run (useful for test runs)",
	)
	parser.add_argument(
		"--stop-on-error",
		action="store_true",
		help="Stop the experiment immediately if one run fails",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Do not execute main.py, only generate experiment plan and commands",
	)
	parser.add_argument(
		"--python-exe",
		default=sys.executable,
		help="Python executable used to run main.py",
	)
	parser.add_argument(
		"--rebuild-report-dir",
		default=None,
		help=(
			"Optional: existing experiment directory (with summary.json) to rebuild comparative "
			"tables and analytics report without running experiments again."
		),
	)
	return parser.parse_args()


def _to_key_from_dem_stem(stem: str) -> str:
	name = stem.upper()
	if name.startswith("DEM_ATES_"):
		suffix = name[len("DEM_ATES_") :]
	elif name.startswith("DEM_"):
		suffix = name[len("DEM_") :]
	else:
		suffix = name
	return suffix.replace("_", "")


def _forest_candidates_for_dem(dem_name: str) -> list[str]:
	candidates: list[str] = []
	upper = dem_name.upper()
	if upper.startswith("DEM_ATES_"):
		suffix = dem_name[len("DEM_ATES_") :]
		candidates.append(f"FOREST_ATES_{suffix}")
		candidates.append(f"FOREST_{suffix}")
	if upper.startswith("DEM_"):
		suffix = dem_name[len("DEM_") :]
		candidates.append(f"FOREST_{suffix}")
		candidates.append(f"FOREST_ATES_{suffix}")
	candidates.append(dem_name.replace("DEM_", "FOREST_", 1))
	candidates.append(dem_name.replace("DEM_ATES_", "FOREST_ATES_", 1))

	# Keep order and uniqueness
	out: list[str] = []
	seen: set[str] = set()
	for c in candidates:
		if c not in seen:
			out.append(c)
			seen.add(c)
	return out


def discover_datasets(inputs_dir: Path, validated_dir: Path) -> dict[str, DatasetSpec]:
	datasets: dict[str, DatasetSpec] = {}
	for dem_path in sorted(inputs_dir.glob("DEM*.tif")):
		dem_stem = dem_path.stem
		key = _to_key_from_dem_stem(dem_stem)

		forest_path = None
		for forest_stem in _forest_candidates_for_dem(dem_stem):
			candidate = inputs_dir / f"{forest_stem}.tif"
			if candidate.exists():
				forest_path = candidate
				break
		if forest_path is None:
			continue

		validated_candidates = [
			validated_dir / f"ATES_COMPROVAT_{key}.tif",
			validated_dir / f"ATES_COMPROVAT_{dem_stem.replace('DEM_ATES_', '').replace('DEM_', '').replace('_', '')}.tif",
		]
		validated_path = next((p for p in validated_candidates if p.exists()), None)
		if validated_path is None:
			continue

		datasets[key] = DatasetSpec(
			key=key,
			dem_path=dem_path,
			forest_path=forest_path,
			validated_path=validated_path,
		)
	return datasets


def _quick_param_sweeps() -> list[ParamSweep]:
	return [
		ParamSweep("radius", [4, 6, 8]),
		ParamSweep("pra-thd", [0.1, 0.15, 0.2]),
		ParamSweep("divisor-stream-threshold", [500, 850, 1200]),
		ParamSweep("watershed-threshold", [8000, 12000, 18000]),
		ParamSweep("flowpy-alpha", [18, 22, 26]),
		ParamSweep("flowpy-flux", [0.001, 0.003, 0.006]),
		ParamSweep("overhead-cellcount-weight", [0.1, 0.5, 2.0]),
		ParamSweep("ates-forest-adjustment", ["legacy", "paper_pra", "paper_runout"]),
		ParamSweep("terrain-energy-trauma-threshold", [0.25, 0.35, 0.5]),
		ParamSweep("zones-start-threshold", [0.95, 0.99, 0.999]),
		ParamSweep("zones-ending-threshold", [0.05, 0.075, 0.1]),
		ParamSweep("runout-min-evidence-threshold", [0.02, 0.03, 0.05]),
	]


def _extensive_param_sweeps() -> list[ParamSweep]:
	return [
		ParamSweep("forest-type", ["stems", "bav", "pcc", "sen2cc"]),
		ParamSweep("radius", [4, 5, 6, 7, 8]),
		ParamSweep("prob", [0.4, 0.5, 0.6, 0.7, 0.8]),
		ParamSweep("winddir", [0, 45, 90, 180, 270]),
		ParamSweep("windtol", [45, 90, 135, 180, 225]),
		ParamSweep("pra-thd", [0.08, 0.12, 0.15, 0.18, 0.22]),
		ParamSweep("sf", [1, 2, 3, 4, 5]),
		ParamSweep("divisor-stream-threshold", [400, 650, 850, 1100, 1400]),
		ParamSweep("divisor-channel-init-exponent", [0.0, 0.3, 0.6, 0.9, 1.2]),
		ParamSweep("divisor-channel-min-slope", [0.002, 0.004, 0.005, 0.007, 0.01]),
		ParamSweep("watershed-threshold", [6000, 9000, 12000, 16000, 22000]),
		ParamSweep("watershed-memory", [300, 500, 800, 1200, 2000]),
		ParamSweep("flowpy-alpha", [16, 19, 22, 25, 28]),
		ParamSweep("flowpy-exponent", [6, 7, 8, 9, 10]),
		ParamSweep("flowpy-flux", [0.001, 0.002, 0.003, 0.005, 0.008]),
		ParamSweep("flowpy-max-z", [2000, 4000, 8000, 12000, 16000]),
		ParamSweep("overhead-cellcount-weight", [0.0, 0.25, 0.5, 0.75, 2.0]),
		ParamSweep("ates-forest-window", [3, 5, 7, 9, 11]),
		ParamSweep("ates-slope-sigma", [0.5, 0.8, 1.0, 1.5, 2.0]),
		ParamSweep("ates-forest-adjustment", ["legacy", "conservative", "paper_pra", "paper_runout"]),
		ParamSweep("landform-windows", ["5,10,15,20,25,30", "5,6,7,8,9,10", ",".join(str(v) for v in range(5, 31))]),
		ParamSweep("landform-curvature-threshold", [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]),
		ParamSweep("landform-flat-gradient-eps", [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]),
		ParamSweep("terrain-forest-tree-threshold", [20.0, 30.0, 35.0, 45.0, 55.0]),
		ParamSweep("terrain-energy-trauma-threshold", [0.2, 0.3, 0.35, 0.45, 0.6]),
		ParamSweep("terrain-gully-energy-threshold", [0.12, 0.18, 0.22, 0.3, 0.4]),
		ParamSweep("terrain-gully-spi-m", [0.6, 0.8, 1.0, 1.2, 1.4]),
		ParamSweep("terrain-gully-spi-n", [0.6, 0.8, 1.0, 1.2, 1.4]),
		ParamSweep("terrain-gully-spi-threshold", [0.0, 0.1, 0.2, 0.3, 0.5]),
		ParamSweep("terrain-gully-spi-percentile", [80.0, 85.0, 88.0, 92.0, 96.0]),
		ParamSweep("terrain-gully-min-drainage-area-m2", [2000.0, 3000.0, 4000.0, 6000.0, 8000.0]),
		ParamSweep("terrain-gully-min-slope-deg", [8.0, 10.0, 13.0, 16.0, 20.0]),
		ParamSweep("terrain-gully-max-slope-deg", [40.0, 45.0, 48.0, 52.0, 58.0]),
		ParamSweep("terrain-lake-max-slope-deg", [3.0, 4.5, 6.0, 8.0, 10.0]),
		ParamSweep("terrain-lake-tpi-threshold", [-3.0, -2.2, -1.8, -1.4, -1.0]),
		ParamSweep("terrain-lake-max-spi-threshold", [0.0, 0.05, 0.1, 0.2, 0.3]),
		ParamSweep("terrain-lake-max-spi-percentile", [20.0, 30.0, 35.0, 45.0, 55.0]),
		ParamSweep("zones-start-threshold", [0.92, 0.96, 0.99, 0.995, 0.999]),
		ParamSweep("zones-ending-threshold", [0.03, 0.05, 0.075, 0.1, 0.15]),
		ParamSweep("runout-flux-min-threshold", [0.005, 0.01, 0.02, 0.03, 0.05]),
		ParamSweep("runout-min-evidence-threshold", [0.01, 0.02, 0.03, 0.05, 0.08]),
		ParamSweep("ponderador-forest-type", ["stems", "bav", "pcc", "sen2cc"]),
	]


def _ultrafast_param_sweeps() -> list[ParamSweep]:
	# UltraFast avoids parameters that require recomputing steps 1..6.
	# This sweep is intentionally aggressive: 5 values per parameter (where possible)
	# with exaggerated extremes to stress post-FlowPy sensitivity.
	return [
		ParamSweep("ates-forest-window", [1, 3, 5, 9, 15]),
		ParamSweep("ates-slope-sigma", [0.1, 0.5, 1.0, 2.5, 5.0]),
		ParamSweep("ates-forest-adjustment", ["legacy", "conservative", "paper_pra", "paper_runout"]),
		ParamSweep("landform-windows", ["3,5,7", "5,10,15,20,25,30", "3,4,5,6,7,8,9", "10,15,20,25,30,35,40", "2,3,4,5,6,7,8,9,10,11,12"]),
		ParamSweep("landform-curvature-threshold", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
		ParamSweep("landform-flat-gradient-eps", [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]),
		ParamSweep("terrain-forest-tree-threshold", [5.0, 20.0, 35.0, 60.0, 90.0]),
		ParamSweep("terrain-energy-trauma-threshold", [0.05, 0.15, 0.35, 0.7, 1.2]),
		ParamSweep("terrain-gully-energy-threshold", [0.01, 0.08, 0.22, 0.5, 1.0]),
		ParamSweep("terrain-gully-spi-m", [0.2, 0.5, 1.0, 1.8, 3.0]),
		ParamSweep("terrain-gully-spi-n", [0.2, 0.5, 1.0, 1.8, 3.0]),
		ParamSweep("terrain-gully-spi-threshold", [0.0, 0.05, 0.2, 0.5, 1.0]),
		ParamSweep("terrain-gully-spi-percentile", [60.0, 75.0, 88.0, 96.0, 99.5]),
		ParamSweep("terrain-gully-min-drainage-area-m2", [200.0, 1000.0, 4000.0, 15000.0, 50000.0]),
		ParamSweep("terrain-gully-min-slope-deg", [2.0, 8.0, 13.0, 22.0, 35.0]),
		ParamSweep("terrain-gully-max-slope-deg", [25.0, 35.0, 48.0, 60.0, 80.0]),
		ParamSweep("terrain-lake-max-slope-deg", [1.0, 3.0, 6.0, 12.0, 20.0]),
		ParamSweep("terrain-lake-tpi-threshold", [-8.0, -4.0, -1.8, -0.8, -0.2]),
		ParamSweep("terrain-lake-max-spi-threshold", [0.0, 0.02, 0.1, 0.3, 0.7]),
		ParamSweep("terrain-lake-max-spi-percentile", [5.0, 20.0, 35.0, 60.0, 90.0]),
		ParamSweep("zones-start-threshold", [0.2, 0.6, 0.9, 0.99, 0.9999]),
		ParamSweep("zones-ending-threshold", [0.001, 0.02, 0.075, 0.2, 0.6]),
		ParamSweep("runout-flux-min-threshold", [0.0001, 0.001, 0.01, 0.1, 1.0]),
		ParamSweep("runout-min-evidence-threshold", [0.001, 0.01, 0.03, 0.1, 0.3]),
		ParamSweep("ponderador-forest-type", ["stems", "bav", "pcc", "sen2cc"]),
	]


def _default_post_params() -> dict[str, Any]:
	return {
		"ates-forest-window": 5,
		"ates-slope-sigma": 1.0,
		"ates-forest-adjustment": "paper_pra",
		"landform-windows": ",".join(str(v) for v in range(5, 31)),
		"landform-curvature-threshold": 1e-4,
		"landform-flat-gradient-eps": 1e-10,
		"terrain-forest-tree-threshold": 35.0,
		"terrain-energy-trauma-threshold": 0.35,
		"terrain-gully-energy-threshold": 0.22,
		"terrain-gully-spi-m": 1.0,
		"terrain-gully-spi-n": 1.0,
		"terrain-gully-spi-threshold": 0.0,
		"terrain-gully-spi-percentile": 88.0,
		"terrain-gully-min-drainage-area-m2": 4000.0,
		"terrain-gully-min-slope-deg": 13.0,
		"terrain-gully-max-slope-deg": 48.0,
		"terrain-lake-max-slope-deg": 6.0,
		"terrain-lake-tpi-threshold": -1.8,
		"terrain-lake-max-spi-threshold": 0.0,
		"terrain-lake-max-spi-percentile": 35.0,
		"zones-start-threshold": 0.99,
		"zones-ending-threshold": 0.075,
		"runout-flux-min-threshold": 0.01,
		"runout-min-evidence-threshold": 0.03,
		"ponderador-forest-type": "pcc",
		"ponderador-output-name": "Ponderador_ATES.tif",
	}


def _load_main_module(main_script: Path, repo_root: Path) -> Any:
	import importlib.util

	# Ensure project-local package imports from main.py work when loaded dynamically.
	for p in (repo_root, main_script.parent):
		p_str = str(p.resolve())
		if p_str not in sys.path:
			sys.path.insert(0, p_str)

	spec = importlib.util.spec_from_file_location("app_ates_main_module", str(main_script))
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Unable to import main.py from {main_script}")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _prepare_ultrafast_baseline(
	args: argparse.Namespace,
	repo_root: Path,
	main_script: Path,
	dataset: DatasetSpec,
	baseline_dir: Path,
) -> tuple[bool, str, str]:
	cmd = [
		args.python_exe,
		str(main_script),
		"--dem",
		str(dataset.dem_path),
		"--forest",
		str(dataset.forest_path),
		"--outputs-dir",
		str(baseline_dir),
		"--until-n",
		"6",
		"--quiet",
	]
	proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
	stdout_tail = "\n".join(proc.stdout.splitlines()[-30:])
	stderr_tail = "\n".join(proc.stderr.splitlines()[-30:])
	return proc.returncode == 0, stdout_tail, stderr_tail


def _copy_baseline_exposures_to_run(baseline_dir: Path, run_dir: Path) -> None:
	baseline_def = baseline_dir / "Definitive_Layers"
	run_def = run_dir / "Definitive_Layers"
	run_def.mkdir(parents=True, exist_ok=True)

	for basin_dir in baseline_def.glob("Basin*"):
		if not basin_dir.is_dir():
			continue
		src = basin_dir / "Exposure_zdelta_cellcount.tif"
		if not src.exists():
			continue
		dst_basin = run_def / basin_dir.name
		dst_basin.mkdir(parents=True, exist_ok=True)
		shutil.copy2(src, dst_basin / "Exposure_zdelta_cellcount.tif")


def _run_ultrafast_postpipeline_once(
	main_mod: Any,
	dataset: DatasetSpec,
	baseline_dir: Path,
	run_dir: Path,
	param_name: str,
	param_value: Any,
) -> None:
	params = _default_post_params()
	params[param_name] = param_value

	# Step precomputed assets (from baseline up to step 6)
	out_02 = baseline_dir / "Preprocess"
	out_05 = baseline_dir / "Watershed_Subdivisions"
	out_06 = baseline_dir / "Flow-Py"
	dem_filled = out_02 / "dem_filled_simple.tif"
	forest_aligned = out_02 / "forest_aligned.tif"

	if not dem_filled.exists():
		raise RuntimeError(f"UltraFast baseline missing DEM: {dem_filled}")
	forest_for_steps = forest_aligned if forest_aligned.exists() else dataset.forest_path

	_copy_baseline_exposures_to_run(baseline_dir=baseline_dir, run_dir=run_dir)
	out_08 = run_dir / "Definitive_Layers"
	out_08.mkdir(parents=True, exist_ok=True)

	# Step 7
	main_mod.step_07_postprocess_flowpy(
		flowpy_out_dir=out_06,
		out_dir=out_08,
		dem_original_path=dataset.dem_path,
	)

	# Step 9
	main_mod.step_09_slope_and_forest_classification(
		dem_path=dem_filled,
		forest_pcc_path=forest_for_steps,
		out_dir=out_08,
		forest_window=int(params["ates-forest-window"]),
		slope_sigma=float(params["ates-slope-sigma"]),
		forest_adjustment=str(params["ates-forest-adjustment"]),
	)

	# Step 10
	main_mod.step_10_landforms_multiscale(
		dem_path=dem_filled,
		out_dir=out_08,
		landform_windows=str(params["landform-windows"]),
		curvature_threshold=float(params["landform-curvature-threshold"]),
		flat_gradient_eps=float(params["landform-flat-gradient-eps"]),
	)

	# Step 11
	gully_spi_threshold = float(params["terrain-gully-spi-threshold"])
	lake_spi_threshold = float(params["terrain-lake-max-spi-threshold"])
	main_mod.step_11_terrain_traps(
		dem_path=dem_filled,
		forest_path=forest_for_steps,
		definitive_layers_dir=out_08,
		flowpy_out_dir=out_06,
		forest_tree_threshold=float(params["terrain-forest-tree-threshold"]),
		energy_trauma_threshold=float(params["terrain-energy-trauma-threshold"]),
		gully_energy_threshold=float(params["terrain-gully-energy-threshold"]),
		gully_spi_m=float(params["terrain-gully-spi-m"]),
		gully_spi_n=float(params["terrain-gully-spi-n"]),
		gully_spi_threshold=(None if gully_spi_threshold <= 0 else gully_spi_threshold),
		gully_spi_percentile=float(params["terrain-gully-spi-percentile"]),
		gully_min_drainage_area_m2=float(params["terrain-gully-min-drainage-area-m2"]),
		gully_min_slope_deg=float(params["terrain-gully-min-slope-deg"]),
		gully_max_slope_deg=float(params["terrain-gully-max-slope-deg"]),
		lake_max_slope_deg=float(params["terrain-lake-max-slope-deg"]),
		lake_tpi_threshold=float(params["terrain-lake-tpi-threshold"]),
		lake_max_spi_threshold=(None if lake_spi_threshold <= 0 else lake_spi_threshold),
		lake_max_spi_percentile=float(params["terrain-lake-max-spi-percentile"]),
	)

	# Step 12
	main_mod.step_12_start_propagating_ending_zones(
		flowpy_out_dir=out_06,
		definitive_layers_dir=out_08,
		start_threshold=float(params["zones-start-threshold"]),
		ending_threshold=float(params["zones-ending-threshold"]),
	)

	# Step 13
	main_mod.step_13_runout_zone_characteristics(
		definitive_layers_dir=out_08,
		flowpy_out_dir=out_06,
		flux_min_threshold=float(params["runout-flux-min-threshold"]),
		min_evidence_threshold=float(params["runout-min-evidence-threshold"]),
	)

	# Step 14
	main_mod.step_14_ponderador_autoates(
		dem_path=dem_filled,
		forest_path=forest_for_steps,
		watershed_out_dir=out_05,
		flowpy_out_dir=out_06,
		definitive_layers_dir=out_08,
		forest_type=str(params["ponderador-forest-type"]),
		output_name=str(params["ponderador-output-name"]),
	)


def _cap_values(values: list[Any], requested: int | None) -> list[Any]:
	if requested is None:
		return values
	if requested < 3:
		raise ValueError("--values-per-param must be >= 3")
	if len(values) <= requested:
		return values
	return values[:requested]


def _select_datasets(all_sets: dict[str, DatasetSpec], selector: str) -> list[DatasetSpec]:
	if selector.strip().lower() == "all":
		return [all_sets[k] for k in sorted(all_sets.keys())]

	requested_tokens = [t.strip() for t in selector.split(",") if t.strip()]
	if not requested_tokens:
		raise ValueError("No datasets selected.")

	by_key = {k.upper(): v for k, v in all_sets.items()}
	by_dem_name = {v.dem_path.name.upper(): v for v in all_sets.values()}

	selected: list[DatasetSpec] = []
	seen: set[str] = set()
	for token in requested_tokens:
		token_u = token.upper()
		item = by_key.get(token_u) or by_dem_name.get(token_u)
		if item is None:
			raise ValueError(f"Dataset selector not found: {token}")
		if item.key not in seen:
			selected.append(item)
			seen.add(item.key)
	return selected


def _build_main_command(
	python_exe: str,
	main_script: Path,
	dataset: DatasetSpec,
	out_dir: Path,
	param_name: str,
	param_value: Any,
) -> list[str]:
	cmd = [
		python_exe,
		str(main_script),
		"--dem",
		str(dataset.dem_path),
		"--forest",
		str(dataset.forest_path),
		"--outputs-dir",
		str(out_dir),
		"--quiet",
	]

	# Always run full pipeline to produce final ATES comparable raster.
	cmd.extend(["--until-n", "14"])
	cmd.extend([f"--{param_name}", str(param_value)])
	return cmd


def _read_masked_raster(path: Path) -> tuple[np.ndarray, np.ndarray, Any, Any, Any]:
	with rasterio.open(path) as src:
		band = src.read(1, masked=True)
		# Convert to float before filling with NaN to avoid integer fill_value errors.
		data = np.array(band.astype(np.float64).filled(np.nan), dtype=np.float64)
		valid_mask = ~band.mask
		return data, valid_mask, src.transform, src.crs, src.res


def _class_distribution(data: np.ndarray, valid_mask: np.ndarray) -> dict[int, int]:
	out: dict[int, int] = {}
	for c in CLASSES_OF_INTEREST:
		out[c] = int(np.count_nonzero((data == c) & valid_mask))
	return out


def compare_rasters(test_raster: Path, ref_raster: Path) -> dict[str, Any]:
	test_data, test_valid, test_transform, test_crs, test_res = _read_masked_raster(test_raster)
	ref_data, ref_valid, ref_transform, ref_crs, ref_res = _read_masked_raster(ref_raster)

	test_counts = _class_distribution(test_data, test_valid)
	ref_counts = _class_distribution(ref_data, ref_valid)

	test_sum = sum(test_counts.values())
	ref_sum = sum(ref_counts.values())

	per_class: dict[str, Any] = {}
	ratio_abs_diffs: list[float] = []
	for c in CLASSES_OF_INTEREST:
		test_ratio = (test_counts[c] / test_sum) if test_sum > 0 else 0.0
		ref_ratio = (ref_counts[c] / ref_sum) if ref_sum > 0 else 0.0
		diff = abs(test_ratio - ref_ratio)
		ratio_abs_diffs.append(diff)
		per_class[str(c)] = {
			"test_count": test_counts[c],
			"ref_count": ref_counts[c],
			"test_ratio_in_234": test_ratio,
			"ref_ratio_in_234": ref_ratio,
			"abs_ratio_diff": diff,
		}

	score_distribution = float(np.mean(ratio_abs_diffs)) if ratio_abs_diffs else float("nan")

	same_grid = (
		test_data.shape == ref_data.shape
		and test_transform == ref_transform
		and str(test_crs) == str(ref_crs)
	)

	cellwise_agreement_234 = float("nan")
	if same_grid:
		test_234 = np.isin(test_data, CLASSES_OF_INTEREST) & test_valid
		ref_234 = np.isin(ref_data, CLASSES_OF_INTEREST) & ref_valid
		union_234 = test_234 | ref_234
		if np.any(union_234):
			cellwise_agreement_234 = float(np.mean(test_data[union_234] == ref_data[union_234]))

	return {
		"test_shape": list(test_data.shape),
		"ref_shape": list(ref_data.shape),
		"test_res": list(test_res),
		"ref_res": list(ref_res),
		"same_grid": same_grid,
		"test_valid_cells": int(np.count_nonzero(test_valid)),
		"ref_valid_cells": int(np.count_nonzero(ref_valid)),
		"test_sum_234": test_sum,
		"ref_sum_234": ref_sum,
		"distribution_score_mean_abs_diff": score_distribution,
		"cellwise_agreement_234": cellwise_agreement_234,
		"per_class": per_class,
	}


def _result_raster_path(run_dir: Path) -> Path:
	return run_dir / "Definitive_Layers" / "Ponderador_ATES.tif"


def _is_valid_threshold_combo(param_name: str, param_value: Any) -> bool:
	# Defaults in main.py impose start > ending. We vary one parameter at a time.
	if param_name == "zones-start-threshold" and float(param_value) <= 0.075:
		return False
	if param_name == "zones-ending-threshold" and float(param_value) >= 0.99:
		return False
	return True


def _fmt_float(v: Any, digits: int = 6) -> str:
	if v is None:
		return ""
	if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
		return ""
	if isinstance(v, (int, float)):
		return f"{float(v):.{digits}f}"
	return str(v)


def _value_sort_key(v: Any) -> tuple[int, Any]:
	if isinstance(v, (int, float)):
		return (0, float(v))
	return (1, str(v))


def _safe_float(v: Any) -> float | None:
	try:
		f = float(v)
	except (TypeError, ValueError):
		return None
	if math.isnan(f) or math.isinf(f):
		return None
	return f


def _compute_general_stats(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
	total_runs = len(summary_rows)
	status_counts: dict[str, int] = {}
	for row in summary_rows:
		status = str(row.get("status", "unknown"))
		status_counts[status] = status_counts.get(status, 0) + 1

	ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
	with_dist = [r for r in ok_rows if _safe_float(r.get("distribution_score_mean_abs_diff")) is not None]
	with_agreement = [r for r in ok_rows if _safe_float(r.get("cellwise_agreement_234")) is not None]

	all_dist = [_safe_float(r.get("distribution_score_mean_abs_diff")) for r in with_dist]
	all_dist = [v for v in all_dist if v is not None]
	all_agreement = [_safe_float(r.get("cellwise_agreement_234")) for r in with_agreement]
	all_agreement = [v for v in all_agreement if v is not None]

	global_best_run: dict[str, Any] | None = None
	if with_dist:
		global_best_run = min(with_dist, key=lambda r: float(r["distribution_score_mean_abs_diff"]))

	per_dataset: dict[str, dict[str, Any]] = {}
	datasets = sorted({str(r.get("dataset", "")) for r in summary_rows if r.get("dataset")})
	for ds in datasets:
		ds_rows = [r for r in summary_rows if r.get("dataset") == ds]
		ds_ok = [r for r in ds_rows if r.get("status") == "ok"]
		ds_dist = [_safe_float(r.get("distribution_score_mean_abs_diff")) for r in ds_ok]
		ds_dist = [v for v in ds_dist if v is not None]
		ds_ag = [_safe_float(r.get("cellwise_agreement_234")) for r in ds_ok]
		ds_ag = [v for v in ds_ag if v is not None]

		best_row: dict[str, Any] | None = None
		if ds_dist:
			best_row = min(
				[r for r in ds_ok if _safe_float(r.get("distribution_score_mean_abs_diff")) is not None],
				key=lambda r: float(r["distribution_score_mean_abs_diff"]),
			)

		per_dataset[ds] = {
			"total_runs": len(ds_rows),
			"ok_runs": len(ds_ok),
			"mean_dist_score": float(np.mean(ds_dist)) if ds_dist else None,
			"min_dist_score": float(np.min(ds_dist)) if ds_dist else None,
			"max_dist_score": float(np.max(ds_dist)) if ds_dist else None,
			"mean_agreement": float(np.mean(ds_ag)) if ds_ag else None,
			"best_param": best_row.get("param") if best_row else None,
			"best_value": best_row.get("value") if best_row else None,
			"best_run_dir": best_row.get("run_dir") if best_row else None,
		}

	param_sensitivity: list[dict[str, Any]] = []
	for ds in datasets:
		ds_ok = [r for r in ok_rows if r.get("dataset") == ds]
		params = sorted({str(r.get("param", "")) for r in ds_ok if r.get("param")})
		for param in params:
			rows = [r for r in ds_ok if r.get("param") == param]
			dists = [_safe_float(r.get("distribution_score_mean_abs_diff")) for r in rows]
			dists = [v for v in dists if v is not None]
			if not dists:
				continue
			param_sensitivity.append(
				{
					"dataset": ds,
					"param": param,
					"min_dist_score": float(np.min(dists)),
					"max_dist_score": float(np.max(dists)),
					"span_dist_score": float(np.max(dists) - np.min(dists)),
					"n_values": len(dists),
				}
			)

	best_by_dataset_param: list[dict[str, Any]] = []
	for ds in datasets:
		ds_ok = [r for r in ok_rows if r.get("dataset") == ds]
		params = sorted({str(r.get("param", "")) for r in ds_ok if r.get("param")})
		for param in params:
			rows = [r for r in ds_ok if r.get("param") == param]
			rows = [r for r in rows if _safe_float(r.get("distribution_score_mean_abs_diff")) is not None]
			if not rows:
				continue
			best_row = min(rows, key=lambda r: float(r["distribution_score_mean_abs_diff"]))
			best_by_dataset_param.append(
				{
					"dataset": ds,
					"param": param,
					"best_value": best_row.get("value"),
					"best_dist_score": float(best_row["distribution_score_mean_abs_diff"]),
					"best_agreement": _safe_float(best_row.get("cellwise_agreement_234")),
					"run_dir": best_row.get("run_dir"),
				}
			)

	return {
		"total_runs": total_runs,
		"status_counts": status_counts,
		"ok_runs": len(ok_rows),
		"ok_runs_with_dist": len(with_dist),
		"ok_runs_with_agreement": len(with_agreement),
		"mean_dist_score": float(np.mean(all_dist)) if all_dist else None,
		"min_dist_score": float(np.min(all_dist)) if all_dist else None,
		"max_dist_score": float(np.max(all_dist)) if all_dist else None,
		"mean_agreement": float(np.mean(all_agreement)) if all_agreement else None,
		"min_agreement": float(np.min(all_agreement)) if all_agreement else None,
		"max_agreement": float(np.max(all_agreement)) if all_agreement else None,
		"global_best_run": global_best_run,
		"per_dataset": per_dataset,
		"param_sensitivity": sorted(
			param_sensitivity,
			key=lambda x: (str(x["dataset"]), -float(x["span_dist_score"]), str(x["param"])),
		),
		"best_by_dataset_param": sorted(
			best_by_dataset_param,
			key=lambda x: (str(x["dataset"]), str(x["param"])),
		),
	}


def _build_html_table(headers: list[str], rows: list[list[str]]) -> str:
	head_cells = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
	body_parts: list[str] = []
	for row in rows:
		cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
		body_parts.append(f"<tr>{cells}</tr>")
	body = "\n".join(body_parts)
	return f"<table><thead><tr>{head_cells}</tr></thead><tbody>{body}</tbody></table>"


def _build_analytics_report(report_dir: Path, summary_rows: list[dict[str, Any]], dry_run: bool) -> tuple[Path, Path]:
	stats = _compute_general_stats(summary_rows)
	stats_json = report_dir / "general_stats.json"
	stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=True), encoding="utf-8")

	best_table_rows = []
	for row in stats.get("best_by_dataset_param", []):
		best_table_rows.append(
			[
				str(row.get("dataset", "")),
				str(row.get("param", "")),
				str(row.get("best_value", "")),
				_fmt_float(row.get("best_dist_score"), 6),
				_fmt_float(row.get("best_agreement"), 6),
				str(row.get("run_dir", "")),
			]
		)

	per_dataset_rows = []
	for ds, ds_stats in sorted(stats.get("per_dataset", {}).items()):
		per_dataset_rows.append(
			[
				ds,
				str(ds_stats.get("total_runs", "")),
				str(ds_stats.get("ok_runs", "")),
				_fmt_float(ds_stats.get("mean_dist_score"), 6),
				_fmt_float(ds_stats.get("min_dist_score"), 6),
				_fmt_float(ds_stats.get("max_dist_score"), 6),
				_fmt_float(ds_stats.get("mean_agreement"), 6),
				str(ds_stats.get("best_param", "")),
				str(ds_stats.get("best_value", "")),
			]
		)

	status_counts = stats.get("status_counts", {})

	plotly_error = ""
	chart_sections: list[str] = []
	try:
		import importlib

		go = importlib.import_module("plotly.graph_objects")
		plot = importlib.import_module("plotly.offline").plot

		fig_status = go.Figure(
			data=[
				go.Bar(
					x=list(status_counts.keys()),
					y=[int(status_counts[k]) for k in status_counts.keys()],
				)
			],
		)
		fig_status.update_layout(title="Run Status Count", xaxis_title="Status", yaxis_title="Runs")
		chart_sections.append("<h3>Run Status</h3>" + plot(fig_status, include_plotlyjs="cdn", output_type="div"))

		ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
		ds_for_dist = [str(r.get("dataset", "")) for r in ok_rows if _safe_float(r.get("distribution_score_mean_abs_diff")) is not None]
		dist_for_dist = [float(r.get("distribution_score_mean_abs_diff")) for r in ok_rows if _safe_float(r.get("distribution_score_mean_abs_diff")) is not None]
		if ds_for_dist:
			fig_dist = go.Figure(data=[go.Box(x=ds_for_dist, y=dist_for_dist, boxmean=True)])
			fig_dist.update_layout(
				title="DistScore Distribution by DEM",
				xaxis_title="DEM",
				yaxis_title="DistScore (lower is better)",
			)
			chart_sections.append("<h3>DistScore by DEM</h3>" + plot(fig_dist, include_plotlyjs=False, output_type="div"))

		ag_rows = [r for r in ok_rows if _safe_float(r.get("cellwise_agreement_234")) is not None and _safe_float(r.get("distribution_score_mean_abs_diff")) is not None]
		if ag_rows:
			fig_scatter = go.Figure(
				data=[
					go.Scatter(
						x=[float(r["distribution_score_mean_abs_diff"]) for r in ag_rows],
						y=[float(r["cellwise_agreement_234"]) for r in ag_rows],
						mode="markers",
						text=[
							f"{r.get('dataset')} | {r.get('param')}={r.get('value')}"
							for r in ag_rows
						],
						hovertemplate="%{text}<br>DistScore=%{x:.6f}<br>Agreement=%{y:.6f}<extra></extra>",
					)
				]
			)
			fig_scatter.update_layout(
				title="DistScore vs Agreement234",
				xaxis_title="DistScore (lower is better)",
				yaxis_title="Agreement234 (higher is better)",
			)
			chart_sections.append("<h3>DistScore vs Agreement</h3>" + plot(fig_scatter, include_plotlyjs=False, output_type="div"))

		sensitivity = stats.get("param_sensitivity", [])
		if sensitivity:
			top_sens = sensitivity[:20]
			fig_sens = go.Figure(
				data=[
					go.Bar(
						x=[f"{r['dataset']}::{r['param']}" for r in top_sens],
						y=[float(r["span_dist_score"]) for r in top_sens],
					)
				]
			)
			fig_sens.update_layout(
				title="Top Parameter Sensitivity (DistScore span)",
				xaxis_title="DEM::Parameter",
				yaxis_title="Span (max-min DistScore)",
				xaxis_tickangle=-30,
			)
			chart_sections.append("<h3>Parameter Sensitivity</h3>" + plot(fig_sens, include_plotlyjs=False, output_type="div"))

		forest_rows = [
			r for r in ok_rows if str(r.get("param", "")) == "ponderador-forest-type" and _safe_float(r.get("distribution_score_mean_abs_diff")) is not None
		]
		if forest_rows:
			fig_forest = go.Figure()
			datasets = sorted({str(r.get("dataset", "")) for r in forest_rows})
			for ds in datasets:
				ds_rows = [r for r in forest_rows if r.get("dataset") == ds]
				fig_forest.add_trace(
					go.Bar(
						name=ds,
						x=[str(r.get("value", "")) for r in ds_rows],
						y=[float(r["distribution_score_mean_abs_diff"]) for r in ds_rows],
					)
				)
			fig_forest.update_layout(
				title="Forest Type Impact on DistScore",
				xaxis_title="ponderador-forest-type",
				yaxis_title="DistScore",
				barmode="group",
			)
			chart_sections.append("<h3>Forest Type Impact</h3>" + plot(fig_forest, include_plotlyjs=False, output_type="div"))

	except Exception as e:
		plotly_error = str(e)

	overview_stats = [
		["total_runs", str(stats.get("total_runs", ""))],
		["ok_runs", str(stats.get("ok_runs", ""))],
		["ok_runs_with_dist", str(stats.get("ok_runs_with_dist", ""))],
		["ok_runs_with_agreement", str(stats.get("ok_runs_with_agreement", ""))],
		["mean_dist_score", _fmt_float(stats.get("mean_dist_score"), 6)],
		["min_dist_score", _fmt_float(stats.get("min_dist_score"), 6)],
		["max_dist_score", _fmt_float(stats.get("max_dist_score"), 6)],
		["mean_agreement", _fmt_float(stats.get("mean_agreement"), 6)],
		["min_agreement", _fmt_float(stats.get("min_agreement"), 6)],
		["max_agreement", _fmt_float(stats.get("max_agreement"), 6)],
	]

	global_best = stats.get("global_best_run")
	global_best_html = "<p>No valid run with DistScore was found.</p>"
	if global_best:
		global_best_rows = [
			["dataset", str(global_best.get("dataset", ""))],
			["param", str(global_best.get("param", ""))],
			["value", str(global_best.get("value", ""))],
			["dist_score", _fmt_float(global_best.get("distribution_score_mean_abs_diff"), 6)],
			["agreement", _fmt_float(global_best.get("cellwise_agreement_234"), 6)],
			["run_dir", str(global_best.get("run_dir", ""))],
		]
		global_best_html = _build_html_table(["Field", "Value"], global_best_rows)

	plotly_note = ""
	if plotly_error:
		plotly_note = (
			"<p><strong>Plotly charts were not generated</strong>: "
			+ html.escape(plotly_error)
			+ "</p>"
		)

	report_html = report_dir / "analytics_report.html"
	html_content = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Experiment Analytics Report</title>
  <style>
    body { font-family: Segoe UI, Tahoma, sans-serif; margin: 24px; color: #1f2933; background: #f5f7fa; }
    h1, h2, h3 { color: #102a43; }
    .card { background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    table { border-collapse: collapse; width: 100%; margin-top: 8px; }
    th, td { border: 1px solid #bcccdc; padding: 6px 8px; text-align: left; font-size: 13px; }
    th { background: #e4e7eb; }
    .meta { color: #486581; font-size: 13px; }
  </style>
</head>
<body>
  <h1>Experiment Analytics Report</h1>
  <p class=\"meta\">Generated: __GENERATED__ | Dry-run: __DRY_RUN__</p>

  <div class=\"card\">
    <h2>Overview</h2>
    __OVERVIEW_TABLE__
  </div>

  <div class=\"card\">
    <h2>Global Best Run</h2>
    __GLOBAL_BEST_TABLE__
  </div>

  <div class=\"card\">
    <h2>Per DEM Summary</h2>
    __PER_DATASET_TABLE__
  </div>

  <div class=\"card\">
    <h2>Best Value per DEM and Parameter</h2>
    __BEST_TABLE__
  </div>

  <div class=\"card\">
    <h2>Charts</h2>
    __PLOTLY_NOTE__
    __CHARTS__
  </div>
</body>
</html>
"""

	html_content = html_content.replace("__GENERATED__", datetime.now().isoformat(timespec="seconds"))
	html_content = html_content.replace("__DRY_RUN__", "true" if dry_run else "false")
	html_content = html_content.replace("__OVERVIEW_TABLE__", _build_html_table(["Metric", "Value"], overview_stats))
	html_content = html_content.replace("__GLOBAL_BEST_TABLE__", global_best_html)
	html_content = html_content.replace(
		"__PER_DATASET_TABLE__",
		_build_html_table(
			[
				"DEM",
				"Total runs",
				"OK runs",
				"Mean DistScore",
				"Min DistScore",
				"Max DistScore",
				"Mean Agreement",
				"Best parameter",
				"Best value",
			],
			per_dataset_rows,
		),
	)
	html_content = html_content.replace(
		"__BEST_TABLE__",
		_build_html_table(
			["DEM", "Parameter", "Best value", "Best DistScore", "Best Agreement", "Run directory"],
			best_table_rows,
		),
	)
	html_content = html_content.replace("__PLOTLY_NOTE__", plotly_note)
	html_content = html_content.replace("__CHARTS__", "\n".join(chart_sections) if chart_sections else "<p>No charts available.</p>")

	report_html.write_text(html_content, encoding="utf-8")
	return report_html, stats_json


def _rebuild_reports_from_existing_dir(report_dir: Path) -> tuple[Path, Path, Path]:
	summary_json = report_dir / "summary.json"
	if not summary_json.exists():
		raise RuntimeError(f"Missing summary.json in report dir: {summary_json}")

	summary_rows = json.loads(summary_json.read_text(encoding="utf-8"))
	plan_json = report_dir / "plan.json"
	dry_run = False
	if plan_json.exists():
		plan_payload = json.loads(plan_json.read_text(encoding="utf-8"))
		dry_run = bool(plan_payload.get("dry_run", False))

	comparative_md = report_dir / "comparative_tables.md"
	comparative_text = _build_comparative_tables(summary_rows=summary_rows, dry_run=dry_run)
	comparative_md.write_text(comparative_text, encoding="utf-8")

	report_html, stats_json = _build_analytics_report(report_dir=report_dir, summary_rows=summary_rows, dry_run=dry_run)
	return comparative_md, report_html, stats_json


def _build_comparative_tables(summary_rows: list[dict[str, Any]], dry_run: bool) -> str:
	lines: list[str] = []
	lines.append("# Comparative Tables")
	lines.append("")
	lines.append(
		"- DistScore: mean absolute difference between class distribution ratios (2/3/4) against validated raster (lower is better)."
	)
	lines.append(
		"- Agreement234: cell-wise agreement on classes 2/3/4 where test and validated overlap in active cells (higher is better)."
	)
	lines.append("- dC2/dC3/dC4: difference in class counts (test - validated).")
	lines.append("")

	datasets = sorted({str(row.get("dataset", "")) for row in summary_rows if row.get("dataset")})
	for ds in datasets:
		lines.append(f"## DEM: {ds}")
		lines.append("")
		ds_rows = [r for r in summary_rows if r.get("dataset") == ds]
		params = sorted({str(r.get("param", "")) for r in ds_rows if r.get("param")})

		for param in params:
			lines.append(f"### Param: {param}")
			lines.append("")
			if dry_run:
				lines.append("| Value | Status | RunDir |")
				lines.append("|---|---|---|")
				for row in sorted(
					[r for r in ds_rows if r.get("param") == param],
					key=lambda r: _value_sort_key(r.get("value")),
				):
					lines.append(
						f"| {row.get('value', '')} | {row.get('status', '')} | {row.get('run_dir', '')} |"
					)
				lines.append("")
				continue

			lines.append("| Value | Status | DistScore | Agreement234 | dC2 | dC3 | dC4 |")
			lines.append("|---|---|---:|---:|---:|---:|---:|")
			for row in sorted(
				[r for r in ds_rows if r.get("param") == param],
				key=lambda r: _value_sort_key(r.get("value")),
			):
				metrics_json = row.get("metrics_json")
				dc2 = ""
				dc3 = ""
				dc4 = ""
				if metrics_json:
					try:
						metrics = json.loads(metrics_json)
						pc = metrics.get("per_class", {})
						dc2 = str(int(pc.get("2", {}).get("test_count", 0) - pc.get("2", {}).get("ref_count", 0)))
						dc3 = str(int(pc.get("3", {}).get("test_count", 0) - pc.get("3", {}).get("ref_count", 0)))
						dc4 = str(int(pc.get("4", {}).get("test_count", 0) - pc.get("4", {}).get("ref_count", 0)))
					except Exception:
						pass

				lines.append(
					"| "
					+ f"{row.get('value', '')} | "
					+ f"{row.get('status', '')} | "
					+ f"{_fmt_float(row.get('distribution_score_mean_abs_diff'), 6)} | "
					+ f"{_fmt_float(row.get('cellwise_agreement_234'), 6)} | "
					+ f"{dc2} | {dc3} | {dc4} |"
				)
			lines.append("")

	return "\n".join(lines) + "\n"


def run_experiment(args: argparse.Namespace) -> Path:
	repo_root = Path(__file__).resolve().parents[1]
	main_script = (repo_root / args.main_script).resolve()
	inputs_dir = (repo_root / args.inputs_dir).resolve()
	validated_dir = (repo_root / args.validated_dir).resolve()
	runs_root = (repo_root / args.runs_root).resolve()

	print("[experiment] Starting experiment run...", flush=True)
	print(f"[experiment] Mode: {args.mode}", flush=True)
	print(f"[experiment] Dry-run: {args.dry_run}", flush=True)
	print(f"[experiment] Main script: {main_script}", flush=True)
	print(f"[experiment] Inputs dir: {inputs_dir}", flush=True)
	print(f"[experiment] Validated dir: {validated_dir}", flush=True)

	datasets = discover_datasets(inputs_dir=inputs_dir, validated_dir=validated_dir)
	if not datasets:
		raise RuntimeError(
			f"No datasets discovered in {inputs_dir} with matching FOREST_* and ATES_COMPROVAT_* references in {validated_dir}."
		)
	print(f"[experiment] Datasets discovered: {len(datasets)}", flush=True)

	selected = _select_datasets(datasets, args.dems)
	if not selected:
		raise RuntimeError("No datasets selected for execution.")
	print(
		"[experiment] Selected datasets: " + ", ".join(ds.key for ds in selected),
		flush=True,
	)

	if args.mode == "quick":
		sweeps = _quick_param_sweeps()
	elif args.mode == "extensive":
		sweeps = _extensive_param_sweeps()
	else:
		sweeps = _ultrafast_param_sweeps()
	if args.limit_params is not None:
		sweeps = sweeps[: args.limit_params]
	print(f"[experiment] Parameters to sweep: {len(sweeps)}", flush=True)

	for sw in sweeps:
		sw.values = _cap_values(sw.values, args.values_per_param)
		if len(sw.values) < 3:
			raise RuntimeError(f"Parameter {sw.name} has fewer than 3 test values after capping.")
		print(f"[experiment]  - {sw.name}: {len(sw.values)} values -> {sw.values}", flush=True)

	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	report_dir = runs_root / f"experiment_{args.mode}_{ts}"
	report_dir.mkdir(parents=True, exist_ok=True)
	print(f"[experiment] Report directory: {report_dir}", flush=True)

	total_planned_runs = 0
	for ds in selected:
		for sweep in sweeps:
			for value in sweep.values:
				if _is_valid_threshold_combo(sweep.name, value):
					total_planned_runs += 1
	print(f"[experiment] Total runs planned: {total_planned_runs}", flush=True)

	summary_rows: list[dict[str, Any]] = []
	run_idx = 0
	main_mod = None
	if args.mode == "ultrafast" and not args.dry_run:
		main_mod = _load_main_module(main_script=main_script, repo_root=repo_root)

	for ds in selected:
		print(f"[experiment] Dataset {ds.key}: starting", flush=True)
		baseline_dir = report_dir / "baselines" / ds.key
		if args.mode == "ultrafast":
			print(
				f"[experiment] Dataset {ds.key}: UltraFast baseline to step 6 (single run)",
				flush=True,
			)
			if args.dry_run:
				print(f"[experiment] Dataset {ds.key}: baseline planned at {baseline_dir}", flush=True)
			else:
				ok, out_tail, err_tail = _prepare_ultrafast_baseline(
					args=args,
					repo_root=repo_root,
					main_script=main_script,
					dataset=ds,
					baseline_dir=baseline_dir,
				)
				if not ok:
					raise RuntimeError(
						"UltraFast baseline failed for dataset "
						+ ds.key
						+ "\nSTDOUT:\n"
						+ out_tail
						+ "\nSTDERR:\n"
						+ err_tail
					)
				print(f"[experiment] Dataset {ds.key}: baseline ready -> {baseline_dir}", flush=True)
		for sweep in sweeps:
			print(f"[experiment]   Param {sweep.name}: starting", flush=True)
			for value in sweep.values:
				if not _is_valid_threshold_combo(sweep.name, value):
					print(
						f"[experiment]   Param {sweep.name} value {value}: skipped (invalid threshold combo)",
						flush=True,
					)
					continue

				run_idx += 1
				run_name = f"{run_idx:04d}_{ds.key}_{sweep.name.replace('-', '_')}_{str(value).replace('.', 'p')}"
				run_dir = report_dir / "runs" / run_name
				print(
					f"[experiment] [{run_idx}/{total_planned_runs}] {ds.key} --{sweep.name}={value}",
					flush=True,
				)

				if args.mode == "ultrafast":
					cmd = [
						"ULTRAFAST_INTERNAL",
						f"baseline={baseline_dir}",
						f"dataset={ds.key}",
						f"param={sweep.name}",
						f"value={value}",
					]
				else:
					cmd = _build_main_command(
						python_exe=args.python_exe,
						main_script=main_script,
						dataset=ds,
						out_dir=run_dir,
						param_name=sweep.name,
						param_value=value,
					)

				row: dict[str, Any] = {
					"run_id": run_idx,
					"dataset": ds.key,
					"dem": str(ds.dem_path),
					"forest": str(ds.forest_path),
					"validated_raster": str(ds.validated_path),
					"param": sweep.name,
					"value": value,
					"run_dir": str(run_dir),
					"command": " ".join(cmd),
					"status": "planned" if args.dry_run else "pending",
				}

				if args.dry_run:
					print(f"[experiment]     planned: {run_dir}", flush=True)
					summary_rows.append(row)
					continue

				run_dir.mkdir(parents=True, exist_ok=True)
				if args.mode == "ultrafast":
					print("[experiment]     executing UltraFast post-FlowPy steps (7..14)...", flush=True)
					try:
						_run_ultrafast_postpipeline_once(
							main_mod=main_mod,
							dataset=ds,
							baseline_dir=baseline_dir,
							run_dir=run_dir,
							param_name=sweep.name,
							param_value=value,
						)
					except Exception as e:
						row["status"] = "failed"
						row["exit_code"] = 1
						row["stdout_tail"] = ""
						row["stderr_tail"] = str(e)
						print(f"[experiment]     FAILED ({e})", flush=True)
						summary_rows.append(row)
						if args.stop_on_error:
							raise
						continue
					row["exit_code"] = 0
					row["stdout_tail"] = "ultrafast_internal_ok"
					row["stderr_tail"] = ""
				else:
					print("[experiment]     executing main.py...", flush=True)
					proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

					row["exit_code"] = proc.returncode
					row["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-30:])
					row["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-30:])

					if proc.returncode != 0:
						row["status"] = "failed"
						print(f"[experiment]     FAILED (exit code {proc.returncode})", flush=True)
						summary_rows.append(row)
						if args.stop_on_error:
							raise RuntimeError(
								f"Experiment run failed (run_id={run_idx}, dataset={ds.key}, param={sweep.name}, value={value})."
							)
						continue

				produced = _result_raster_path(run_dir)
				if not produced.exists():
					row["status"] = "failed_missing_output"
					print(f"[experiment]     FAILED missing output: {produced}", flush=True)
					summary_rows.append(row)
					if args.stop_on_error:
						raise RuntimeError(f"Expected output not found: {produced}")
					continue

				print("[experiment]     comparing result raster with validated raster...", flush=True)
				metrics = compare_rasters(produced, ds.validated_path)
				row["status"] = "ok"
				row["distribution_score_mean_abs_diff"] = metrics["distribution_score_mean_abs_diff"]
				row["cellwise_agreement_234"] = metrics["cellwise_agreement_234"]
				row["same_grid"] = metrics["same_grid"]
				row["test_sum_234"] = metrics["test_sum_234"]
				row["ref_sum_234"] = metrics["ref_sum_234"]
				row["metrics_json"] = json.dumps(metrics, ensure_ascii=True)
				print(
					"[experiment]     OK "
					+ f"DistScore={_fmt_float(row['distribution_score_mean_abs_diff'], 6)} "
					+ f"Agreement234={_fmt_float(row['cellwise_agreement_234'], 6)}",
					flush=True,
				)
				summary_rows.append(row)

	summary_json = report_dir / "summary.json"
	summary_csv = report_dir / "summary.csv"
	plan_json = report_dir / "plan.json"

	plan_payload = {
		"mode": args.mode,
		"dry_run": args.dry_run,
		"datasets": [
			{
				"key": ds.key,
				"dem": str(ds.dem_path),
				"forest": str(ds.forest_path),
				"validated": str(ds.validated_path),
			}
			for ds in selected
		],
		"sweeps": [{"name": sw.name, "values": sw.values} for sw in sweeps],
	}
	plan_json.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=True), encoding="utf-8")
	summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True), encoding="utf-8")
	print(f"[experiment] Wrote plan: {plan_json}", flush=True)
	print(f"[experiment] Wrote summary json: {summary_json}", flush=True)

	with summary_csv.open("w", newline="", encoding="utf-8") as f:
		if summary_rows:
			fieldnames = sorted({k for row in summary_rows for k in row.keys()})
		else:
			fieldnames = ["run_id", "dataset", "param", "value", "status"]
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in summary_rows:
			clean_row = {
				k: (
					""
					if isinstance(v, float) and (math.isnan(v) or math.isinf(v))
					else v
				)
				for k, v in row.items()
			}
			writer.writerow(clean_row)
	print(f"[experiment] Wrote summary csv: {summary_csv}", flush=True)

	comparative_md = report_dir / "comparative_tables.md"
	comparative_text = _build_comparative_tables(summary_rows=summary_rows, dry_run=args.dry_run)
	comparative_md.write_text(comparative_text, encoding="utf-8")
	report_html, stats_json = _build_analytics_report(
		report_dir=report_dir,
		summary_rows=summary_rows,
		dry_run=args.dry_run,
	)

	print(f"[experiment] Comparative table saved: {comparative_md}", flush=True)
	print(f"[experiment] Analytics report saved: {report_html}", flush=True)
	print(f"[experiment] General stats saved: {stats_json}", flush=True)
	if args.dry_run:
		print("[experiment] Dry-run mode: comparative table includes planned values and run directories.", flush=True)
	else:
		print(
			"[experiment] Comparative table includes metric impact per value "
			"(DistScore, Agreement234, dC2/dC3/dC4).",
			flush=True,
		)

	return report_dir


def main() -> None:
	args = _parse_args()
	if args.rebuild_report_dir:
		report_dir = Path(args.rebuild_report_dir).resolve()
		comparative_md, report_html, stats_json = _rebuild_reports_from_existing_dir(report_dir)
		print(f"Rebuilt comparative table: {comparative_md}")
		print(f"Rebuilt analytics report: {report_html}")
		print(f"Rebuilt general stats: {stats_json}")
		return

	report_dir = run_experiment(args)
	print(f"Experiment finished. Report directory: {report_dir}")


if __name__ == "__main__":
	main()

