#!/usr/bin/env python3
"""Compute overlap-aware propagation coverage index per starting-zone PRA.

For each basin and avalanche id (Ava_X):
- Denominator: number of start-zone cells (value 1 in Ava_X.tif)
- Numerator: effective propagated cells from Flow-Py outputs, overlap-adjusted

Effective propagated cells are computed from source_ids_bitmask constrained by a
multi-layer propagation evidence score built from:
- flux.tif
- z_delta.tif
- SL_travel_angle.tif

In overlapping propagated cells (shared by multiple avalanche ids), each PRA gets
fractional credit 1 / overlap_count for that pixel.

Output raster stores the coverage index on start-zone cells:
    coverage_index = effective_propagated_cells / start_cells

Cells outside start zones are 0.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio


DEFAULT_OUTPUT_NODATA = 0.0


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_single_band(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
        data = np.asarray(band.data)
        valid = ~np.asarray(band.mask)
        profile = src.profile.copy()
    return data, valid, profile


def _check_alignment(profile_a: dict, profile_b: dict, label_a: str, label_b: str) -> None:
    checks = [
        ("width", profile_a.get("width"), profile_b.get("width")),
        ("height", profile_a.get("height"), profile_b.get("height")),
        ("transform", profile_a.get("transform"), profile_b.get("transform")),
        ("crs", profile_a.get("crs"), profile_b.get("crs")),
    ]
    mismatches = [name for name, a, b in checks if a != b]
    if mismatches:
        raise ValueError(
            f"Rasters are not aligned ({label_a} vs {label_b}): {', '.join(mismatches)}"
        )


def _latest_result_dir(basin_dir: Path) -> Optional[Path]:
    res_dirs = [p for p in basin_dir.glob("res_*") if p.is_dir()]
    if not res_dirs:
        return None
    return max(res_dirs, key=lambda p: p.stat().st_mtime)


def _normalize_robust_0_1(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    sample_mask = np.logical_and(valid, np.isfinite(values))
    if not np.any(sample_mask):
        return out

    sample = values[sample_mask].astype(np.float32, copy=False)
    p05 = float(np.percentile(sample, 5.0))
    p95 = float(np.percentile(sample, 95.0))
    if np.isclose(p95, p05):
        out[sample_mask] = 0.0
        return out

    scaled = (sample - p05) / (p95 - p05)
    out[sample_mask] = np.clip(scaled, 0.0, 1.0)
    return out


def _popcount_uint64(bitmask: np.ndarray) -> np.ndarray:
    if bitmask.dtype != np.uint64:
        bitmask = bitmask.astype(np.uint64, copy=False)

    h, w = bitmask.shape
    bytes_view = bitmask.view(np.uint8).reshape(h, w, 8)
    bits = np.unpackbits(bytes_view, axis=2)
    return bits.sum(axis=2).astype(np.uint8, copy=False)


def _avalanche_ids_present(bitmask: np.ndarray) -> List[int]:
    ids: List[int] = []
    for avalanche_id in range(1, 65):
        bit = np.uint64(1) << np.uint64(avalanche_id - 1)
        if np.any((bitmask & bit) > 0):
            ids.append(avalanche_id)
    return ids


def _extract_basin_id_from_flowpy(name: str) -> int:
    match = re.match(r"^pra_basin_(\d+)$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid Flow-Py basin folder name: {name}")
    return int(match.group(1))


def compute_starting_zones_coverage(
    definitive_layers_dir: str | Path,
    flowpy_root: str | Path,
    out_raster_path: str | Path,
    out_stats_csv: str | Path,
    min_evidence_threshold: float = 0.05,
) -> List[Path]:
    definitive = Path(definitive_layers_dir).expanduser().resolve()
    flowpy = Path(flowpy_root).expanduser().resolve()
    out_raster = Path(out_raster_path).expanduser().resolve()
    out_stats = Path(out_stats_csv).expanduser().resolve()

    if not definitive.exists():
        raise FileNotFoundError(f"Definitive layers dir not found: {definitive}")
    if not flowpy.exists():
        raise FileNotFoundError(f"Flow-Py root not found: {flowpy}")

    ref_profile: Optional[dict] = None
    coverage_raster: Optional[np.ndarray] = None
    stats_rows: List[Dict[str, object]] = []

    basin_dirs = sorted([p for p in flowpy.iterdir() if p.is_dir() and p.name.lower().startswith("pra_basin_")])
    for basin_dir in basin_dirs:
        basin_id = _extract_basin_id_from_flowpy(basin_dir.name)
        res_dir = _latest_result_dir(basin_dir)
        if res_dir is None:
            continue

        bitmask_path = res_dir / "source_ids_bitmask.tif"
        flux_path = res_dir / "flux.tif"
        z_delta_path = res_dir / "z_delta.tif"
        sl_angle_path = res_dir / "SL_travel_angle.tif"

        required = [bitmask_path, flux_path, z_delta_path, sl_angle_path]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required Flow-Py outputs in {res_dir}: {', '.join(missing)}"
            )

        bitmask_raw, bitmask_valid, bitmask_profile = _read_single_band(bitmask_path)
        bitmask = bitmask_raw.astype(np.uint64, copy=False)

        flux, flux_valid, flux_profile = _read_single_band(flux_path)
        z_delta, z_valid, z_profile = _read_single_band(z_delta_path)
        sl_angle, sl_valid, sl_profile = _read_single_band(sl_angle_path)

        _check_alignment(bitmask_profile, flux_profile, "bitmask", "flux")
        _check_alignment(bitmask_profile, z_profile, "bitmask", "z_delta")
        _check_alignment(bitmask_profile, sl_profile, "bitmask", "SL_travel_angle")

        if ref_profile is None:
            ref_profile = bitmask_profile.copy()
            coverage_raster = np.full(
                (ref_profile["height"], ref_profile["width"]),
                DEFAULT_OUTPUT_NODATA,
                dtype=np.float32,
            )
        else:
            _check_alignment(ref_profile, bitmask_profile, "reference", str(bitmask_path))

        flux = flux.astype(np.float32, copy=False)
        z_delta = z_delta.astype(np.float32, copy=False)
        sl_angle = sl_angle.astype(np.float32, copy=False)

        flux_norm = _normalize_robust_0_1(flux, np.logical_and(flux_valid, flux > 0.0))
        z_norm = _normalize_robust_0_1(z_delta, np.logical_and(z_valid, z_delta > 0.0))
        sl_norm = _normalize_robust_0_1(sl_angle, np.logical_and(sl_valid, sl_angle > 0.0))

        evidence = (flux_norm + z_norm + sl_norm) / 3.0
        evidence_valid = np.logical_and.reduce(
            [
                bitmask_valid,
                np.isfinite(evidence),
                evidence > min_evidence_threshold,
            ]
        )

        overlap_count = _popcount_uint64(bitmask)
        overlap_safe = np.maximum(overlap_count.astype(np.float32, copy=False), 1.0)

        definitive_basin_dir = definitive / f"Basin{basin_id}" / "Star_propagating_Ending_Zones"
        if not definitive_basin_dir.exists():
            continue

        avalanche_ids = _avalanche_ids_present(bitmask)
        for avalanche_id in avalanche_ids:
            ava_path = definitive_basin_dir / f"Ava_{avalanche_id}.tif"
            if not ava_path.exists():
                continue

            ava_zone, ava_valid, ava_profile = _read_single_band(ava_path)
            _check_alignment(bitmask_profile, ava_profile, "bitmask", str(ava_path))

            start_mask = np.logical_and(ava_valid, ava_zone == 1)
            start_cells = int(np.count_nonzero(start_mask))
            if start_cells == 0:
                continue

            bit = np.uint64(1) << np.uint64(avalanche_id - 1)
            avalanche_reach = np.logical_and((bitmask & bit) > 0, evidence_valid)

            weighted_credit = np.zeros_like(evidence, dtype=np.float32)
            weighted_credit[avalanche_reach] = evidence[avalanche_reach] / overlap_safe[avalanche_reach]

            propagated_effective_cells = float(np.sum(weighted_credit))
            propagated_raw_cells = int(np.count_nonzero(avalanche_reach))
            coverage_index = propagated_effective_cells / float(start_cells)

            if coverage_raster is not None:
                # If two PRA start zones overlap, preserve the strongest index.
                coverage_raster[start_mask] = np.maximum(coverage_raster[start_mask], np.float32(coverage_index))

            stats_rows.append(
                {
                    "basin": f"Basin{basin_id}",
                    "avalanche_id": avalanche_id,
                    "start_cells": start_cells,
                    "propagated_raw_cells": propagated_raw_cells,
                    "propagated_effective_cells": round(propagated_effective_cells, 6),
                    "coverage_index": round(coverage_index, 6),
                }
            )

    if ref_profile is None or coverage_raster is None:
        raise RuntimeError("No compatible basins/res_* data found to compute coverage")

    _ensure_dir(out_raster.parent)
    out_profile = ref_profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=DEFAULT_OUTPUT_NODATA, compress="deflate")
    with rasterio.open(out_raster, "w", **out_profile) as dst:
        dst.write(coverage_raster.astype(np.float32, copy=False), 1)

    _ensure_dir(out_stats.parent)
    with out_stats.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "basin",
                "avalanche_id",
                "start_cells",
                "propagated_raw_cells",
                "propagated_effective_cells",
                "coverage_index",
            ],
        )
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    return [out_raster, out_stats]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute overlap-aware starting-zone coverage index from Flow-Py outputs.",
    )
    parser.add_argument(
        "--definitive-layers-dir",
        default="outputs/Definitive_Layers",
        help="Definitive layers folder containing BasinX/Star_propagating_Ending_Zones",
    )
    parser.add_argument(
        "--flowpy-root",
        default="outputs/Flow-Py",
        help="Flow-Py root with pra_basin_X/res_* outputs",
    )
    parser.add_argument(
        "--out-raster",
        default=None,
        help="Output raster path (default: <definitive>/5_StartingZones_Coverages.tif)",
    )
    parser.add_argument(
        "--out-stats",
        default=None,
        help="Output CSV path (default: <definitive>/5_StartingZones_Coverages_stats.csv)",
    )
    parser.add_argument(
        "--min-evidence-threshold",
        type=float,
        default=0.05,
        help="Minimum combined evidence score to consider a propagated cell (default: 0.05)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_root = Path(__file__).resolve().parents[1]

    definitive = Path(args.definitive_layers_dir).expanduser()
    if not definitive.is_absolute():
        definitive = (app_root / definitive).resolve()

    flowpy = Path(args.flowpy_root).expanduser()
    if not flowpy.is_absolute():
        flowpy = (app_root / flowpy).resolve()

    if args.out_raster is None:
        out_raster = definitive / "5_StartingZones_Coverages.tif"
    else:
        out_raster = Path(args.out_raster).expanduser()
        if not out_raster.is_absolute():
            out_raster = (app_root / out_raster).resolve()

    if args.out_stats is None:
        out_stats = definitive / "5_StartingZones_Coverages_stats.csv"
    else:
        out_stats = Path(args.out_stats).expanduser()
        if not out_stats.is_absolute():
            out_stats = (app_root / out_stats).resolve()

    outputs = compute_starting_zones_coverage(
        definitive_layers_dir=definitive,
        flowpy_root=flowpy,
        out_raster_path=out_raster,
        out_stats_csv=out_stats,
        min_evidence_threshold=args.min_evidence_threshold,
    )

    print("Done. Starting-zone coverage outputs:")
    for p in outputs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
