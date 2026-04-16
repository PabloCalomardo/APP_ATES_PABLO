#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio


DEFAULT_OUTPUT_NODATA = -9999.0
DEFAULT_CELLCOUNT_WEIGHT = 0.5


def validate_cellcount_weight(cellcount_weight: float) -> float:
    weight = float(cellcount_weight)
    if (weight < 0.0 or weight > 1.0) and not np.isclose(weight, 2.0):
        raise ValueError(
            f"Invalid cell_count weight: {cellcount_weight}. Expected [0, 1] or 2 (max mode)."
        )
    return weight


def read_single_band_raster(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a single-band raster and return data, valid mask, and profile.

    The valid mask is True where data is valid (not nodata).
    """
    raster_path = Path(path).expanduser().resolve()
    with rasterio.open(raster_path) as src:
        if src.count != 1:
            raise ValueError(f"Raster must be single-band: {raster_path}")

        band = src.read(1, masked=True)
        data = np.asarray(band.data)
        valid_mask = ~np.asarray(band.mask)
        profile = src.profile.copy()

    return data, valid_mask, profile


def validate_rasters_aligned(profile_a: dict, profile_b: dict) -> None:
    """Validate that two rasters share grid geometry and CRS."""
    checks: list[Tuple[str, object, object]] = [
        ("width", profile_a.get("width"), profile_b.get("width")),
        ("height", profile_a.get("height"), profile_b.get("height")),
        ("transform", profile_a.get("transform"), profile_b.get("transform")),
        ("crs", profile_a.get("crs"), profile_b.get("crs")),
    ]

    mismatches = [name for name, a, b in checks if a != b]
    if mismatches:
        details = ", ".join(mismatches)
        raise ValueError(
            "Input rasters are not aligned (different "
            f"{details}). Ensure same size, transform, and CRS."
        )


def minmax_scale_0_100(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Scale valid pixels with min-max normalization to [0, 100]."""
    scaled = np.full(data.shape, np.nan, dtype=np.float32)

    if not np.any(valid_mask):
        return scaled

    valid_values = data[valid_mask].astype(np.float32, copy=False)
    vmin = np.min(valid_values)
    vmax = np.max(valid_values)

    if np.isclose(vmax, vmin):
        scaled[valid_mask] = 0.0
        return scaled

    scaled_values = ((valid_values - vmin) / (vmax - vmin)) * 100.0
    scaled[valid_mask] = scaled_values.astype(np.float32, copy=False)
    return scaled


def compute_overhead_exposure(
    cell_count: Optional[np.ndarray],
    cell_count_valid: Optional[np.ndarray],
    z_delta: Optional[np.ndarray],
    z_delta_valid: Optional[np.ndarray],
    cellcount_weight: float = DEFAULT_CELLCOUNT_WEIGHT,
    output_nodata: float = DEFAULT_OUTPUT_NODATA,
) -> np.ndarray:
    """Compute overhead exposure from normalized cell_count and z_delta layers.

    When cellcount_weight is 2, use max(cell_count_norm, z_delta_norm) per pixel.
    Otherwise use weighted average: w*cell_count_norm + (1-w)*z_delta_norm.
    """
    if cell_count is None or cell_count_valid is None:
        raise ValueError("cell_count raster is required")
    if z_delta is None or z_delta_valid is None:
        raise ValueError("z_delta raster is required")

    if cell_count.shape != z_delta.shape:
        raise ValueError("cell_count and z_delta arrays must have the same shape")

    w_cell = validate_cellcount_weight(cellcount_weight)

    out = np.full(cell_count.shape, output_nodata, dtype=np.float32)

    cell_count_scaled = minmax_scale_0_100(cell_count, cell_count_valid)
    z_delta_scaled = minmax_scale_0_100(z_delta, z_delta_valid)
    valid_both = cell_count_valid & z_delta_valid

    if np.isclose(w_cell, 2.0):
        out[valid_both] = np.maximum(cell_count_scaled[valid_both], z_delta_scaled[valid_both])
        return out

    w_zdelta = 1.0 - w_cell
    out[valid_both] = (w_cell * cell_count_scaled[valid_both]) + (w_zdelta * z_delta_scaled[valid_both])
    return out


def save_raster(path: str | Path, data: np.ndarray, profile_ref: dict, nodata: float) -> Path:
    """Save output raster preserving spatial metadata from reference profile."""
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile_ref.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="deflate",
    )

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(data.astype(np.float32, copy=False), 1)

    return out_path


def compute_overhead_exposure_from_files(
    cell_count_path: str | Path,
    z_delta_path: str | Path,
    output_path: str | Path,
    cellcount_weight: float = DEFAULT_CELLCOUNT_WEIGHT,
    nodata: float = DEFAULT_OUTPUT_NODATA,
) -> Path:
    """Compute and write weighted overhead exposure raster from input rasters."""
    cell_count, cell_count_valid, cell_count_profile = read_single_band_raster(cell_count_path)
    z_delta, z_delta_valid, z_delta_profile = read_single_band_raster(z_delta_path)

    validate_rasters_aligned(cell_count_profile, z_delta_profile)
    profile_ref = cell_count_profile

    exposure = compute_overhead_exposure(
        cell_count=cell_count,
        cell_count_valid=cell_count_valid,
        z_delta=z_delta,
        z_delta_valid=z_delta_valid,
        cellcount_weight=cellcount_weight,
        output_nodata=nodata,
    )

    return save_raster(
        path=output_path,
        data=exposure,
        profile_ref=profile_ref,
        nodata=nodata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute overhead exposure from cell_count and z_delta.",
    )
    parser.add_argument("--cell-count", required=True, help="Path to cell_count raster")
    parser.add_argument("--z-delta", required=True, help="Path to z_delta raster")
    parser.add_argument("--output", required=True, help="Output overhead exposure raster path")
    parser.add_argument(
        "--cellcount-weight",
        type=float,
        default=DEFAULT_CELLCOUNT_WEIGHT,
        help=(
            "Weight for normalized cell_count in final exposure [0..1] "
            "(default: 0.5). Use 2 for max mode: max(cell_count_norm, z_delta_norm)."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=DEFAULT_OUTPUT_NODATA,
        help=f"Output nodata value (default: {DEFAULT_OUTPUT_NODATA})",
    )
    args = parser.parse_args()
    try:
        validate_cellcount_weight(args.cellcount_weight)
    except ValueError as e:
        parser.error(str(e))
    return args


def main() -> None:
    args = parse_args()
    out_path = compute_overhead_exposure_from_files(
        cell_count_path=args.cell_count,
        z_delta_path=args.z_delta,
        output_path=args.output,
        cellcount_weight=args.cellcount_weight,
        nodata=args.nodata,
    )

    print(f"Done. Overhead exposure raster written to: {out_path}")


if __name__ == "__main__":
    main()
