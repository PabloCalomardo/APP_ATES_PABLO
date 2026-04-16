#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio


DEFAULT_OUTPUT_NODATA = -9999.0
EXPOSURE_MODES = ("zdelta_cellcount", "zdelta", "cellcount")


def normalize_exposure_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in EXPOSURE_MODES:
        raise ValueError(
            f"Unsupported exposure mode: {mode}. Valid values: {', '.join(EXPOSURE_MODES)}"
        )
    return normalized


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
    mode: str = "zdelta_cellcount",
    output_nodata: float = DEFAULT_OUTPUT_NODATA,
) -> np.ndarray:
    """Compute overhead exposure from cell_count, z_delta, or both normalized layers."""
    mode = normalize_exposure_mode(mode)

    if cell_count is not None:
        shape = cell_count.shape
    elif z_delta is not None:
        shape = z_delta.shape
    else:
        raise ValueError("At least one input raster array is required")

    out = np.full(shape, output_nodata, dtype=np.float32)

    if mode == "cellcount":
        if cell_count is None or cell_count_valid is None:
            raise ValueError("cell_count raster is required for mode=cellcount")
        cell_count_scaled = minmax_scale_0_100(cell_count, cell_count_valid)
        out[cell_count_valid] = cell_count_scaled[cell_count_valid]
        return out

    if mode == "zdelta":
        if z_delta is None or z_delta_valid is None:
            raise ValueError("z_delta raster is required for mode=zdelta")
        z_delta_scaled = minmax_scale_0_100(z_delta, z_delta_valid)
        out[z_delta_valid] = z_delta_scaled[z_delta_valid]
        return out

    if cell_count is None or cell_count_valid is None or z_delta is None or z_delta_valid is None:
        raise ValueError("cell_count and z_delta rasters are required for mode=zdelta_cellcount")

    cell_count_scaled = minmax_scale_0_100(cell_count, cell_count_valid)
    z_delta_scaled = minmax_scale_0_100(z_delta, z_delta_valid)
    valid_both = cell_count_valid & z_delta_valid
    out[valid_both] = (cell_count_scaled[valid_both] + z_delta_scaled[valid_both]) / 2.0
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
    cell_count_path: str | Path | None,
    z_delta_path: str | Path | None,
    output_path: str | Path,
    mode: str = "zdelta_cellcount",
    nodata: float = DEFAULT_OUTPUT_NODATA,
) -> Path:
    """Compute and write overhead exposure raster from selected input raster paths."""
    mode = normalize_exposure_mode(mode)

    cell_count = None
    cell_count_valid = None
    cell_count_profile = None
    z_delta = None
    z_delta_valid = None
    z_delta_profile = None

    if mode in ("zdelta_cellcount", "cellcount"):
        if cell_count_path is None:
            raise ValueError("cell_count_path is required for selected mode")
        cell_count, cell_count_valid, cell_count_profile = read_single_band_raster(cell_count_path)

    if mode in ("zdelta_cellcount", "zdelta"):
        if z_delta_path is None:
            raise ValueError("z_delta_path is required for selected mode")
        z_delta, z_delta_valid, z_delta_profile = read_single_band_raster(z_delta_path)

    if cell_count_profile is not None and z_delta_profile is not None:
        validate_rasters_aligned(cell_count_profile, z_delta_profile)

    profile_ref = cell_count_profile if cell_count_profile is not None else z_delta_profile
    if profile_ref is None:
        raise ValueError("No input profile available to write output raster")

    exposure = compute_overhead_exposure(
        cell_count=cell_count,
        cell_count_valid=cell_count_valid,
        z_delta=z_delta,
        z_delta_valid=z_delta_valid,
        mode=mode,
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
        description="Compute overhead exposure from cell_count, z_delta, or both.",
    )
    parser.add_argument("--cell-count", required=False, help="Path to cell_count raster")
    parser.add_argument("--z-delta", required=False, help="Path to z_delta raster")
    parser.add_argument("--output", required=True, help="Output overhead exposure raster path")
    parser.add_argument(
        "--mode",
        choices=EXPOSURE_MODES,
        default="zdelta_cellcount",
        help="Exposure mode: cellcount, zdelta, or zdelta_cellcount (default)",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=DEFAULT_OUTPUT_NODATA,
        help=f"Output nodata value (default: {DEFAULT_OUTPUT_NODATA})",
    )
    args = parser.parse_args()
    mode = normalize_exposure_mode(args.mode)
    if mode in ("zdelta_cellcount", "cellcount") and not args.cell_count:
        parser.error("--cell-count is required for mode=cellcount or mode=zdelta_cellcount")
    if mode in ("zdelta_cellcount", "zdelta") and not args.z_delta:
        parser.error("--z-delta is required for mode=zdelta or mode=zdelta_cellcount")
    return args


def main() -> None:
    args = parse_args()
    out_path = compute_overhead_exposure_from_files(
        cell_count_path=args.cell_count,
        z_delta_path=args.z_delta,
        output_path=args.output,
        mode=args.mode,
        nodata=args.nodata,
    )

    print(f"Done. Overhead exposure raster written to: {out_path}")


if __name__ == "__main__":
    main()
