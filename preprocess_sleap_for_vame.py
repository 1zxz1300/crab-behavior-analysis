
#!/usr/bin/env python3
"""
Preprocess SLEAP-exported keypoint CSV for downstream VAME analysis.

Workflow
--------
1) Read a SLEAP CSV export with columns:
   frame_idx, instance.score, and for each keypoint:
   <kp>.x, <kp>.y, <kp>.score

2) Mask low-confidence keypoints:
   - If keypoint score < score_threshold, set x/y/score to NaN.

3) Remove implausible jumps (outliers):
   - For each keypoint, compute frame-to-frame displacement.
   - Use a robust median + MAD rule to flag sudden jumps.
   - Flagged points are set to NaN.

4) Interpolate missing short gaps and smooth trajectories:
   - First interpolate short interior gaps linearly (up to max_linear_gap frames).
   - Then fill any remaining internal gaps using shape-preserving PCHIP interpolation.
   - Finally smooth x/y with a Savitzky–Golay filter.

5) Write a cleaned CSV with the same wide format as the input.

Notes
-----
- This script is a reconstruction of the preprocessing pipeline described in the
  manuscript. If you have the original parameter values, adjust the defaults below.
- Coordinates are kept in the original pixel space. No additional body-scale
  normalization is applied here because VAME centers/alines poses internally.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception as exc:  # pragma: no cover
    savgol_filter = None

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None


def infer_keypoints(columns: Sequence[str]) -> List[str]:
    """Infer keypoint names from <kp>.x / <kp>.y / <kp>.score columns."""
    kps = []
    seen = set()
    for c in columns:
        if c.endswith(".x"):
            kp = c[:-2]
            if kp not in seen and f"{kp}.y" in columns:
                kps.append(kp)
                seen.add(kp)
    return kps


def robust_mad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to be comparable to std."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def mask_low_confidence(df: pd.DataFrame, keypoints: Sequence[str], score_threshold: float) -> Dict[str, int]:
    masked = {}
    for kp in keypoints:
        score_col = f"{kp}.score"
        x_col = f"{kp}.x"
        y_col = f"{kp}.y"
        if score_col not in df.columns:
            continue
        low = df[score_col].notna() & (df[score_col] < score_threshold)
        masked[kp] = int(low.sum())
        df.loc[low, [x_col, y_col, score_col]] = np.nan
    return masked


def mask_outlier_jumps(
    df: pd.DataFrame,
    keypoints: Sequence[str],
    mad_k: float = 6.0,
    min_pairs: int = 20,
    iterations: int = 2,
) -> Dict[str, int]:
    """
    Flag implausible frame-to-frame jumps using a robust median + MAD rule.
    The current point is marked as NaN when the displacement from the previous
    valid point is an extreme outlier.
    """
    outlier_counts: Dict[str, int] = {kp: 0 for kp in keypoints}

    for _ in range(iterations):
        for kp in keypoints:
            x_col, y_col, s_col = f"{kp}.x", f"{kp}.y", f"{kp}.score"
            x = df[x_col].to_numpy(dtype=float)
            y = df[y_col].to_numpy(dtype=float)

            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < min_pairs + 1:
                continue

            # Frame-to-frame displacement only where both frames are valid.
            disp = np.full(len(df), np.nan, dtype=float)
            prev_valid_idx = None
            for i in range(len(df)):
                if not valid[i]:
                    continue
                if prev_valid_idx is not None:
                    disp[i] = float(np.hypot(x[i] - x[prev_valid_idx], y[i] - y[prev_valid_idx]))
                prev_valid_idx = i

            d = disp[np.isfinite(disp)]
            if len(d) < min_pairs:
                continue

            med = float(np.median(d))
            mad = float(robust_mad(d))
            if not np.isfinite(mad) or mad == 0:
                # Fallback: very conservative threshold if motion is extremely stable.
                threshold = med * 10.0 + 1.0
            else:
                threshold = med + mad_k * mad

            jump_idx = np.where(np.isfinite(disp) & (disp > threshold))[0]
            if len(jump_idx) > 0:
                outlier_counts[kp] += int(len(jump_idx))
                df.loc[jump_idx, [x_col, y_col, s_col]] = np.nan

    return outlier_counts


def interpolate_with_pchip(
    s: pd.Series,
    max_linear_gap: int = 5,
) -> pd.Series:
    """
    First linearly interpolate short interior gaps.
    Then use PCHIP to fill any remaining internal gaps.
    """
    s0 = s.copy()

    # Stage 1: short-gap linear interpolation.
    s1 = s0.interpolate(method="linear", limit=max_linear_gap, limit_area="inside")

    # Stage 2: fill remaining internal gaps with PCHIP if available.
    if s1.isna().any():
        if PchipInterpolator is None:
            # Fallback to a second linear pass if SciPy is unavailable.
            s1 = s1.interpolate(method="linear", limit_direction="both")
            return s1

        x = np.arange(len(s1), dtype=float)
        y = s1.to_numpy(dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            return s1

        f = PchipInterpolator(x[ok], y[ok], extrapolate=False)
        y2 = y.copy()
        miss = ~ok
        if miss.any():
            y2[miss] = f(x[miss])

        s2 = pd.Series(y2, index=s1.index)

        # Fill boundary NaNs conservatively using nearest observed values.
        s2 = s2.ffill().bfill()
        return s2

    return s1


def smooth_series(y: pd.Series, window: int = 7, polyorder: int = 2) -> pd.Series:
    """Smooth a 1D series with Savitzky-Golay filter."""
    if savgol_filter is None:
        return y

    arr = y.to_numpy(dtype=float)
    if np.isnan(arr).any():
        # Should not happen after interpolation, but keep safe.
        arr = pd.Series(arr).interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)

    n = len(arr)
    if n < 3:
        return y

    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 3:
        return y
    if win <= polyorder:
        win = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
        if win > n:
            return y

    sm = savgol_filter(arr, window_length=win, polyorder=polyorder, mode="interp")
    return pd.Series(sm, index=y.index)


def preprocess(
    input_csv: Path,
    output_csv: Path,
    score_threshold: float = 0.5,
    max_linear_gap: int = 5,
    outlier_mad_k: float = 6.0,
    smooth_window: int = 7,
    smooth_polyorder: int = 2,
) -> None:
    df = pd.read_csv(input_csv)

    if "frame_idx" not in df.columns:
        raise ValueError("Input CSV must contain a 'frame_idx' column.")

    keypoints = infer_keypoints(df.columns)
    if not keypoints:
        raise ValueError("No keypoints found. Expected columns like '<kp>.x', '<kp>.y', '<kp>.score'.")

    # Ensure frame order and complete frame index coverage.
    df = df.sort_values("frame_idx").reset_index(drop=True)
    full_idx = np.arange(int(df["frame_idx"].min()), int(df["frame_idx"].max()) + 1)
    df = df.set_index("frame_idx").reindex(full_idx)
    df.index.name = "frame_idx"
    df = df.reset_index()

    # Carry forward / back instance score only if you want a continuous frame-level quality indicator.
    # Here we keep it as-is and only interpolate keypoint coordinates.
    # Mask low-confidence points.
    low_counts = mask_low_confidence(df, keypoints, score_threshold)

    # Mask implausible jumps.
    outlier_counts = mask_outlier_jumps(df, keypoints, mad_k=outlier_mad_k, iterations=2)

    # Interpolate and smooth x/y for each keypoint.
    interp_reports = {}
    for kp in keypoints:
        for coord in ("x", "y"):
            col = f"{kp}.{coord}"
            s = df[col]
            s_interp = interpolate_with_pchip(s, max_linear_gap=max_linear_gap)
            s_smooth = smooth_series(s_interp, window=smooth_window, polyorder=smooth_polyorder)
            df[col] = s_smooth
        interp_reports[kp] = int(df[[f"{kp}.x", f"{kp}.y"]].isna().any(axis=1).sum())

    # Reorder columns to match original layout.
    original_cols = [c for c in pd.read_csv(input_csv, nrows=0).columns]
    remaining_cols = [c for c in df.columns if c not in original_cols]
    ordered_cols = [c for c in original_cols if c in df.columns] + remaining_cols
    df = df[ordered_cols]

    # Save.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Write a companion log for reproducibility.
    log_path = output_csv.with_suffix(".log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Input: {input_csv}\n")
        f.write(f"Output: {output_csv}\n")
        f.write(f"Keypoints ({len(keypoints)}): {', '.join(keypoints)}\n")
        f.write(f"Score threshold: {score_threshold}\n")
        f.write(f"Max linear interpolation gap: {max_linear_gap} frames\n")
        f.write(f"Outlier rule: median + {outlier_mad_k} * MAD on frame-to-frame displacement\n")
        f.write(f"Savgol window: {smooth_window}, polyorder: {smooth_polyorder}\n")
        f.write("Masked low-confidence counts per keypoint:\n")
        for kp, n in low_counts.items():
            f.write(f"  {kp}: {n}\n")
        f.write("Masked jump-outlier counts per keypoint:\n")
        for kp, n in outlier_counts.items():
            f.write(f"  {kp}: {n}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess SLEAP CSV keypoints for VAME.")
    p.add_argument("--input", required=True, type=Path, help="Input SLEAP CSV file.")
    p.add_argument("--output", required=True, type=Path, help="Output cleaned CSV file.")
    p.add_argument("--score-threshold", type=float, default=0.5, help="Low-confidence threshold.")
    p.add_argument("--max-linear-gap", type=int, default=5, help="Max gap (frames) for linear interpolation.")
    p.add_argument("--outlier-mad-k", type=float, default=6.0, help="Robust threshold factor for jump outliers.")
    p.add_argument("--smooth-window", type=int, default=7, help="Savitzky-Golay window length (odd number).")
    p.add_argument("--smooth-polyorder", type=int, default=2, help="Savitzky-Golay polynomial order.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(
        input_csv=args.input,
        output_csv=args.output,
        score_threshold=args.score_threshold,
        max_linear_gap=args.max_linear_gap,
        outlier_mad_k=args.outlier_mad_k,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )


if __name__ == "__main__":
    main()
