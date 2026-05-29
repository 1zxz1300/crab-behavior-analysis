
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
