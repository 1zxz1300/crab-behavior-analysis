# Crab Behavioral Analysis Scripts

This repository contains the custom scripts used for SLEAP-based preprocessing, VAME-based motif analysis, community-level transition analysis, and visualization in a study of swimming crab behavior.

The workflow in this repository is intended to improve reproducibility by sharing the analysis scripts that were developed for this study. The VAME framework used in the analysis is open source; only the custom scripts and the minimal source files used for reproduction are included here.

## Repository structure

```text
crab_behavior_analysis/
├── README.md
├── LICENSE
├── scripts/
│   ├── preprocess_sleap_for_vame.py
│   ├── motif_metrics_rg_lme.py
│   ├── community_usage.py
│   ├── community_segment_usage_summary.py
│   ├── community_transition_probability.py
│   ├── compute_behavioral_transition_entropy.py
│   ├── prepare_community_edges_for_gephi.py
│   ├── expert_vame_confusion_matrix.py
│   └── plot_expert_vame_timeline.py
└── vame/
    ├── __init__.py
    ├── align_egocentrical.py
    ├── create_training.py
    ├── evaluate.py
    ├── pose_segmentation.py
    └── tree_hierarchy.py
```

> Note: the `vame/` folder is only needed if you decide to include the VAME-related source files used for reproduction. If you prefer, you may keep only the custom scripts in `scripts/` and cite the public VAME repository in the manuscript.

## Requirements

- Python 3.9 or later
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- openpyxl
- xlsxwriter

You can install the dependencies with:

```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl xlsxwriter
```

If you prefer a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy pandas scipy matplotlib seaborn openpyxl xlsxwriter
```

## Data availability

- Raw videos are not included in this repository.
- Only scripts, documentation, and optionally small example files should be uploaded.
- If you want to provide example inputs, place small anonymized examples in a separate folder such as `examples/`.

## VAME workflow note

VAME is an open-source framework for behavioral motif analysis. The workflow used in this study follows the public VAME pipeline for pose embedding, motif segmentation, and hierarchical motif analysis. The VAME source code itself is publicly available and therefore does not need to be duplicated here; this repository only includes the custom scripts and any modified source files used for reproducibility.

## Script descriptions

### SLEAP preprocessing
- `preprocess_sleap_for_vame.py`  
  Cleans SLEAP-exported keypoint CSV files for downstream VAME analysis by masking low-confidence points, removing implausible jumps, interpolating short gaps, and smoothing trajectories.

### Motif-level metrics
- `motif_metrics_rg_lme.py`  
  Computes motif-level movement metrics, including radius of gyration (Rg) and relative limb movement energy (LME_rel), from SLEAP keypoint time series.

### Community usage and aggregation
- `community_usage.py`  
  Aggregates motif usage into predefined community-level usage across replicate sheets.

- `community_segment_usage_summary.py`  
  Summarizes community usage by segment and treatment, producing per-community tables.

### Community transition analysis
- `community_transition_probability.py`  
  Converts motif-level transition probabilities into community-level transition matrices and averages results across replicates.

- `prepare_community_edges_for_gephi.py`  
  Converts a community transition matrix into a Gephi-compatible edge table after filtering weak edges and rescaling weights.

### Entropy analysis
- `compute_behavioral_transition_entropy.py`  
  Calculates overall behavioral transition entropy and state-wise conditional entropies from community usage and transition matrices.

### Validation and visualization
- `expert_vame_confusion_matrix.py`  
  Compares expert framewise labels with VAME labels and generates row-normalized confusion heatmaps.

- `plot_expert_vame_timeline.py`  
  Plots the framewise label timeline for expert annotations and VAME predictions.

## Typical workflow

1. Run `preprocess_sleap_for_vame.py` on SLEAP CSV exports.
2. Use the cleaned keypoint data in the VAME analysis pipeline.
3. Compute motif-level metrics with `motif_metrics_rg_lme.py`.
4. Summarize motif usage and community usage with `community_usage.py` and `community_segment_usage_summary.py`.
5. Compute community transition matrices with `community_transition_probability.py`.
6. Estimate behavioral transition entropy with `compute_behavioral_transition_entropy.py`.
7. Generate downstream figures and validation plots with `prepare_community_edges_for_gephi.py`, `expert_vame_confusion_matrix.py`, and `plot_expert_vame_timeline.py`.

## Reproducibility notes

- The scripts in this repository were written for the analysis reported in the manuscript.
- Paths inside the scripts should be adjusted to match your local file structure.
- If you use the scripts with your own data, make sure the column names and file formats match the expectations described in each script header.
- The exact VAME version used in the analysis should be reported in the manuscript and/or in the repository release notes if applicable.

## Citation

If you use these scripts in your own work, please cite the manuscript and the public VAME framework used in the analysis.

## License

This repository is released under the terms of the license included in `LICENSE`.
