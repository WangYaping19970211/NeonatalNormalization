# Neonatal Normalization

A neonatal-to-adult normalization pipeline designed to minimize systematic biases in multi-step registration, enabling reliable cross-age comparisons in large-scale neuroimaging studies.

## Background

In large-scale neuroimaging studies — especially those involving longitudinal analysis — a critical first step is to transform individual brain images into a common template space. This standardization enables meaningful group-level comparisons and statistical analyses.

For **adult** brains, this is typically achieved through **one-step** direct normalization to a standard template. This approach, however, does not generalize well to **neonatal** brains: due to substantial anatomical differences between neonatal and adult brains, direct normalization to an adult template often fails or yields suboptimal results.

A commonly adopted solution is a **multi-step** strategy: first normalize neonatal brains to age-specific templates (e.g., the [UNC BCP 4D atlas](https://www.nitrc.org/projects/uncbcp_4d_atlas/)), then progressively map them to older templates in a stepwise manner until reaching adult template space. This has been shown to improve registration performance.

However, the multi-step approach may introduce **systematic biases**. If not properly addressed, these biases can propagate into downstream analyses and be mistakenly interpreted as biological signals.

## Objective

Develop a neonatal normalization pipeline that:

- **minimizes systematic biases** introduced by multi-step registration, and
- **achieves consistency with adult normalization**,

thereby enabling more accurate and reliable cross-age comparisons.

## Pipeline Overview

1. **Skull stripping** — extract brain tissue using iBEAT 2.0.
2. **Age-specific normalization** — register each subject to the corresponding age-specific template (UNC BCP 4D atlas).
3. **Visual quality check** — inspect skull-stripping and registration results.
4. **Stepwise mapping to adult space** — progressively map to older templates up to the adult template.

## Repository Structure

```
NeonatalNormalization/
├── Notebook/
│   └── SickKids_tutorial.ipynb     # Walkthrough / tutorial
├── Script/
│   ├── pre01_iBEAT/                # iBEAT 2.0 preprocessing (skull stripping)
│   └── tpl_xfm_build.py            # Template transform construction
└── templates/                      # Age-specific & adult templates (BCP atlas)
```