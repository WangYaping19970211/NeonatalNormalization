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

## Pipeline

`Script/neonorm.py` works on one visit at a time, from neonates to adults:

1. **Preprocessing** (`preprocess`): orientation (`fslreorient2std`, only if needed), brain mask (age-specific SynthStrip: below 12 months the union of the default and the infant model; or an own mask), skull stripping and N4 bias correction -> `<prefix>_T1w_brain_n4.nii.gz`.
2. **Direct normalisation** (`direct_normalize`): `antsRegistrationSyN.sh` (T1 only) to the BCP templates of the age (UNC BCP 4D atlas: 0-12, 15, 18, 21, 24, 36, 48, 60 months) and to MNI (MNI152NLin2009cSym). Its QC (`direct_norm_qc`): T1 correlation, brain and cerebellum Dice and HD95, minimum Jacobian and a figure per registration; new visits are flagged against the QC of this project (`direct_qc_flags`).
3. **To MNI** (`visit_to_mni`): the visit to MNI directly, or along a chain of BCP templates (tpl2tpl: template-to-template registrations; avgsubj: the average of the visits' registrations per edge), as one composed field: the T1, cerebellum, brain mask and log Jacobian determinant at MNI.

Steps 1-2 are what a site runs on its own data; step 3 can follow later and elsewhere, as it needs only the outputs of steps 1-2 and the template edges.

## Tutorials

| `docs/tutorials/` | what |
|---|---|
| `1.quickstart_direct_normalize.ipynb` | steps 1-2: one visit in Python, a dataset on a Slurm cluster; section 4: SynthStrip only, compared with a manual mask |
| `1.quickstart_qc.ipynb` | QC of the registrations: the metrics by age against the project, flags, review table |
| `2.full_pipeline.ipynb` | step 3: to MNI by each way |

The example visits (`example/`) are two infants of the open Baby Open Brains dataset (CC-BY-4.0; source and citation in `example/README.md`).

## Installation

On a Linux server or a Mac, with conda and FSL (`FSLDIR` set) and, if there, FreeSurfer (`FREESURFER_HOME` set):

```bash
bash Script/setup_neonorm.sh /path/to/toolbox/neonorm [user@server:/path/to/NeonatalNormalization/templates]
source /path/to/toolbox/neonorm/neonorm_env.sh
```

The script sets up the conda environment `neonorm` (`environment.yml`), ANTs and the SynthStrip models, checks the templates against `Script/data/templates_md5.txt`, writes `neonorm_env.sh` (the settings: `source` it in the shell and in job scripts) and compares the versions with the validated ones (below). Steps already done are skipped.

- Without FreeSurfer (e.g. on a Mac), it installs `mri_synthstrip` of FreeSurfer 8.0.0 on its own (conda environment `neonorm-synthstrip`) and the default model from the SynthStrip website.
- Without the templates source, it checks the templates in git (BCP, MNI): enough for steps 1-2. Step 3 also needs the template edges (`templates/tpl_xfm_build/`, 16 GB, not in git).
- On an older Linux, choose another ANTs build with `ANTS_PLATFORM` (e.g. `centos7-X64-gcc`).

| Software | Version validated | Used for |
|---|---|---|
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/) | 6.0.6.2 | `fslreorient2std` |
| [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) | 8.0.0 | `mri_synthstrip` |
| [ANTs](https://github.com/ANTsX/ANTs) | 2.6.2 (release binaries) | N4, registrations, transforms, Jacobian |
| SynthStrip models | `synthstrip.1.pt` (md5 `27f86a16f396ac6db3023c2f70b97ae1`), `synthstrip.infant.1.pt` (md5 `37a6e0fa34adec4ab4ffba22de551595`, in this repository) | brain mask |
| Python packages | `environment.yml` (Python 3.12.2, ANTsPy 0.6.1, SUITPy 2.0.1, ...) | rigid T2-to-T1 registration, N4, SUITPy cerebellum (QC) |

Without the setup script, neonorm finds the tools through `FSLDIR`, `FREESURFER_HOME`, `ANTSPATH` or the `PATH`, and the data through `NEONORM_SYNTHSTRIP_MODELS` (a folder with both models), `NEONORM_BCP_TEMPLATES` and `NEONORM_MNI_TEMPLATE`; `NEONORM_NTHREADS` sets the CPU threads (in a Slurm job: its CPUs). Most have a command-line option too (`python Script/neonorm.py -h`).

## Command line

```bash
python Script/neonorm.py --check                                         # the setup: what was found, what is missing
python Script/neonorm.py --preprocess sub-001_T1w.nii.gz 4.5 out/sub-001  # step 1
python Script/neonorm.py --direct_normalize out/sub-001/sub-001_T1w_brain_n4.nii.gz 4.5 out/sub-001  # step 2
python Script/neonorm.py --direct_norm_qc out/sub-001/sub-001_direct_normalize.json                  # its QC
```

- `age_months` is the age at scan in months: it chooses the SynthStrip models and the templates. From a postmenstrual age in weeks: `neonorm.age_months_from_pma(pma_weeks)` (0 before 40 weeks).
- Step 1 (`--preprocess` is optional: step 1 is the default): `--t2` (the T2 of the session, optional), `--prefix` (default: the T1 name without `_T1w.nii.gz`), `--brain_mask` (an own mask on the T1 as acquired instead of SynthStrip; give the run its own prefix, e.g. `<id>_manual_mask`).
- Step 2: `--templates` (default: the nearest BCP template and MNI; the tutorials use `direct_templates(age)`, which adds the other templates of the visit's age window), `--no_rotation_init` (the centre-of-mass start instead of a rotation search).
- In Python, and on a Slurm cluster (job arrays for many visits; job scripts and logs in `<out_dir>/log/`): the tutorials.

## Outputs

In `out_dir`; the inputs are only read.

| File | Step |
|---|---|
| `<prefix>_T1w_brain_mask.nii.gz`, `<prefix>_T1w_brain_n4.nii.gz` (and `_T2w_brain_n4.nii.gz`) | 1: brain mask; skull-stripped, bias-corrected T1 (and T2) |
| `<prefix>_T2w_rigid2T1.nii.gz`, `_0GenericAffine.mat` | 1, with a T2: the T2 on the T1 grid and its transform |
| `<prefix>_preprocess.json` | 1: inputs, software (versions, model checksums), outputs |
| `<prefix>_direct_to-<template>_0GenericAffine.mat`, `_1Warp.nii.gz` | 2: visit -> template (apply as `[warp, affine]`) |
| `<prefix>_direct_to-<template>_1InverseWarp.nii.gz` | 2: template -> visit (apply as `[affine, inverse warp]`, affine inverted) |
| `<prefix>_direct_to-<template>_Warped.nii.gz` | 2: the N4 T1 in template space |
| `<prefix>_direct_normalize.json` | 2: inputs, software, the `antsRegistration` calls, outputs per template |
| `<prefix>_direct_to-<template>_qc.png`, `<prefix>_direct_normalize_qc.json`, `<prefix>_cerebellum_suitpy_from-brain-N4_dseg.nii.gz` | QC of step 2: figures, metrics, the reference cerebellum of the visit |
| `<prefix>_<way>[-<path>]_to-MNI152NLin2009cSym_T1w.nii.gz`, `_cerebellum_dseg.nii.gz`, `_brain_mask.nii.gz`, `_logjd.nii.gz`, `.json` | 3: T1, cerebellum, brain mask and log Jacobian determinant at MNI; the transforms and ANTs commands |
| `log/<prefix>_*.sh`, `.out` | Slurm: job scripts and their full output |

## Repository structure

```
NeonatalNormalization/
├── environment.yml                  # Python environment of neonorm
├── docs/tutorials/                  # the tutorials (above) and example/ (two Baby Open Brains infants)
├── Script/
│   ├── neonorm.py                   # the pipeline (command line and Python functions)
│   ├── setup_neonorm.sh             # one-time setup (above)
│   ├── notebook_clean_filter.py     # git filter: tutorials committed without outputs and local paths (set up: see the file)
│   ├── data/                        # QC reference of the project (no IDs), md5 of the templates
│   └── skullstrip/synthstrip/model/synthstrip.infant.1.pt   # SynthStrip infant model (not in FreeSurfer)
└── templates/                       # BCP 4D atlas (T1, mask, SUITPy cerebellum) and MNI152NLin2009cSym;
                                     #   the template edges (tpl_xfm_build/, 16 GB) via setup_neonorm.sh
```
