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

`Script/neonorm.py`, one visit at a time, from neonates to adults, with steps optimised for the first months of life (tutorials: `docs/tutorials/1.quickstart_direct_normalize.ipynb`, steps 1-2 for one visit and a dataset, what a site runs on its own data, without waiting for any QC, and SynthStrip only with a manual brain mask compared; `docs/tutorials/1.quickstart_qc.ipynb`, the QC after the registrations, apart from the runs; `docs/tutorials/2.full_pipeline.ipynb`, step 3 after them):

1. **Preprocessing** (`preprocess()`; command line `neonorm.py <t1> <age> <out_dir>`), once per brain mask, in three parts: 1a orientation (`orient`): check whether the T1 (and T2) are in the standard orientation, reorient with `fslreorient2std` if not (temporary files only), with a T2 the rigid T2 on the T1 grid; 1b brain mask: SynthStrip (`synthstrip_age_specific`; below 12 months the union of the default and the infant model, from 12 months the default model) or an own mask (`own_brain_mask`, `--brain_mask`: on the T1 as acquired, reoriented like the T1, made binary); 1c skull stripping + N4 (`head2brain_and_n4correction`) -> `<prefix>_T1w_brain_n4.nii.gz`. No QC in step 1: errors of the brain mask show in the QC of the registrations (`direct_norm_qc`).
2. **Direct normalisation** — `antsRegistrationSyN.sh` (ANTs, default parameters, T1 only) of the skull-stripped T1 to the age-specific template (UNC BCP 4D atlas, nearest atlas month: 0-12, 15, 18, 21, 24, 36, 48, 60) and directly to MNI space (MNI152NLin2009cSym). Evaluation phase: both for every age (`AGE_TEMPLATE_BELOW_MONTHS = inf`); planned: the age-specific template only below 6 months. `direct_normalize(<prefix>_T1w_brain_n4.nii.gz, ...)`: the registrations of the N4 T1 of step 1. The QC, apart (tutorial: `1.quickstart_qc.ipynb`): `direct_norm_qc(<prefix>_direct_normalize.json)`. Command line: `neonorm.py --direct_normalize <prefix>_T1w_brain_n4.nii.gz ...`, `neonorm.py --direct_norm_qc <json>`; Slurm: `submit_direct_normalize(..., qc=True)` (step 2, then the QC as a dependent job).
3. **To MNI** — `visit_to_mni()`: the visit to MNI by one way, direct (its MNI registration of step 2) or along a chain of BCP templates (tpl2tpl: template-to-template registrations; avgsubj: the average of the visits' registrations per edge), in three parts: 3a all transforms composed into one field on the MNI grid; 3b the T1, the reference cerebellum and the brain mask moved with it (one interpolation) and its log Jacobian determinant; 3c `mni_metrics` (T1 r, brain and cerebellum Dice and HD95 with MNI, mean log Jacobian). Slurm: `submit_visits_to_mni`.

## Repository Structure

```
NeonatalNormalization/
├── environment.yml                   # Python environment of neonorm
├── docs/
│   └── tutorials/
│       ├── 1.quickstart_direct_normalize.ipynb   # Steps 1-2, one visit and a dataset (Slurm), no QC in the way
│       ├── 1.quickstart_qc.ipynb                 # QC of the registrations (flags against the project), review table
│       ├── 2.full_pipeline.ipynb                 # Step 3 after them: to MNI by each way, using the outputs
│       └── example/                       # Example visits: two infants of the open Baby Open Brains dataset (CC-BY-4.0; README.md: source, citation); outputs in out/, not in git
├── Notebook/                         # Notebooks of the project (BCP, HCPD; not in the repository: NDA data in their outputs)
├── Script/
│   ├── neonorm.py                    # The pipeline (command line and Python functions)
│   ├── setup_neonorm.sh              # One-time setup of a server (environment, ANTs, models, templates, check)
│   ├── data/
│   │   ├── direct_qc_reference.tsv   # Reference of the registration QC of new visits (direct_qc_flags)
│   │   ├── direct_qc_reference_values.tsv  # Its registrations by age, no IDs (plot_direct_qc_by_age)
│   │   └── templates_md5.txt         # md5 of the template files neonorm reads (setup_neonorm.sh checks them)
│   └── skullstrip/synthstrip/model/synthstrip.infant.1.pt   # SynthStrip infant model (not in FreeSurfer)
└── templates/                        # BCP 4D atlas (T1, mask, SUITPy cerebellum) and MNI152NLin2009cSym; the template
                                      #   edges (tpl_xfm_build/, 16 GB) are copied by setup_neonorm.sh
```

## neonorm

### Software

| Software | Version validated | Used for |
|---|---|---|
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/) | 6.0.6.2 | `fslreorient2std` |
| [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) | 8.0.0 (`freesurfer-linux-centos8_x86_64-8.0.0-20250203`) | `mri_synthstrip` |
| [ANTs](https://github.com/ANTsX/ANTs) | 2.6.2 (release binaries) | steps 1c-3: `N4BiasFieldCorrection`, `antsApplyTransforms`, `CreateJacobianDeterminantImage`, `ImageMath`, `antsRegistrationSyN.sh` (which calls `antsRegistration` and `PrintHeader` from the same folder) |
| SynthStrip default model `synthstrip.1.pt` | md5 `27f86a16f396ac6db3023c2f70b97ae1` | included in FreeSurfer 8.0 (`$FREESURFER_HOME/models`) |
| SynthStrip infant model `synthstrip.infant.1.pt` | md5 `37a6e0fa34adec4ab4ffba22de551595` | **not included in FreeSurfer**: download it from the [SynthStrip website](https://synthstrip.io) and put it in one folder with `synthstrip.1.pt` |
| Python packages | `environment.yml` (ANTsPy 0.6.1, SUITPy 2.0.1 from its git commit, as it is not on PyPI; numpy 2.4.6, nibabel 5.3.2, ...) | rigid T2-to-T1 registration, N4 bias correction, SUITPy cerebellum for the QC |

```bash
conda env create -f environment.yml
conda activate neonorm
```

All of it on a new server in one go (conda environment, ANTs release binaries, SynthStrip models, templates checked against `Script/data/templates_md5.txt`, a settings file `neonorm_env.sh` to `source`, and the versions compared with the validated ones; FSL and FreeSurfer are set up before):

```bash
bash Script/setup_neonorm.sh /path/to/toolbox/neonorm user@server:/path/to/NeonatalNormalization/templates
source /path/to/toolbox/neonorm/neonorm_env.sh
```

### Settings for your computer

Nothing in the code needs to be changed. The tools are found on the `PATH` or through their usual environment variables; a command-line option overrides the environment variable.

| Environment variable | Option | Meaning | Default |
|---|---|---|---|
| `FSLDIR` | | FSL installation | `fslreorient2std` on the `PATH` |
| `FREESURFER_HOME` | | FreeSurfer installation | `mri_synthstrip` on the `PATH` |
| `NEONORM_SYNTHSTRIP_MODELS` | `--synthstrip_models` | folder with `synthstrip.1.pt` and `synthstrip.infant.1.pt` | `$FREESURFER_HOME/models` (no infant model there) |
| `ANTSPATH` | `--antspath` | ANTs `bin` folder with `antsRegistrationSyN.sh`, `N4BiasFieldCorrection`, `ImageMath`; used before the `PATH`, so it picks one ANTs version on a computer with several | `antsRegistrationSyN.sh` on the `PATH` |
| `NEONORM_NTHREADS` | `--threads` | CPU threads of SynthStrip and ANTs | `SLURM_CPUS_PER_TASK`, else the tool defaults |
| `NEONORM_BCP_TEMPLATES` | | UNC BCP 4D atlas folder (`<NN>Month/BCP-<NN>M-T1.nii.gz`, `-Mask.nii.gz`); steps 2-3 | none (or pass the folder to `normalization_targets`) |
| `NEONORM_MNI_TEMPLATE` | | templateflow folder `tpl-MNI152NLin2009cSym` with `_res-1_desc-brain_mask.nii.gz` and the skull-stripped `_res-1_desc-brain_T1w.nii.gz` (see Usage); steps 2-3 | none (or pass the folder to `normalization_targets`) |

For example, in the job script or `~/.bashrc` of each server:

```bash
export ANTSPATH=/path/to/ants-2.6.2/bin
export NEONORM_SYNTHSTRIP_MODELS=/path/to/synthstrip_models
```

Check the setup before the first run; it lists what was found and what is missing:

```bash
python Script/neonorm.py --check
```

### Usage

```bash
python Script/neonorm.py sub-001_T1w.nii.gz 4.5 out/sub-001 --t2 sub-001_T2w.nii.gz
```

`age_months` is the age at scan in months (it chooses the SynthStrip models and the BCP templates). Options: `--prefix` (file name prefix, default: the T1 file name without `_T1w.nii.gz`), `--brain_mask <own mask>` (an own brain mask on the T1 as acquired instead of SynthStrip; give the run its own prefix, e.g. `<id>_manual_mask`: it reuses the rigid T2 of the SynthStrip run of the visit, if there is one). For a postmenstrual age in weeks, `neonorm.age_months_from_pma(pma_weeks)` gives the age after term in months (0 before 40 weeks). On a Slurm cluster, `submit_preprocess_array(runs, run_dir, setup=setup)` submits step 1 of many runs (one dict per run with the arguments of `preprocess`) as one job array `prep` (2 CPUs, 24 GB per task, `PREPROCESS_JOB`); per run the job script and its full output in `<out_dir>/log/<prefix>_preprocess.sh` / `.out`.

Step 2 (direct normalisation), after step 1: `t1` is `<prefix>_T1w_brain_n4.nii.gz` of step 1; the brain mask and the rigid T2 of the run are recorded from `<prefix>_preprocess.json` for the QC (`--brain_mask`, `--t2`: other files). The registration uses the T1 only for now (a T1 + T2 normalisation may follow); the T2 is used for the SUITPy cerebellum of the QC. Then the QC apart:
- step 2 (`--direct_normalize`): registration of the T1 to every template of the age; writes `<prefix>_direct_normalize.json`;
- QC (`--direct_norm_qc <out_dir>/<prefix>_direct_normalize.json`, apart): QC (`direct_norm_qc`): the reference cerebellum of the visit (`brain_to_cerebellum`: SUITPy on the skull-stripped T1 with the T2 and the brain mask, medoid of 2-3 runs, chosen without any registration; made if missing), QC per template (T1 correlation, brain and cerebellum Dice, HD95 and log10(HD95), Jacobian), figures; writes `<prefix>_direct_normalize_qc.json`.

```bash
python Script/neonorm.py --direct_normalize out/sub-001/sub-001_T1w_brain_n4.nii.gz 4.5 out/sub-001 \
    --bcp_templates /path/to/BCP-atlas-for_release-Ver2.0.0 --mni_template /path/to/tpl-MNI152NLin2009cSym
python Script/neonorm.py --direct_norm_qc out/sub-001/sub-001_direct_normalize.json   # QC
```

Options of `--direct_normalize`: `--prefix`, `--templates` (template names, default the BCP template of the nearest atlas month and MNI; for a new dataset `direct_templates(age)`: the nearest BCP template, the other templates of the visit's avgsubj edges, `avgsubj_edges(age)`, and MNI, so that every way of step 3 can use the visit and it can join the avgsubj averages; templates registered before are kept), `--no_rotation_init` (by default every registration starts from the rotation found by a rotation search against the BCP template of the age, `rotation_search_init`: ANTs `antsAI`, rigid, rotations in 30 degree steps at 4 mm, for a head that lies rotated in the field of view, which the centre-of-mass start of `antsRegistrationSyN.sh` does not correct; for MNI with the centres of mass aligned, `rotation_start`, as a search against the adult contrast of MNI picked a wrong rotation below about 11 months; this option: the centre-of-mass start of `antsRegistrationSyN.sh` instead); `out_dir` may be left out (default: the folder of the T1). The same in Python: `neonorm.direct_normalize(t1_brain, age_months, out_dir, prefix='sub-001', rotation_init=True)` and `neonorm.direct_norm_qc(out_dir / 'sub-001_direct_normalize.json')`, or part by part:

```python
import neonorm
brains = neonorm.head2brain_and_n4correction(t1, brain_mask, t2=None, out_dir=out_dir, prefix='sub-001')
for tpl in neonorm.normalization_targets(age_months, bcp_dir, mni_dir):  # BCP template of the nearest month + MNI
    reg = neonorm.register_to_template(brains['t1_brain'], tpl['name'], tpl['t1'], tpl['mask'], out_dir, prefix='sub-001')
    print(neonorm.registration_qc(reg, brain_mask))
```

On a Slurm cluster, `submit_direct_normalize(..., qc=True)` submits step 2 (8 CPUs, 16 GB) and the QC (4 CPUs, 24 GB for SUITPy) as two jobs; the second starts once the first has finished successfully (`--dependency=afterok`). Slurm job names (`squeue`): `<number>_direct`, `<number>_directqc` (`sub-001` -> `001`). The job scripts and their full output (every command, ANTs iterations included) are kept in `<out_dir>/log/`: `<prefix>_direct_normalize.sh` / `.out`, `<prefix>_direct_normalize_qc.sh` / `.out` (a new submission replaces only the files of its own job, so running the QC again keeps the log of the registrations):

```python
setup = 'module load ... && conda activate neonorm'  # ANTSPATH, NEONORM_* are written into the scripts
neonorm.submit_direct_normalize(t1_brain, age_months, out_dir, prefix='sub-001', setup=setup, qc=True)
neonorm.submit_direct_normalize(t1_brain, age_months, out_dir, prefix='sub-001', register=False, qc=True, setup=setup)  # QC again
```

`submit_direct_normalize` computes nothing itself: it writes the job scripts (`direct_normalize_job_scripts`) that run `direct_normalize` (and `direct_norm_qc`) and submits them. For many visits, `submit_direct_normalize_array(visits, run_dir, setup=setup, max_running=50, qc=True)` (visits: one dict per visit with `t1_brain`, `age_months` and optional `out_dir`, `prefix`, `templates`) submits two Slurm job arrays with one `sbatch` each: `direct` (step 2, one task per visit) and `directqc` (the QC, task i after task i of `direct` has succeeded, `--dependency=aftercorr`). At most `max_running` tasks of each array run at once; the rest wait in the Slurm queue. Each task runs the job script of its visit, so the scripts and logs per visit are the same as with `submit_direct_normalize`; `run_dir` gets the task lists, the array scripts and Slurm's own messages per task. Every task counts for the cluster's limit of submitted jobs, and an array has at most `MaxArraySize - 1` tasks.

`head2brain_and_n4correction` checks that the T2 (if given) is on the T1 grid and runs `ImageMath 3 <brain> m <image> <brain mask>` and `N4BiasFieldCorrection -d 3 -i <brain> -x <brain mask>` (the bias field is estimated inside the brain mask; without `-x` the zero background would count); `register_to_template` runs `antsRegistrationSyN.sh -d 3 -f <skull-stripped template> -m <T1 brain> -x <template mask>` with its default parameters. The QC (`registration_qc`, no thresholds yet): correlation of the warped T1 with the template, Dice of the warped brain mask with the template mask, Dice of the warped cerebellum with the template cerebellum (if both are given), Jacobian determinant of the warp.

The templates must be skull-stripped. The BCP templates are; for the MNI template, make the skull-stripped T1w once from the TemplateFlow files:

```bash
cd /path/to/tpl-MNI152NLin2009cSym
curl -L -O https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-1_T1w.nii.gz
curl -L -O https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cSym/tpl-MNI152NLin2009cSym_res-1_desc-brain_mask.nii.gz
ImageMath 3 tpl-MNI152NLin2009cSym_res-1_desc-brain_T1w.nii.gz m tpl-MNI152NLin2009cSym_res-1_T1w.nii.gz tpl-MNI152NLin2009cSym_res-1_desc-brain_mask.nii.gz
```

### Outputs (in `out_dir`; inputs are only read)

| File | When |
|---|---|
| `<prefix>_T1w_brain_mask.nii.gz` | always |
| `<prefix>_preprocess.json` | always (step 1): inputs, software actually used (versions, model checksums), outputs |
| `<prefix>_T2w_rigid2T1.nii.gz`, `<prefix>_T2w_rigid2T1_0GenericAffine.mat` | with a T2 (1a): the T2 registered rigidly to the T1 (whole head) on the T1 grid; its transform (a run with an own mask reuses those of the SynthStrip run) |
| `<prefix>_T1w_brain_n4.nii.gz`, `<prefix>_T2w_brain_n4.nii.gz` | step 1 (1c): bias-corrected, skull-stripped T1 (and T2); the input of the registration and of later analyses |
| `<prefix>_direct_to-<template>_0GenericAffine.mat`, `_1Warp.nii.gz` | step 2: subject -> template (apply as `[warp, affine]`) |
| `<prefix>_direct_to-<template>_1InverseWarp.nii.gz` | step 2: template -> subject (apply as `[affine, inverse warp]`, affine inverted) |
| `<prefix>_direct_to-<template>_Warped.nii.gz` | step 2: the skull-stripped, bias-corrected T1 in template space |
| `<prefix>_direct_normalize.json` | step 2: inputs, software (ANTs version, the `antsRegistration` calls), the N4 images of step 1 (`head2brain`), outputs of every template (`normalize`) |
| `<prefix>_cerebellum_suitpy_from-brain-N4_dseg.nii.gz` (runs, json and figure in `cerebellum_suitpy/`), `<prefix>_direct_to-<template>_qc.png`, `<prefix>_direct_normalize_qc.json` | QC (`direct_norm_qc()`; the reference cerebellum of `brain_to_cerebellum`) |
| `log/<prefix>_direct_normalize.sh` / `.out`, `log/<prefix>_direct_normalize_qc.sh` / `.out` | step 2 with `submit_direct_normalize`: job scripts and full output of the step 2 job and the QC job |
| `<prefix>_<way>[-<path>]_to-MNI152NLin2009cSym_T1w.nii.gz`, `_cerebellum_dseg.nii.gz`, `_brain_mask.nii.gz`, `_logjd.nii.gz`, `.json` | step 3 (`visit_to_mni`): T1, reference cerebellum, brain mask and log Jacobian determinant at MNI; the json lists the transforms and the ANTs commands |

