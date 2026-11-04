"""neonorm: spatial normalisation of T1w (+ optional T2w) MRI from neonates to adults, optimised for the first months of
life; one visit at a time.

Steps of the pipeline (each returns a dict of the files it wrote):
    1. preprocess(t1, age_months, out_dir, t2)   preprocessing in three parts, writes <prefix>_preprocess.json:
       1a orient                                 check the orientation; reorient with fslreorient2std if needed
       1b skull_strip_step                       SynthStrip brain mask (< 12 months: default OR infant model; from 12
                                                 months: default model); with a T2: rigid T2 on the T1 grid
       1c skull_strip_step (qc=True)             automatic QC: SUITPy cerebellum (twice) inside the mask, brain volume
                                                 for the age
    2. direct_normalize(t1, age_months, brain_mask, out_dir, t2, prefix, steps=(1, 2, 3))
                                                 direct normalisation in three parts (`steps` 1, 2, 3 = parts 2a, 2b,
                                                 2c; any of them can be run alone): the visit registered to each
                                                 template itself (indirect normalisation through chains of templates,
                                                 tpl2tpl / avgsubj: see the section 'Indirect normalisation')
       2a head2brain_and_n4correction            skull stripping + N4BiasFieldCorrection inside the brain mask of the
                                                 T1 and, if given, of the T2 on the T1 grid (rigid T2 of step 1) (saved)
       2b register_to_template                   antsRegistrationSyN.sh (ANTs, default parameters, T1 only for now; a
                                                 T1 + T2 normalisation may follow) of the skull-stripped T1 to every
                                                 template of normalization_targets(age_months)
                                                 (BCP template of the nearest atlas month and MNI; evaluation phase:
                                                 every age, planned: below 6 months, MNI only from 6 months)
       2c direct_normalization_qc                QC: the reference cerebellum of the visit (brain_to_cerebellum: SUITPy
                                                 on the skull-stripped T1 + T2 + brain mask, medoid of 2-3 runs; made
                                                 if missing), correlation, brain and cerebellum Dice, cerebellum HD95,
                                                 Jacobian, figures
       submit_direct_normalize() submits 2a-2b and 2c as two Slurm jobs (the QC after the registrations).
    3. visit_to_mni(normalize_json, way, path)   the visit to MNI by one way (direct, or along a chain of templates:
                                                 tpl2tpl / avgsubj), in three parts (submit_visits_to_mni):
       3a                                        all transforms of the way composed into one field on the MNI grid
       3b                                        T1, reference cerebellum and brain mask moved with that field (one
                                                 interpolation); log Jacobian determinant of the field
       3c mni_metrics                            T1 r, brain and cerebellum Dice and HD95 with MNI, mean log Jacobian,
                                                 share of folded voxels (evaluation figure: plot_by_month)

Software (see README.md): FSL (fslreorient2std), FreeSurfer (mri_synthstrip) with the SynthStrip default and infant
models, ANTs (antsRegistrationSyN.sh, N4BiasFieldCorrection, ImageMath; steps 2-3) and the Python packages of
environment.yml (ANTsPy, SUITPy, ...). Settings that depend on the computer are read from the command line, then from
environment variables, then the defaults below:
    FSLDIR                          FSL installation (or fslreorient2std on the PATH)
    FREESURFER_HOME                 FreeSurfer installation (or mri_synthstrip on the PATH)
    ANTSPATH                        ANTs bin folder with antsRegistrationSyN.sh, N4BiasFieldCorrection, ImageMath
                                    (else antsRegistrationSyN.sh on the PATH)
    NEONORM_SYNTHSTRIP_MODELS    folder with synthstrip.1.pt and synthstrip.infant.1.pt
                                    (default: $FREESURFER_HOME/models, which has no infant model)
    NEONORM_NTHREADS             CPU threads (default: SLURM_CPUS_PER_TASK, else the tool defaults)
    NEONORM_BCP_TEMPLATES        UNC BCP 4D atlas folder (<NN>Month/BCP-<NN>M-T1.nii.gz, -Mask.nii.gz)
    NEONORM_MNI_TEMPLATE         templateflow folder tpl-MNI152NLin2009cSym (res-1 desc-brain_T1w and desc-brain_mask)
Check the setup with:  python neonorm.py --check

Usage:
    python neonorm.py <t1> <age_months> <out_dir> [--t2 <t2>] [--prefix <prefix>] [--save_brain] [--qc_figure]  (step 1)
    python neonorm.py --direct_normalize <t1> <age_months> [<out_dir>] --brain_mask <mask> [--t2 <T2 on the T1 grid>]
                      [--prefix <prefix>] [--steps 1 2 3] [--bcp_templates <dir>] [--mni_template <dir>]    (step 2;
                      --steps: parts 2a 2b 2c)
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

__version__ = '0.1.0'

HERE = Path(__file__).resolve().parent
# Brain volume (SynthStrip default model) per age bin of the BCP visits; median and robust SD, no individual data
VOLUME_NORMS = HERE / 'data' / 'bcp_synthstrip_volume_by_age.tsv'
SYNTHSTRIP_MODELS = {'synthstrip': 'synthstrip.1.pt', 'synthstrip_infant': 'synthstrip.infant.1.pt'}
UNION_BELOW_MONTHS = 12  # brain mask: synthstrip OR synthstrip_infant below, synthstrip (default model) from here on
COV_MIN = 0.95           # QC: share of the SUITPy cerebellum inside the brain mask
DICE_MIN = 0.95          # QC: agreement of the two SUITPy runs (SUITPy is not deterministic and can misplace the cerebellum)
Z_MAX = 3                # QC: |robust z| of the SynthStrip volume within the age bin
CEREB_DICE_MIN = 0.90    # QC of 2c: below this cerebellum Dice with the main template, needs_visual_check (provisional)
# Normalisation: BCP template of the nearest atlas month and MNI below this age, MNI only from it on. Evaluation phase:
# every age (inf); planned: 6.
AGE_TEMPLATE_BELOW_MONTHS = float('inf')
BCP_MONTHS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24, 36, 48, 60)  # months of the UNC BCP 4D atlas
REGISTRATION = 'antsRegistrationSyN.sh'  # ANTs, default parameters (-t s), as in the earlier T1-only registrations
MNI_NAME = 'MNI152NLin2009cSym'
SHOW_TOOL_OUTPUT = False  # True: steps 2-3 print every command and its full output (set by the command line, e.g. Slurm logs)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def find_tool(name, home_var, subdirs):
    """Executable `name`: on the PATH, else in $home_var/<subdir>; None if not found."""
    found = shutil.which(name)
    if found:
        return found
    home = os.environ.get(home_var)
    for sub in subdirs if home else ():
        p = Path(home) / sub / name
        if p.exists():
            return str(p)
    return None


def fslreorient2std():
    return find_tool('fslreorient2std', 'FSLDIR', ['share/fsl/bin', 'bin'])


def mri_synthstrip():
    return find_tool('mri_synthstrip', 'FREESURFER_HOME', ['bin'])


def ants_registration_syn():
    """antsRegistrationSyN.sh: in $ANTSPATH (or $ANTSPATH/bin) if set, else on the PATH; None if not found."""
    home = os.environ.get('ANTSPATH')
    for d in ([Path(home), Path(home) / 'bin'] if home else []):
        if (d / 'antsRegistrationSyN.sh').exists():
            return str(d / 'antsRegistrationSyN.sh')
    return shutil.which('antsRegistrationSyN.sh')


def _ants_env():
    """Environment for antsRegistrationSyN.sh, which calls antsRegistration and PrintHeader from its own folder."""
    d = str(Path(ants_registration_syn()).parent)
    return dict(os.environ, ANTSPATH=d + os.sep, PATH=d + os.pathsep + os.environ.get('PATH', ''))


def _run(cmd, env=None):
    """Run a command and return its output (stdout and stderr). With SHOW_TOOL_OUTPUT the command and its output are
    printed line by line as it runs; otherwise it is quiet and the output is shown only if it fails."""
    cmd = [str(c) for c in cmd]
    if SHOW_TOOL_OUTPUT:
        print('$ ' + ' '.join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    out = []
    for line in proc.stdout:
        out.append(line)
        if SHOW_TOOL_OUTPUT:
            print(line, end='', flush=True)
    if proc.wait():
        tail = '' if SHOW_TOOL_OUTPUT else ''.join(out)[-5000:]
        raise RuntimeError(f'{Path(cmd[0]).name} failed ({proc.returncode}):\n{" ".join(cmd)}\n{tail}')
    return ''.join(out)


def _section(title):
    if SHOW_TOOL_OUTPUT:
        print(f'\n===== {title} =====', flush=True)


ANTS_COMMANDS = None  # a list: run_ants appends every command line it runs (visit_to_mni saves them in its json)


def run_ants(tool, *args):
    """Run an ANTs command-line tool from the folder of antsRegistrationSyN.sh (see _run for its output)."""
    script = ants_registration_syn()
    if not script:
        raise RuntimeError('ANTs not found: set ANTSPATH to the ANTs bin folder (see README.md)')
    cmd = [Path(script).parent / tool, *args]
    if ANTS_COMMANDS is not None:
        ANTS_COMMANDS.append(shlex.join(str(c) for c in cmd))
    return _run(cmd, env=_ants_env())


def ants_version():
    """'2.6.2' from antsRegistration --version next to antsRegistrationSyN.sh; None if not found."""
    script = ants_registration_syn()
    if not script:
        return None
    r = subprocess.run([str(Path(script).parent / 'antsRegistration'), '--version'], capture_output=True, text=True)
    return r.stdout.split('Version:')[1].split()[0] if 'Version:' in r.stdout else None


def synthstrip_model(method):
    """Model file of 'synthstrip' or 'synthstrip_infant' (None if not found)."""
    folders = [os.environ.get('NEONORM_SYNTHSTRIP_MODELS'),
               os.path.join(os.environ['FREESURFER_HOME'], 'models') if os.environ.get('FREESURFER_HOME') else None]
    for d in folders:
        if d and (Path(d) / SYNTHSTRIP_MODELS[method]).exists():
            return str(Path(d) / SYNTHSTRIP_MODELS[method])
    return None


def n_threads():
    n = os.environ.get('NEONORM_NTHREADS') or os.environ.get('SLURM_CPUS_PER_TASK')
    return int(n) if n else None


def bcp_template(age_months, bcp_dir=None):
    """BCP template (skull-stripped) of the nearest month of BCP_MONTHS, the older one if two are equally near (folder
    from bcp_dir or NEONORM_BCP_TEMPLATES):
    {'name': 'BCP03M', 't1', 'mask', 'cerebellum'}; 'cerebellum' is <...>-T1_cerebellum_suitpy_dseg.nii.gz, the medoid of
    bcp_template_cerebellum (None if missing)."""
    bcp_dir = Path(bcp_dir or os.environ['NEONORM_BCP_TEMPLATES'])
    m = min(BCP_MONTHS, key=lambda month: (abs(month - age_months), -month))
    base = bcp_dir / f'{m:02d}Month' / f'BCP-{m:02d}M'
    cereb = Path(f'{base}-T1_cerebellum_suitpy_dseg.nii.gz')
    return _existing({'name': f'BCP{m:02d}M', 't1': f'{base}-T1.nii.gz', 'mask': f'{base}-Mask.nii.gz',
                      'cerebellum': str(cereb) if cereb.exists() else None})


def mni_template(mni_dir=None):
    """MNI152NLin2009cSym, 1 mm, templateflow folder (from mni_dir or NEONORM_MNI_TEMPLATE): 't1' is the skull-stripped
    <...>_res-1_desc-brain_T1w.nii.gz (templateflow T1w x desc-brain_mask, made once with ImageMath m; templateflow has no
    skull-stripped T1w), 'mask' the desc-brain_mask; 'cerebellum' is None if the folder has no
    <...>_res-1_cerebellum_mask.nii.gz."""
    base = Path(mni_dir or os.environ['NEONORM_MNI_TEMPLATE']) / f'tpl-{MNI_NAME}_res-1'
    cereb = Path(f'{base}_cerebellum_mask.nii.gz')
    return _existing({'name': MNI_NAME, 't1': f'{base}_desc-brain_T1w.nii.gz', 'mask': f'{base}_desc-brain_mask.nii.gz',
                      'cerebellum': str(cereb) if cereb.exists() else None})


def _existing(tpl):
    """tpl, after checking that its T1 and mask exist (a wrong template folder fails here, not in the registration)."""
    missing = [tpl[k] for k in ('t1', 'mask') if not Path(tpl[k]).exists()]
    if missing:
        raise FileNotFoundError(f"template {tpl['name']}: missing {', '.join(missing)}")
    return tpl


def normalization_targets(age_months, bcp_dir=None, mni_dir=None):
    """Templates of a visit: below AGE_TEMPLATE_BELOW_MONTHS (evaluation phase: every age) the BCP template of the
    nearest atlas month and MNI, else MNI."""
    return ([bcp_template(age_months, bcp_dir)] if age_months < AGE_TEMPLATE_BELOW_MONTHS else []) + [mni_template(mni_dir)]


def bracket_templates(age_months):
    """The two neighbouring templates of the chain (adjacent_chain) around age_months, (a, b) with a <= age < b: the
    avgsubj edge a -> b (notebook 4) averages template a -> visit -> template b over the visits of this bracket, so each
    of them is registered to both (notebook 3, 3.3). From the last BCP month on: (BCP60M, MNI). The avgsubj edges of
    notebook 3 and 4 since 2026-11-02: avgsubj_edges (sliding age windows); this bracket is one of them."""
    i = max(k for k, m in enumerate(BCP_MONTHS) if m <= age_months)
    chain = adjacent_chain()
    return chain[i], chain[i + 1]


def avgsubj_edges(age_months):
    """The avgsubj edges (a, b) of the adjacent chain whose age window contains the visit, as the sliding windows of
    the BCP atlas (Chen et al. 2022, NeuroImage 253:119097: template m from the scans of the months m-1 to m+1, not a
    hard age threshold): scan month = age rounded to the month; edge a -> b uses the visits with a <= scan month <= b
    (from the last BCP month on: BCP60M -> MNI). A visit of an atlas month is in the two edges that meet there, any
    other in one; it is registered to all templates of its edges (notebook 3, 3.3)."""
    month = np.floor(age_months + 0.5)
    ends = list(BCP_MONTHS) + [np.inf]
    chain = adjacent_chain()
    return [(chain[i], chain[i + 1]) for i in range(len(BCP_MONTHS)) if ends[i] <= month <= ends[i + 1]]


def direct_templates(age_months, bcp_dir=None):
    """Template names of the direct normalisation of a new visit (direct_normalize(templates=...)), so that every way
    to MNI can use it later (visit_to_mni) and it can join the avgsubj edges of its age: the BCP template of the nearest
    atlas month first (bcp_template: the main template of the QC, the start of tpl2tpl / avgsubj), the other templates
    of its avgsubj edges (avgsubj_edges: usually the two atlas months around the age, three at an atlas month), then
    MNI. E.g. 5.55 months: BCP06M, BCP05M, BCP07M, MNI; 13 months: BCP12M, BCP15M, MNI; 97 months: BCP60M, MNI."""
    names = [bcp_template(age_months, bcp_dir)['name']] + [t for e in avgsubj_edges(age_months) for t in e]
    return list(dict.fromkeys(names + [MNI_NAME]))


def _find_up(path, rel):
    """First parent folder of `path` that contains `rel` (e.g. 'etc/fslversion'); None if none does."""
    for d in Path(path).resolve().parents:
        if (d / rel).exists():
            return d
    return None


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def environment_info():
    """Software actually used (written to every json of neonorm, e.g. <prefix>_preprocess.json)."""
    info = {'neonorm': __version__, 'python': sys.version.split()[0]}
    # the version of the module imported: the package metadata can name another one when an environment holds
    # several of them (until 2026-11-03 the jsons had numpy 2.4.0, nibabel 5.3.3, torch 2.4.1 instead of the imported
    # 2.4.6, 5.3.2, 2.6.0; torch is not used)
    for p, module in [('antspyx', 'ants'), ('SUITPy', 'SUITPy'), ('nibabel', 'nibabel'), ('numpy', 'numpy'),
                      ('pandas', 'pandas'), ('scipy', 'scipy')]:
        try:
            info[p] = __import__(module).__version__
        except Exception:
            info[p] = None
    fsl = fslreorient2std()
    fsl_home = _find_up(fsl, 'etc/fslversion') if fsl else None
    info['fslreorient2std'] = fsl
    info['fsl_version'] = (fsl_home / 'etc' / 'fslversion').read_text().strip() if fsl_home else None
    ss = mri_synthstrip()
    fs_home = _find_up(ss, 'build-stamp.txt') if ss else None
    info['mri_synthstrip'] = ss
    info['freesurfer_version'] = (fs_home / 'build-stamp.txt').read_text().strip() if fs_home else None
    info['antsRegistrationSyN.sh'] = ants_registration_syn()
    info['ants_version'] = ants_version()
    for m in SYNTHSTRIP_MODELS:
        f = synthstrip_model(m)
        info[f'{m}_model'] = f
        info[f'{m}_model_md5'] = _md5(f) if f else None
    info['threads'] = n_threads()
    return info


def check_environment(need_infant=True, need_ants=True, verbose=False):
    """List of problems (empty = ready). need_ants: ANTs is needed for steps 2-3 only. verbose: print what was found."""
    problems = []
    if not fslreorient2std():
        problems.append('fslreorient2std not found: install FSL and set FSLDIR, or put fslreorient2std on the PATH')
    if not mri_synthstrip():
        problems.append('mri_synthstrip not found: install FreeSurfer (>= 7.4) and set FREESURFER_HOME, or put '
                        'mri_synthstrip on the PATH')
    if need_ants and not ants_registration_syn():
        problems.append('antsRegistrationSyN.sh not found: install ANTs (>= 2.4) and set ANTSPATH to its bin folder, or '
                        'put it on the PATH')
    elif need_ants:
        for tool in ('antsRegistration', 'PrintHeader', 'N4BiasFieldCorrection', 'ImageMath'):
            if not (Path(ants_registration_syn()).parent / tool).exists():
                problems.append(f'{tool} not found next to antsRegistrationSyN.sh: use a complete ANTs installation')
    for m in ['synthstrip'] + (['synthstrip_infant'] if need_infant else []):
        if not synthstrip_model(m):
            problems.append(f'{SYNTHSTRIP_MODELS[m]} not found: put it in a folder and set NEONORM_SYNTHSTRIP_MODELS '
                            '(see README.md)')
    for p in ['antspyx', 'SUITPy']:
        try:
            package_version(p)
        except PackageNotFoundError:
            problems.append(f'Python package {p} missing: create the environment of environment.yml')
    if not VOLUME_NORMS.exists():
        problems.append(f'{VOLUME_NORMS} missing (part of the repository)')
    if verbose:
        for k, v in environment_info().items():
            print(f'{k:28s} {v}')
    return problems


def _require(need_infant):
    problems = check_environment(need_infant, need_ants=False)  # preprocess() does step 1
    if problems:
        raise RuntimeError('neonorm cannot run:\n  - ' + '\n  - '.join(problems))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _fsl_env():
    env = dict(os.environ, FSLOUTPUTTYPE='NIFTI_GZ')
    tool = fslreorient2std()
    if tool and 'FSLDIR' not in env and _find_up(tool, 'etc/fslversion'):
        env['FSLDIR'] = str(_find_up(tool, 'etc/fslversion'))  # fslreorient2std calls other FSL tools
    return env


def reorient(in_file, out_file):
    """fslreorient2std: lossless permutation / flip of the voxel axes to the MNI convention (world coordinates kept)."""
    subprocess.run([fslreorient2std(), str(in_file), str(out_file)], check=True, env=_fsl_env())
    return out_file


def is_standard_orientation(image):
    """True if fslreorient2std would not change the image (its transform is the identity)."""
    out = subprocess.run([fslreorient2std(), str(image)], capture_output=True, text=True, check=True, env=_fsl_env()).stdout
    return np.allclose(np.array(out.split(), dtype=float).reshape(4, 4), np.eye(4))


def synthstrip(t1, out_mask, method='synthstrip'):
    """SynthStrip brain mask; method 'synthstrip' (default model, with extra-axial CSF) or 'synthstrip_infant'."""
    cmd = [mri_synthstrip(), '-i', str(t1), '-m', str(out_mask), '--model', synthstrip_model(method)]
    if n_threads():
        cmd += ['-t', str(n_threads())]
    r = subprocess.run(cmd, capture_output=True, text=True)  # quiet; its output is shown only if it fails
    if r.returncode:
        raise RuntimeError(f'mri_synthstrip failed ({r.returncode}):\n{r.stdout}\n{r.stderr}')
    return out_mask


def _as_3d(img):
    import ants
    return ants.slice_image(img, axis=3, idx=0) if img.dimension == 4 else img


def suitpy_cerebellum(t1, out_dseg, t2=None, brain_mask=None):
    """SUITPy isolate() on the T1, optionally with a T2 on the T1 grid and a brain mask: the one SUITPy call of the project
    (whole head: skull_strip_step and skull_strip.head_to_cerebellum; skull-stripped: brain_to_cerebellum; templates:
    bcp_template_cerebellum). Returns out_dseg, or None if SUITPy finds no cerebellum."""
    import ants
    import SUITPy as suit
    with tempfile.TemporaryDirectory() as tmp:
        t1_tmp = os.path.join(tmp, 'T1.nii.gz')
        ants.image_write(_as_3d(ants.image_read(str(t1))), t1_tmp)
        t2_tmp = None
        if t2:
            t2_tmp = os.path.join(tmp, 'T2.nii.gz')
            ants.image_write(_as_3d(ants.image_read(str(t2))), t2_tmp)
        try:
            suit.isolate(t1_file=t1_tmp, t2_file=t2_tmp,
                         brain_mask_file=str(brain_mask) if brain_mask else None, result_folder=tmp, verbose=False)
        except AttributeError:  # no cerebellum found: isolate() fails writing an empty result (e.g. BCP 01M, T1 only)
            return None
        shutil.copy(os.path.join(tmp, 'T1_cerebellum_dseg.nii.gz'), out_dseg)
    return out_dseg


def _load_bool(path):
    return np.squeeze(np.asanyarray(nib.load(path).dataobj)) > 0


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def _hd95(a, b, spacing):
    """95th percentile Hausdorff distance (mm) between the surfaces of two boolean masks on the same grid (voxel size
    spacing): the larger of the two directed 95th percentiles (as medpy.metric.hd95). NaN if a mask is empty."""
    from scipy import ndimage
    if not (a.any() and b.any()):
        return float('nan')
    box = tuple(slice(max(int(i.min()) - 2, 0), int(i.max()) + 3) for i in np.nonzero(a | b))  # faster on the crop
    surface = lambda m: m & ~ndimage.binary_erosion(m)
    sa, sb = surface(a[box]), surface(b[box])
    to = lambda s: ndimage.distance_transform_edt(~s, sampling=spacing)  # distance to the nearest voxel of surface s
    return float(max(np.percentile(to(sb)[sa], 95), np.percentile(to(sa)[sb], 95)))


def default_prefix(t1):
    """'.../sub-001_T1w.nii.gz' -> 'sub-001'."""
    name = Path(t1).name
    for ext in ('.nii.gz', '.nii'):
        name = name[:-len(ext)] if name.endswith(ext) else name
    return name[:-len('_T1w')] if name.endswith('_T1w') else name


# ---------------------------------------------------------------------------
# Step 1 (preprocessing), 1a: orientation
# ---------------------------------------------------------------------------

def orient(t1, t2=None, out_dir='.', prefix=None):
    """Part 1a of the preprocessing, orientation. Images already in the standard (fslreorient2std) orientation are used as they are (no copy); others are
    reoriented to <prefix>_<T1w|T2w>_reorient.nii.gz in out_dir (lossless axis permutation / flip, world coordinates
    unchanged). Returns {'t1': path to use, 't2': path to use or None, 't1_reoriented': bool, 't2_reoriented': bool}."""
    prefix = prefix or default_prefix(t1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for mod, img in (('t1', t1), ('t2', t2)):
        if img is None:
            res[mod], res[f'{mod}_reoriented'] = None, False
        elif is_standard_orientation(img):
            res[mod], res[f'{mod}_reoriented'] = str(img), False
        else:
            dst = out_dir / f'{prefix}_{mod.upper()}w_reorient.nii.gz'
            reorient(img, dst)
            res[mod], res[f'{mod}_reoriented'] = str(dst), True
    return res


# ---------------------------------------------------------------------------
# Step 1 (preprocessing), 1b-1c: skull stripping and QC
# ---------------------------------------------------------------------------

def _save_masked(image, mask, out):
    img = nib.load(image)
    data = np.squeeze(np.asanyarray(img.dataobj)) * mask
    out_img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, out)
    return str(out)


def brain_mask_methods(age_months):
    """SynthStrip models whose masks are united into the brain mask at this age."""
    return ['synthstrip'] + (['synthstrip_infant'] if age_months < UNION_BELOW_MONTHS else [])


def save_union_mask(mask_files, out_file):
    """Union of masks on one grid (e.g. the masks of brain_mask_methods) as uint8, header of the first mask."""
    ref = nib.load(mask_files[0])
    brain = np.any([_load_bool(f) for f in mask_files], axis=0)
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), ref.affine, ref.header), out_file)
    return str(out_file)


def volume_z(volume_mm3, age_months, norms=VOLUME_NORMS):
    """Robust z of a SynthStrip (default model) brain volume within its BCP age bin; (z, bin label, bin median)."""
    t = pd.read_csv(norms, sep='\t', index_col='age_bin_months')
    for label, r in t.iterrows():
        lo, hi = (float(label[:-1]), np.inf) if label.endswith('+') else map(float, label.split('-'))
        if lo <= age_months < hi:
            return (volume_mm3 - r['median_mm3']) / r['robust_sd_mm3'], label, r['median_mm3']
    return np.nan, '', np.nan


def qc_figure(t1, masks, cerebs, out_png, text):
    """T1 with the SynthStrip masks (lines) and the two SUITPy cerebellum runs (run 1 filled, run 2 dashed), through the
    brain centre and the cerebellum; the QC result above the images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'synthstrip': '#a80000', 'synthstrip_infant': '#ff2a2a', 'brain mask': '#ffa899'}
    img = np.squeeze(np.asanyarray(nib.load(t1).dataobj)).astype(float)
    centres = [('brain centre', np.round(np.mean(np.nonzero(masks['brain mask']), axis=1)).astype(int))]
    if cerebs[0].any():
        centres.append(('cerebellum', np.round(np.mean(np.nonzero(cerebs[0]), axis=1)).astype(int)))
    fig, axes = plt.subplots(len(centres), 3, figsize=(12, 4 * len(centres)), squeeze=False)
    vmax = np.percentile(img[img > 0], 99)
    for row, (label, c) in zip(axes, centres):
        for axis, ax in enumerate(row):
            sl = lambda v: np.take(v, c[axis], axis=axis).T
            ax.imshow(sl(img), cmap='gray', vmin=0, vmax=vmax, origin='lower')
            ax.contourf(sl(cerebs[0]).astype(float), levels=[0.5, 1.5], colors=['#1baf7a'], alpha=0.3)
            ax.contour(sl(cerebs[1]).astype(float), levels=[0.5], colors='#e4572e', linestyles='--', linewidths=1.2)
            for k, m in masks.items():
                ax.contour(sl(m).astype(float), levels=[0.5], colors=colors[k], linewidths=1.2)
            ax.axis('off')
        row[0].set_title(label, fontsize=9, loc='left')
    fig.legend(handles=[plt.Line2D([], [], color=colors[k], lw=2, label=k) for k in masks]
               + [plt.Line2D([], [], color='#1baf7a', lw=6, alpha=0.4, label='SUITPy cerebellum, run 1'),
                  plt.Line2D([], [], color='#e4572e', ls='--', lw=2, label='SUITPy cerebellum, run 2')],
               loc='lower center', ncol=len(masks) + 2, frameon=False, fontsize=9)
    fig.suptitle(text, fontsize=9, ha='left', x=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return str(out_png)


def skull_strip_step(t1, age_months, t2=None, out_dir='.', prefix=None, t1_brain=None, save_brain=False,
                     qc=True, always_qc_figure=False):
    """Parts 1b (skull stripping, rigid T2) and 1c (QC, if qc) of the preprocessing. t1 / t2: images of orient (not
    skull-stripped). Brain mask from SynthStrip: below UNION_BELOW_MONTHS
    the default model OR the infant model (the infant model alone excludes extra-axial CSF; the union keeps the
    cerebellum where the default model cuts it), from UNION_BELOW_MONTHS on the default model. With a T2, it is
    registered rigidly to the T1 (whole head) and saved on the T1 grid, <prefix>_T2w_rigid2T1.nii.gz (the T2 input of
    step 2), with its transform. QC, no visual QC of the images needed:
      - SUITPy cerebellum (T1, not skull-stripped) run twice: flag if the runs differ (Dice < DICE_MIN), else if less
        than COV_MIN of it lies inside the brain mask;
      - volume: robust z of the SynthStrip (default model) volume within the BCP age bin, flag |z| > Z_MAX.
    A flagged visit needs a look at <prefix>_skullstrip_qc.png. Returns a dict of the outputs and the QC result."""
    import ants
    prefix = prefix or default_prefix(t1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_threads():
        os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(n_threads()))  # ANTs threads
    res = {'age_months': float(age_months)}
    with tempfile.TemporaryDirectory() as tmp:
        methods = brain_mask_methods(age_months)
        masks = {}
        for m in methods:
            f = os.path.join(tmp, f'{m}_mask.nii.gz')
            synthstrip(t1, f, m)
            masks[m] = _load_bool(f)
        ref = nib.load(os.path.join(tmp, 'synthstrip_mask.nii.gz'))
        brain = np.any(list(masks.values()), axis=0)
        res['brain_mask_rule'] = 'synthstrip OR synthstrip_infant' if len(masks) > 1 else 'synthstrip'
        res['brain_mask'] = str(out_dir / f'{prefix}_T1w_brain_mask.nii.gz')
        nib.save(nib.Nifti1Image(brain.astype(np.uint8), ref.affine, ref.header), res['brain_mask'])
        if save_brain or t1_brain:
            res['t1_brain'] = _save_masked(t1, brain, t1_brain or out_dir / f'{prefix}_T1w_brain.nii.gz')

        if t2 is not None:
            t1_img = _as_3d(ants.image_read(str(t1)))
            reg = ants.registration(fixed=t1_img, moving=_as_3d(ants.image_read(str(t2))), type_of_transform='Rigid')
            res['t2_to_t1_transform'] = str(out_dir / f'{prefix}_T2w_rigid2T1_0GenericAffine.mat')
            shutil.copy(reg['fwdtransforms'][0], res['t2_to_t1_transform'])
            res['t2_on_t1'] = str(out_dir / f'{prefix}_T2w_rigid2T1.nii.gz')
            ants.image_write(reg['warpedmovout'], res['t2_on_t1'])
            if save_brain:
                res['t2_brain'] = _save_masked(res['t2_on_t1'], brain, out_dir / f'{prefix}_T2w_brain.nii.gz')

        if qc:
            vox = float(np.prod(ref.header.get_zooms()[:3]))  # mm3 per voxel
            cerebs = []
            for run in (1, 2):
                f = os.path.join(tmp, f'cerebellum_run{run}.nii.gz')
                suitpy_cerebellum(t1, f)
                cerebs.append(_load_bool(f))
            agree = float(_dice(*cerebs)) if cerebs[0].any() or cerebs[1].any() else np.nan
            coverage = [float((c & brain).sum() / c.sum()) if c.any() else np.nan for c in cerebs]
            vol = float(masks['synthstrip'].sum()) * vox
            z, age_bin, median = volume_z(vol, age_months)
            flags = []
            if not agree >= DICE_MIN:
                flags.append(f'cerebellum: the two SUITPy runs differ (Dice {agree:.3f}), the cerebellum check may be '
                             f'unreliable -> check the cerebellum in the figure; inside the brain mask: run 1 '
                             f'{coverage[0]:.0%}, run 2 {coverage[1]:.0%}')
            elif not min(coverage) >= COV_MIN:
                flags.append(f'cerebellum: only {min(coverage):.0%} of the SUITPy cerebellum inside the brain mask -> does '
                             'the mask cut the cerebellum?')
            if abs(z) > Z_MAX:
                flags.append(f'volume: synthstrip {vol:,.0f} mm3, z = {z:+.1f} vs the BCP median of {age_bin} months '
                             f'({median:,.0f} mm3) -> ' + ('non-brain included, or older than the age' if z > 0
                                                         else 'brain cut off, or younger than the age'))
            res['qc'] = {'suitpy_runs_dice': agree, 'cereb_coverage_run1': coverage[0], 'cereb_coverage_run2': coverage[1],
                         'synthstrip_volume_mm3': vol, 'volume_z': float(z), 'age_bin': age_bin, 'flags': flags,
                         'needs_visual_check': bool(flags)}
            if flags or always_qc_figure:
                text = (f'{prefix} | age {age_months:.1f} months | brain mask: {res["brain_mask_rule"]}\n'
                        + ('\n'.join(flags) if flags else
                           f'QC passed: SUITPy runs Dice {agree:.3f}, cerebellum coverage {min(coverage):.0%}, volume z {z:+.1f}'))
                res['qc_figure'] = qc_figure(t1, {**masks, 'brain mask': brain}, cerebs,
                                             out_dir / f'{prefix}_skullstrip_qc.png', text)
    return res


# ---------------------------------------------------------------------------
# Step 2: direct normalisation and QC
# ---------------------------------------------------------------------------

def registration_qc_figure(tpl, warped, tpl_mask, tpl_cereb, out_png, text):
    """Template (upper row) and the warped T1 (lower row) through the template brain centre (and its cerebellum), with
    the template brain mask (and cerebellum) as lines on both."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    centres = [('brain centre', np.round(np.mean(np.nonzero(tpl_mask), axis=1)).astype(int))]
    if tpl_cereb is not None:
        centres.append(('cerebellum', np.round(np.mean(np.nonzero(tpl_cereb), axis=1)).astype(int)))
    fig, axes = plt.subplots(2 * len(centres), 3, figsize=(12, 7 * len(centres)), squeeze=False)
    for i, (label, c) in enumerate(centres):
        for j, (name, img) in enumerate((('template', tpl), ('warped T1', warped))):
            row = axes[2 * i + j]
            vmax = np.percentile(img[tpl_mask], 99)
            for axis, ax in enumerate(row):
                sl = lambda v: np.take(v, c[axis], axis=axis).T
                ax.imshow(sl(img), cmap='gray', vmin=0, vmax=vmax, origin='lower')
                ax.contour(sl(tpl_mask).astype(float), levels=[0.5], colors='#ff2a2a', linewidths=1)
                if tpl_cereb is not None:
                    ax.contour(sl(tpl_cereb).astype(float), levels=[0.5], colors='#1baf7a', linewidths=1)
                ax.axis('off')
            row[0].set_title(f'{name}, {label}', fontsize=9, loc='left')
    handles = [plt.Line2D([], [], color='#ff2a2a', lw=2, label='template brain mask')]
    if tpl_cereb is not None:
        handles.append(plt.Line2D([], [], color='#1baf7a', lw=2, label='template cerebellum'))
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle(text, fontsize=9, ha='left', x=0.02)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return str(out_png)


def edge_corr(t1, t2_on_t1, brain_mask):
    """Edge agreement of a T1 and a T2 on the same grid inside brain_mask (file or boolean array): correlation of their
    gradient magnitudes. About 0.4-0.6 when the T2 lies on the T1, about 0 when it does not (BCP, 2026-10-27)."""
    mask = brain_mask if isinstance(brain_mask, np.ndarray) else _load_bool(brain_mask)
    grad = lambda f: np.sqrt(sum(g ** 2 for g in np.gradient(np.squeeze(np.asanyarray(nib.load(f).dataobj)).astype(float))))
    return float(np.corrcoef(grad(t1)[mask], grad(t2_on_t1)[mask])[0, 1])


def head2brain_and_n4correction(t1, brain_mask, t2=None, out_dir=None, prefix=None):
    """Part 2a of the direct normalisation: skull stripping and bias correction of the T1 of step 1 (orient) and, if
    given, of the T2. t2 (optional; the registration is T1 only for now, the T2 is prepared for a later T1 + T2
    normalisation): the T2 already on the T1 grid, not skull-stripped (the rigid T2 of step 1,
    <prefix>_T2w_rigid2T1.nii.gz, whole head; no new registration). ANTs, default parameters:
      ImageMath 3 <brain> m <image> <brain_mask>                        (skull stripping)
      N4BiasFieldCorrection -d 3 -i <brain> -x <brain_mask> -o <out>    (bias field estimated inside the brain mask;
                                                                         without -x the zero background would count)
    Saved in out_dir (default: the folder of t1): <prefix>_T1w_brain_n4.nii.gz (input of register_to_template and of later
    analyses) and <prefix>_T2w_brain_n4.nii.gz (on the T1 grid). Returns {'t1_brain', 't2_brain'} ('t2_brain' None
    without a T2)."""
    prefix = prefix or default_prefix(t1)
    out_dir = Path(out_dir) if out_dir else Path(t1).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_threads():
        os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(n_threads()))  # ANTs threads
    res = {'t1_brain': None, 't2_brain': None}
    with tempfile.TemporaryDirectory() as tmp:
        images = {'t1': str(t1)}
        if t2 is not None:
            ref, img = nib.load(t1), nib.load(t2)
            if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-3):
                raise ValueError(f'{t2} is not on the T1 grid: give the rigid T2 of step 1 (<prefix>_T2w_rigid2T1.nii.gz)')
            images['t2'] = str(t2)
        for mod, f in images.items():
            brain = os.path.join(tmp, f'{mod}_brain.nii.gz')
            res[f'{mod}_brain'] = str(out_dir / f'{prefix}_{mod.upper()}w_brain_n4.nii.gz')
            _section(f'{mod.upper()}: skull stripping (ImageMath) and bias correction (N4BiasFieldCorrection)')
            run_ants('ImageMath', 3, brain, 'm', f, brain_mask)
            run_ants('N4BiasFieldCorrection', '-d', 3, '-i', brain, '-x', brain_mask, '-o', res[f'{mod}_brain'])
    return res


def register_to_template(t1_brain, tpl_name, tpl_t1, tpl_mask, out_dir='.', prefix=None, init=None):
    """Part 2b of the direct normalisation: registration of one visit to one template; once per template of
    normalization_targets().
    t1_brain: bias-corrected, skull-stripped T1 of head2brain_and_n4correction; tpl_t1: skull-stripped template T1 (the
    BCP templates are; MNI: see mni_template); then antsRegistrationSyN.sh -d 3 -f <tpl_t1> -m <t1_brain> -x <tpl_mask>
    with its default parameters (-t s: rigid + affine with MI, SyN with CC radius 2; more levels when the template has
    > 256 voxels along an axis, e.g. the BCP templates; T1 only; tpl_mask in all stages). The registration starts from
    the alignment of the centres of mass, or from init (-i: a start transform, e.g. of rotation_search_init, for a head
    that lies rotated in the field of view).
    Saved in out_dir:
      <prefix>_to-<tpl_name>_0GenericAffine.mat, _1Warp.nii.gz   subject -> template: transforms [warp, affine]
      <prefix>_to-<tpl_name>_1InverseWarp.nii.gz                  template -> subject: transforms [affine, inverse warp],
                                                                  whichtoinvert [True, False]
      <prefix>_to-<tpl_name>_Warped.nii.gz                        t1_brain in template space (linear interpolation)
    Returns a dict of the outputs, the template files (with md5) and the antsRegistration call it made ('ants_call').
    QC: direct_normalization_qc / registration_qc."""
    script = ants_registration_syn()
    if not script:
        raise RuntimeError('antsRegistrationSyN.sh not found: set ANTSPATH to the ANTs bin folder (see README.md)')
    prefix = prefix or default_prefix(t1_brain)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n_threads():
        os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(n_threads()))  # ANTs threads
    res = {'template': tpl_name, 't1_brain': str(t1_brain), 'tpl_t1': str(tpl_t1), 'tpl_t1_md5': _md5(tpl_t1),
           'tpl_mask': str(tpl_mask), 'tpl_mask_md5': _md5(tpl_mask), 'registration': REGISTRATION,
           'init': str(init) if init else None}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'reg_')
        cmd = [script, '-d', '3', '-f', str(tpl_t1), '-m', str(t1_brain), '-x', str(tpl_mask), '-o', out] + \
              (['-n', str(n_threads())] if n_threads() else []) + (['-i', str(init)] if init else [])
        _section(f'registration to {tpl_name} (antsRegistrationSyN.sh)')
        lines = _run(cmd, env=_ants_env()).splitlines()
        i = next((k for k, line in enumerate(lines) if 'antsRegistration call:' in line), None)
        res['ants_call'] = ' '.join(lines[i + 2].split()) if i is not None and len(lines) > i + 2 else None
        for key, suffix in (('affine', '0GenericAffine.mat'), ('warp', '1Warp.nii.gz'),
                            ('inverse_warp', '1InverseWarp.nii.gz'), ('warped', 'Warped.nii.gz')):
            res[key] = str(out_dir / f'{prefix}_to-{tpl_name}_{suffix}')
            shutil.copy(out + suffix, res[key])
    return res


def rotation_search_init(t1_brain, tpl_t1, out_mat, spacing=4, search_degrees=30):
    """Start transform of register_to_template (init) for a head that lies rotated in the field of view, which the
    alignment of the centres of mass of antsRegistrationSyN.sh does not correct: antsAI, rigid, Mattes MI, all rotations
    in steps of search_degrees about each axis, of the skull-stripped T1 on the skull-stripped template, both resampled
    to `spacing` mm for speed. Writes out_mat (ITK, template = fixed) and returns its path."""
    with tempfile.TemporaryDirectory() as tmp:
        fixed, moving = os.path.join(tmp, 'fixed.nii.gz'), os.path.join(tmp, 'moving.nii.gz')
        run_ants('ResampleImageBySpacing', 3, tpl_t1, fixed, spacing, spacing, spacing, 0)
        run_ants('ResampleImageBySpacing', 3, t1_brain, moving, spacing, spacing, spacing, 0)
        _section(f'rotation search (antsAI, {search_degrees} degree steps, {spacing} mm)')
        run_ants('antsAI', '-d', 3, '-m', f'Mattes[{fixed},{moving},32,Regular,0.25]', '-t', 'Rigid[0.1]',
                 '-s', f'[{search_degrees},1]', '-c', 10, '-o', out_mat, '-v', 1)
    return str(out_mat)


def rotation_start(init, t1_brain, tpl_t1, out_mat):
    """Start transform of register_to_template for tpl_t1 with the rotation of another start transform (init: of
    rotation_search_init against the BCP template, whose axes are those of MNI) and the centres of mass of tpl_t1 and
    t1_brain aligned. For MNI: a rotation search against MNI itself picked a wrong rotation in infants below about 11
    months (adult vs infant contrast; notebook 3, 3.2: principal axes of the moved brain mask 31-57 degrees off MNI,
    4-12 degrees with this start). Writes out_mat (ITK, template = fixed) and returns its path."""
    import ants
    tr = ants.read_transform(str(init))
    rot = np.column_stack([tr.apply_to_vector(list(e)) for e in np.eye(3)])  # rotation part of init
    c_fix = np.array(ants.get_center_of_mass(ants.image_read(str(tpl_t1))))
    c_mov = np.array(ants.get_center_of_mass(ants.image_read(str(t1_brain))))
    start = ants.create_ants_transform('AffineTransform', precision='double', dimension=3, matrix=rot,
                                       center=list(c_fix), translation=list(c_mov - c_fix))
    ants.write_transform(start, str(out_mat))
    return str(out_mat)


def registration_qc(reg, brain_mask, t1_cerebellum=None, tpl_cerebellum=None, out_png=None, title='',
                    full_jacobian=False):
    """QC of one registration (reg: the dict of register_to_template): correlation of the warped t1_brain (reg['warped']) with
    the template T1 inside the template mask; Dice and HD95 (mm, 95th percentile surface distance; also log10(HD95),
    as HD95 is right-skewed) of the warped brain_mask with the template mask, and of the warped t1_cerebellum with
    tpl_cerebellum if both are given; Jacobian determinant inside the template mask (minimum, share <= 0) of the warp
    (reg['warp'], 1Warp: without the affine), or with full_jacobian of the whole transform (all transforms composed into
    one field: with the affine, e.g. the growth between two templates; tpl2tpl_edge, avgsubj_edge; then 'jacobian_of':
    'whole transform'); with out_png, the figure registration_qc_figure. No thresholds yet. Returns a dict."""
    import ants
    binary_like = lambda ref, f: ref.new_image_like((_as_3d(ants.image_read(str(f))).numpy() > 0).astype('float32'))
    t1_img = _as_3d(ants.image_read(reg['t1_brain']))
    fixed = _as_3d(ants.image_read(reg['tpl_t1']))
    tm = binary_like(fixed, reg['tpl_mask']).numpy() > 0
    fwd = reg.get('transforms') or [reg['warp'], reg['affine']]  # 'transforms': e.g. one field (avgsubj_edge)
    warped = (ants.image_read(reg['warped']) if reg.get('warped') else ants.apply_transforms(fixed, t1_img, fwd)).numpy()
    qc = {'t1_corr': float(np.corrcoef(warped[tm], fixed.numpy()[tm])[0, 1])}
    warped_mask = ants.apply_transforms(fixed, binary_like(t1_img, brain_mask), fwd, interpolator='nearestNeighbor')
    warped_mask = warped_mask.numpy() > 0
    qc['brain_dice'] = float(_dice(warped_mask, tm))
    qc['brain_hd95'] = _hd95(warped_mask, tm, fixed.spacing)
    qc['brain_log10_hd95'] = float(np.log10(qc['brain_hd95']))
    tpl_cereb = None
    if t1_cerebellum and tpl_cerebellum:
        cereb = ants.apply_transforms(fixed, binary_like(t1_img, t1_cerebellum), fwd, interpolator='nearestNeighbor')
        tpl_cereb = binary_like(fixed, tpl_cerebellum).numpy() > 0
        cereb = cereb.numpy() > 0
        qc['cerebellum_dice'] = float(_dice(cereb, tpl_cereb))
        qc['cerebellum_hd95'] = _hd95(cereb, tpl_cereb, fixed.spacing)
        qc['cerebellum_log10_hd95'] = float(np.log10(qc['cerebellum_hd95']))
    with tempfile.TemporaryDirectory() as tmp:
        field = reg['warp']
        if full_jacobian:
            field = os.path.join(tmp, 'field.nii')
            run_ants('antsApplyTransforms', '-d', 3, '-r', reg['tpl_t1'], '-o', f'[{field},1]',
                     *[x for t in fwd for x in ('-t', t)])
        jd = ants.create_jacobian_determinant_image(fixed, field).numpy()[tm]
    if full_jacobian:
        qc['jacobian_of'] = 'whole transform'
    qc['jacobian_min'] = float(jd.min())
    qc['jacobian_nonpositive_share'] = float((jd <= 0).mean())
    if out_png:
        hd95 = lambda k: f'HD95 {qc[k + "_hd95"]:.1f} mm (log10 {qc[k + "_log10_hd95"]:.2f})'
        text = (f'{title} | T1 r {qc["t1_corr"]:.3f} | brain Dice {qc["brain_dice"]:.3f}, {hd95("brain")}'
                + (f' | cerebellum Dice {qc["cerebellum_dice"]:.3f}, {hd95("cerebellum")}'
                   if 'cerebellum_dice' in qc else '')
                + f' | min Jacobian {qc["jacobian_min"]:.2f}')
        qc['qc_figure'] = registration_qc_figure(fixed.numpy(), warped, tm, tpl_cereb, out_png, text)
    return qc


# ---------------------------------------------------------------------------
# One visit: step 1 (preprocess) and step 2 (direct_normalize)
# ---------------------------------------------------------------------------

def preprocess(t1, age_months, out_dir, t2=None, prefix=None, t1_brain=None, save_brain=False, always_qc_figure=False):
    """Step 1 of the pipeline, preprocessing of one visit in three parts:
      1a. orient: the T1 (and T2) in the standard orientation (fslreorient2std; a copy only if needed,
          <prefix>_T1w_reorient.nii.gz / _T2w_reorient.nii.gz);
      1b. skull_strip_step: SynthStrip brain mask, <prefix>_T1w_brain_mask.nii.gz (below UNION_BELOW_MONTHS the
          default OR the infant model); with a T2, the T2 registered rigidly to the T1 (whole head),
          <prefix>_T2w_rigid2T1.nii.gz and _0GenericAffine.mat;
      1c. QC: SUITPy cerebellum of the T1 (twice) inside the brain mask, brain volume for the age (needs_visual_check;
          <prefix>_skullstrip_qc.png if flagged, or always with always_qc_figure).
    t1, t2 (optional): the images as acquired (whole head); age_months: age at scan (chooses the SynthStrip models
    and the age bin of the volume check). Writes <prefix>_preprocess.json in out_dir (inputs, software, 'orient',
    'skull_strip': outputs and QC) and returns its content; step 2 (direct_normalize) uses orient.t1,
    skull_strip.brain_mask and skull_strip.t2_on_t1."""
    _require(need_infant=age_months < UNION_BELOW_MONTHS)
    prefix = prefix or default_prefix(t1)
    out_dir = Path(out_dir)
    log = {'inputs': {'t1': str(t1), 't2': str(t2) if t2 else None, 'age_months': float(age_months), 'prefix': prefix},
           'software': environment_info()}
    log['orient'] = orient(t1, t2, out_dir, prefix)
    log['skull_strip'] = skull_strip_step(log['orient']['t1'], age_months, log['orient']['t2'], out_dir, prefix,
                                          t1_brain, save_brain, always_qc_figure=always_qc_figure)
    with open(out_dir / f'{prefix}_preprocess.json', 'w') as f:
        json.dump(log, f, indent=2, default=str)
    return log


def direct_normalize(t1, age_months, brain_mask, out_dir=None, t2=None, prefix=None, steps=(1, 2, 3), bcp_dir=None,
              mni_dir=None, rotation_init=True, templates=None):
    """Direct normalisation of one visit (step 2 of the pipeline): the visit registered to each template itself, in
    three parts; `steps` chooses which run (1, 2, 3 = parts 2a, 2b, 2c; files of 2b and 2c: <prefix>_direct_*):
      2a. head2brain_and_n4correction(t1, brain_mask, t2): skull stripping, N4 -> <prefix>_T1w_brain_n4.nii.gz (+ T2);
      2b. register_to_template for every template of `templates` (names; default normalization_targets(age_months):
         the BCP template of the nearest atlas month and MNI; e.g. the other templates of the visit's avgsubj edges,
         avgsubj_edges) -> transforms and warped T1; templates registered before and not in `templates` are kept;
         with rotation_init (default), every registration starts from the rotation of rotation_search_init against
         the BCP template of the age (bcp_template; its init.mat is reused if that template is not registered again),
         for the other templates with the centres of mass aligned (rotation_start), so that a head that lies rotated
         in the field of view is registered too (why for every visit: notebook 3, 3.2); <prefix>_direct_to-<template>
         _init.mat, a record only: it is inside 0GenericAffine.mat. A rotation search against MNI itself picked a
         wrong rotation below about 11 months (adult vs infant contrast). rotation_init=False: the centre-of-mass
         start of antsRegistrationSyN.sh;
      2c. direct_normalization_qc -> reference cerebellum (brain_to_cerebellum), QC, figures, <prefix>_direct_normalize_qc.json.
    The registrations use the T1 only (for now; a T1 + T2 normalisation may follow).
    t1: T1 of step 1 (orient; not skull-stripped); brain_mask: mask of step 1; t2 (optional): T2 on the T1 grid,
    not skull-stripped (rigid T2 of step 1, <prefix>_T2w_rigid2T1.nii.gz), only skull-stripped and bias corrected in
    2a and used for the SUITPy cerebellum of 2c; out_dir: default the folder of t1. 2a and 2b
    write their results to <prefix>_direct_normalize.json (an existing
    file is updated, so a part can run alone once the parts before it have run). Returns the content of
    <prefix>_direct_normalize.json (with 2c also 'qc': the content of <prefix>_direct_normalize_qc.json)."""
    prefix = prefix or default_prefix(t1)
    out_dir = Path(out_dir) if out_dir else Path(t1).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f'{prefix}_direct_normalize.json'
    log = json.loads(log_file.read_text()) if log_file.exists() else {}
    log.update(inputs={'t1': str(t1), 't2': str(t2) if t2 else None, 'brain_mask': str(brain_mask),
                       'age_months': float(age_months), 'prefix': prefix}, software=environment_info())

    def save():
        log_file.write_text(json.dumps(log, indent=2, default=str))
    if 1 in steps:
        log['head2brain'] = head2brain_and_n4correction(t1, brain_mask, t2, out_dir, prefix)
        save()
    if 2 in steps:
        t1_brain = log.get('head2brain', {}).get('t1_brain')
        if not (t1_brain and Path(t1_brain).exists()):
            raise FileNotFoundError(f'{prefix}: no skull-stripped T1 of 2a in {log_file}: run 2a (--steps 1) first')
        targets = ([template_by_name(n, bcp_dir, mni_dir) for n in templates] if templates
                   else normalization_targets(age_months, bcp_dir, mni_dir))
        if rotation_init:
            bcp = bcp_template(age_months, bcp_dir)
            bcp_mat = out_dir / f"{prefix}_direct_to-{bcp['name']}_init.mat"
            bcp_init = str(bcp_mat) if bcp_mat.exists() and bcp['name'] not in [t['name'] for t in targets] \
                else rotation_search_init(t1_brain, bcp['t1'], bcp_mat)
        log.setdefault('normalize', {})
        for tpl in targets:
            init = None
            if rotation_init:
                init = bcp_init if tpl['name'] == bcp['name'] else \
                    rotation_start(bcp_init, t1_brain, tpl['t1'], out_dir / f"{prefix}_direct_to-{tpl['name']}_init.mat")
            reg = register_to_template(t1_brain, tpl['name'], tpl['t1'], tpl['mask'], out_dir, f'{prefix}_direct',
                                       init=init)
            log['normalize'][tpl['name']] = {**reg, 'tpl_cerebellum': tpl['cerebellum']}
        save()
    if 3 in steps:
        if not log.get('normalize'):
            raise FileNotFoundError(f'{prefix}: no registrations of 2b in {log_file}: run 2b (--steps 2) first')
        return {**log, 'qc': direct_normalization_qc(log_file)}
    return log


def direct_normalization_qc(normalize_json, qc_figure=True):
    """Part 2c of the direct normalisation: QC of one visit after the registrations (normalize_json:
    <prefix>_direct_normalize.json).
    The cerebellum of the visit is its reference cerebellum, <prefix>_cerebellum_suitpy_from-brain-N4_dseg.nii.gz of
    brain_to_cerebellum
    (SUITPy on the skull-stripped T1 with the T2 and the brain mask, medoid of 2-3 runs, chosen without any
    registration; the same file as in the evaluation, notebook 5), made here if it is missing (brain_to_cerebellum with the
    T2 of normalize_json, inputs.t2, and the whole-head SUITPy cerebellums of skullstrip/, if any). Until 2026-11-03:
    SUITPy on the T1 alone, run again if its cerebellum Dice with the main template was low, the better run used
    (<prefix>_cerebellum_suitpy_from-T1wbrain*_dseg.nii.gz): that choice depended on the registration, and the T1 alone
    cuts off the inferior cerebellum of infants.
    The template cerebellums are looked up again (template_by_name, in the folder of the registered template T1), not
    taken from normalize_json, so a QC run uses the current ones (bcp_template: -T1_cerebellum_suitpy_dseg).
    registration_qc for every template (figures <prefix>_direct_to-<template>_qc.png). Writes
    <prefix>_direct_normalize_qc.json next to normalize_json ('t1_cerebellum'; 'cerebellum': input and checks of
    brain_to_cerebellum; 'qc' per template; 'needs_visual_check': no cerebellum, or its Dice with the main template, the
    first of normalize_json, below CEREB_DICE_MIN) and returns its content."""
    log = json.loads(Path(normalize_json).read_text())
    out_dir, prefix = Path(normalize_json).parent, log['inputs']['prefix']
    brain_mask = log['inputs']['brain_mask']
    main = next(iter(log['normalize']))
    tpl_cereb = {name: template_by_name(name, Path(reg['tpl_t1']).parents[1], Path(reg['tpl_t1']).parent)['cerebellum']
                 for name, reg in log['normalize'].items()}
    cereb, cereb_json = (brain_cerebellum_files(out_dir, prefix)[k] for k in ('dseg', 'json'))
    if not cereb.exists():
        _section('reference cerebellum (brain_to_cerebellum)')
        brain_to_cerebellum(normalize_json, t2=log['inputs'].get('t2'), head_cerebellums=sorted(
            (out_dir / 'skullstrip').glob(f'{prefix}_cerebellum_suitpy_from-T1wReorientHead*_dseg.nii.gz')))
    c = json.loads(cereb_json.read_text()) if cereb_json.exists() else {}
    found = cereb.exists()
    res = {'normalize_json': str(normalize_json), 'main_template': main, 'cerebellum_dice_min': CEREB_DICE_MIN,
           'tpl_cerebellums': tpl_cereb, 'software': environment_info(), 't1_cerebellum': str(cereb) if found else None,
           'cerebellum': {k: c.get(k) for k in ('input', 'rotate', 'n_found', 'medoid_mean_dice', 'head_dice_best',
                                                'needs_visual_check')}, 'qc': {}}
    for name, reg in log['normalize'].items():
        _section(f'QC of the registration to {name}')
        png = out_dir / f'{prefix}_direct_to-{name}_qc.png' if qc_figure else None
        res['qc'][name] = registration_qc(reg, brain_mask, res['t1_cerebellum'], tpl_cereb[name], png, f'{prefix} -> {name}')
    dice = res['qc'][main].get('cerebellum_dice', np.nan)  # nan without a template cerebellum: nothing to check
    res['needs_visual_check'] = bool(not found or dice < CEREB_DICE_MIN)
    with open(out_dir / f'{prefix}_direct_normalize_qc.json', 'w') as f:
        json.dump(res, f, indent=2, default=str)
    return res


def _job_script(job, name, log_dir, commands, setup, cpus, mem, time):
    """Write <log_dir>/<name>.sh: Slurm settings (job name `job`, the short name squeue shows; output
    <log_dir>/<name>.out), setup, ANTSPATH / NEONORM_* of this Python session, commands. It runs with sbatch (one job)
    or with bash inside a task of a job array (submit_direct_normalize_array). A new script replaces the old one.
    Returns its path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    exports = [f'export {k}={shlex.quote(os.environ[k])}' for k in
               ('ANTSPATH', 'NEONORM_BCP_TEMPLATES', 'NEONORM_MNI_TEMPLATE', 'NEONORM_NTHREADS') if os.environ.get(k)]
    script = '\n'.join(['#!/bin/bash', f'#SBATCH --job-name={job}', f'#SBATCH --cpus-per-task={cpus}',
                        f'#SBATCH --mem={mem}', f'#SBATCH --time={time}', f'#SBATCH --output={log_dir}/{name}.out',
                        setup, *exports, 'set -e', *[' '.join(shlex.quote(str(c)) for c in cmd) for cmd in commands]]) + '\n'
    sh = log_dir / f'{name}.sh'
    sh.write_text(script)
    return sh


def _sbatch(script, *options):
    """sbatch --parsable <options> <script>; returns the job id."""
    r = subprocess.run(['sbatch', '--parsable', *options, str(script)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'sbatch failed: {r.stderr.strip()}')
    return r.stdout.strip().split(';')[0]


NORM_JOB = {'cpus': 8, 'mem': '16G', 'time': '04:00:00'}  # Slurm resources of 2a-2b (registrations)
QC_JOB = {'cpus': 4, 'mem': '24G', 'time': '01:00:00'}    # 2c (SUITPy needs about 20G)


def direct_normalize_job_scripts(t1, age_months, brain_mask, out_dir=None, t2=None, prefix=None, steps=(1, 2, 3),
                                 setup='', rotation_init=True, templates=None):
    """Write, without submitting, the Slurm job scripts of direct_normalize() for one visit: 2a-2b (`steps` 1, 2,
    if in `steps`) -> <out_dir>/log/<prefix>_direct_normalize.sh (job <number>_direct, NORM_JOB; output <prefix>_direct_normalize.out;
    with `templates`: <prefix>_direct_normalize_to-<template>[-<template>].sh / .out, so earlier logs are kept)
    and 2c (3, if in `steps`) -> <prefix>_direct_normalize_qc.sh (job <number>_directqc, QC_JOB; output
    <prefix>_direct_normalize_qc.out). <number>: the prefix without 'sub-'. Each script runs `python neonorm.py
    --direct_normalize ... --steps ...` (and --no_rotation_init, --templates), i.e. direct_normalize(). Used by submit_direct_normalize (one visit) and
    submit_direct_normalize_array (many visits). Returns {'prefix', 'age_months', 't1', 't2', 'brain_mask', 'templates',
    'steps', 'rotation_init'} and 'script', 'log' (2a-2b) / 'qc_script', 'qc_log' (2c)."""
    prefix = prefix or default_prefix(t1)
    names = list(templates) if templates else [tpl['name'] for tpl in normalization_targets(age_months)]
    for name in names:
        template_by_name(name)  # fails here if a template file is missing
    out_dir = Path(out_dir) if out_dir else Path(t1).parent
    log_dir = out_dir / 'log'
    number = prefix.replace('sub-', '')
    if 3 in steps and 2 not in steps and not (out_dir / f'{prefix}_direct_normalize.json').exists():
        raise FileNotFoundError(f'{out_dir}/{prefix}_direct_normalize.json missing: run 2a-2b (--steps 1 2) first')
    cmd = ['python', Path(__file__).resolve(), '--direct_normalize', t1, age_months, out_dir,
           '--brain_mask', brain_mask, '--prefix', prefix] + (['--t2', t2] if t2 else []) + \
          ([] if rotation_init else ['--no_rotation_init']) + (['--templates', *names] if templates else [])
    res = {'prefix': prefix, 'age_months': float(age_months), 't1': str(t1), 't2': str(t2) if t2 else None,
           'brain_mask': str(brain_mask), 'templates': names, 'steps': tuple(steps),
           'rotation_init': bool(rotation_init)}
    first = [s for s in (1, 2) if s in steps]
    if first:
        name = f'{prefix}_direct_normalize' + (f"_to-{'-'.join(names)}" if templates else '')
        res['script'] = str(_job_script(f'{number}_direct', name, log_dir, [cmd + ['--steps', *first]], setup,
                                        **NORM_JOB))
        res['log'] = str(log_dir / f'{name}.out')
    if 3 in steps:
        res['qc_script'] = str(_job_script(f'{number}_directqc', f'{prefix}_direct_normalize_qc', log_dir,
                                           [cmd + ['--steps', 3]], setup, **QC_JOB))
        res['qc_log'] = str(log_dir / f'{prefix}_direct_normalize_qc.out')
    return res


def submit_direct_normalize(t1, age_months, brain_mask, out_dir=None, t2=None, prefix=None, steps=(1, 2, 3), setup='',
                            rotation_init=True, templates=None):
    """direct_normalize() of one visit as Slurm jobs; computes nothing itself. Writes the job scripts
    (direct_normalize_job_scripts) and submits them: 2a-2b as job <number>_direct, 2c as job <number>_directqc
    after the first job has finished successfully. A new submission replaces the scripts and outputs of its jobs only, so running 2c again keeps the
    log of the registrations. setup: shell lines that prepare the Python environment on your cluster (module load /
    conda activate). Returns the dict of direct_normalize_job_scripts with 'job_id' / 'qc_job_id'. For many visits use
    submit_direct_normalize_array."""
    res = direct_normalize_job_scripts(t1, age_months, brain_mask, out_dir, t2, prefix, steps, setup, rotation_init,
                                       templates)
    if 'script' in res:
        res['job_id'] = _sbatch(res['script'])
    if 'qc_script' in res:
        res['qc_job_id'] = _sbatch(res['qc_script'], *([f'--dependency=afterok:{res["job_id"]}'] if 'job_id' in res else []))
    return res


def submit_direct_normalize_array(visits, run_dir, steps=(1, 2, 3), setup='', max_running=50, qc_job=None):
    """direct_normalize() of many visits as two Slurm job arrays, so that all waiting happens in the Slurm queue: array
    `direct` (2a-2b, one task per visit, NORM_JOB) and array `directqc` (2c, QC_JOB), whose task i starts once
    task i of `direct` has finished successfully (--dependency=aftercorr; it is cancelled if that task failed,
    --kill-on-invalid-dep=yes). At most max_running tasks of each array run at the same time; the others wait in the
    queue. qc_job: Slurm resources of `directqc` instead of QC_JOB (e.g. less for 2c again, when the SUITPy files
    exist and are reused). visits: one dict per visit with the arguments of direct_normalize_job_scripts (t1,
    age_months, brain_mask; optional out_dir, t2, prefix, rotation_init, templates). Each task runs the job script of its visit with bash, so
    scripts and outputs per visit are the same as with submit_direct_normalize (<out_dir>/log/<prefix>_direct_normalize.sh
    / .out, <prefix>_direct_normalize_qc.sh / .out). run_dir gets the task lists (<array>_tasks.tsv: script and output,
    line i = task i), the array scripts and the Slurm output
    per task (<array>_<job id>_<task>.out: only messages of Slurm itself, e.g. a memory or time limit). Slurm limits: at
    most MaxArraySize - 1 tasks per array, and every task counts for MaxSubmitJobs (sacctmgr show assoc). Returns
    {'job_id', 'qc_job_id', 'run_dir', 'visits': the dicts of direct_normalize_job_scripts, each with 'task' (array
    index)}."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    visits = [{**direct_normalize_job_scripts(**v, steps=steps, setup=setup), 'task': i}
              for i, v in enumerate(visits, 1)]
    res, after = {'run_dir': str(run_dir), 'visits': visits}, []
    for key, array, job in (('', 'direct', NORM_JOB), ('qc_', 'directqc', qc_job or QC_JOB)):
        if f'{key}script' not in visits[0]:
            continue
        tasks = run_dir / f'{array}_tasks.tsv'
        tasks.write_text(''.join(f"{v[key + 'script']}\t{v[key + 'log']}\n" for v in visits))
        sh = run_dir / f'{array}_array.sh'
        sh.write_text('\n'.join([
            '#!/bin/bash', f'#SBATCH --job-name={array}', f'#SBATCH --array=1-{len(visits)}%{max_running}',
            f'#SBATCH --cpus-per-task={job["cpus"]}', f'#SBATCH --mem={job["mem"]}', f'#SBATCH --time={job["time"]}',
            f'#SBATCH --output={run_dir}/{array}_%A_%a.out',
            f'IFS=$\'\\t\' read -r script out < <(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {shlex.quote(str(tasks))})',
            'bash "$script" > "$out" 2>&1']) + '\n')
        res[f'{key}job_id'] = _sbatch(sh, *after)
        after = [f'--dependency=aftercorr:{res[f"{key}job_id"]}', '--kill-on-invalid-dep=yes']
    return res


# ---------------------------------------------------------------------------
# Indirect normalisation (notebook 4, evaluation phase): visit -> BCP template -> ... -> MNI
# File names (way = direct / tpl2tpl / avgsubj): edge (one step) <mov>_<way>_to-<fix>_*; visit along a path
# <id>_<way>-<path>_to-<target>_*, <path> = all BCP months passed, start first, joined by '-' (e.g. 00-06; no shorthand)
# ---------------------------------------------------------------------------

def template_by_name(name, bcp_dir=None, mni_dir=None):
    """Template dict (as bcp_template / mni_template) of a name: 'BCP03M' or MNI_NAME."""
    return mni_template(mni_dir) if name == MNI_NAME else bcp_template(int(name[3:-1]), bcp_dir)


def adjacent_chain():
    """Template names from the youngest BCP template to MNI in steps of one atlas month:
    ['BCP00M', 'BCP01M', ..., 'BCP60M', 'MNI152NLin2009cSym']."""
    return [f'BCP{m:02d}M' for m in BCP_MONTHS] + [MNI_NAME]


def tpl2tpl_edge(mov, fix, out_dir, bcp_dir=None, mni_dir=None, qc_figure=True, register=True):
    """tpl2tpl: registration of template `mov` to template `fix` (names as template_by_name), the same as a visit:
    register_to_template with the skull-stripped mov T1 as moving image (antsRegistrationSyN.sh defaults, T1 only, fix
    mask), then registration_qc as in direct_normalization_qc (T1 r; Dice, HD95, log10(HD95) of the mov brain mask and
    of the cerebellum; figure), but the Jacobian of the whole transform (0GenericAffine + 1Warp, full_jacobian; as
    avgsubj_edge, whose average field has the affine in it). The cerebellums are the template cerebellums that are the
    reference of the direct QC (BCP-<m>M-T1_cerebellum_suitpy_dseg, MNI cerebellum_mask), so no SUITPy run;
    'needs_visual_check' as in direct_normalization_qc: cerebellum Dice below CEREB_DICE_MIN. register=False: the
    registration of the json is kept, only the QC again (e.g. after a change of the QC). Saved in out_dir:
    <mov>_tpl2tpl_to-<fix>_{0GenericAffine.mat, 1Warp.nii.gz, 1InverseWarp.nii.gz, Warped.nii.gz}, _qc.png and
    <mov>_tpl2tpl_to-<fix>.json (registration, QC, software), whose content it returns."""
    m, f = template_by_name(mov, bcp_dir, mni_dir), template_by_name(fix, bcp_dir, mni_dir)
    if register:
        reg = register_to_template(m['t1'], f['name'], f['t1'], f['mask'], out_dir, prefix=f'{mov}_tpl2tpl')
    else:
        old = json.loads((Path(out_dir) / f'{mov}_tpl2tpl_to-{fix}.json').read_text())
        reg = {k: v for k, v in old.items() if k not in ('mov_mask', 'mov_cerebellum', 'tpl_cerebellum', 'qc',
                                                          'cerebellum_dice_min', 'needs_visual_check', 'software')}
    png = Path(out_dir) / f'{mov}_tpl2tpl_to-{fix}_qc.png' if qc_figure else None
    _section(f'QC of {mov} -> {fix}')
    qc = registration_qc(reg, m['mask'], m['cerebellum'], f['cerebellum'], png, f'{mov} -> {fix}', full_jacobian=True)
    res = {**reg, 'mov_mask': m['mask'], 'mov_cerebellum': m['cerebellum'], 'tpl_cerebellum': f['cerebellum'], 'qc': qc,
           'cerebellum_dice_min': CEREB_DICE_MIN, 'needs_visual_check': not qc['cerebellum_dice'] >= CEREB_DICE_MIN,
           'software': environment_info()}
    (Path(out_dir) / f'{mov}_tpl2tpl_to-{fix}.json').write_text(json.dumps(res, indent=2, default=str))
    return res


def submit_tpl2tpl(pairs, out_dir, setup='', register=True):
    """tpl2tpl_edge of each (mov, fix) of pairs as one Slurm job (NORM_JOB; job t2t_<mov>_<fix>), script and output in
    <out_dir>/log/<mov>_tpl2tpl_to-<fix>.sh / .out; register=False: the QC alone (tpl2tpl_edge). Returns one dict per
    pair: mov, fix, script, log, job_id."""
    out_dir = Path(out_dir)
    jobs = []
    for mov, fix in pairs:
        code = (f'import sys; sys.path.insert(0, {str(HERE)!r}); import neonorm; neonorm.SHOW_TOOL_OUTPUT = True; '
                f'neonorm.tpl2tpl_edge({mov!r}, {fix!r}, {str(out_dir)!r}, register={register!r})')
        sh = _job_script(f't2t_{mov}_{fix}', f'{mov}_tpl2tpl_to-{fix}', out_dir / 'log', [['python', '-c', code]], setup,
                         **NORM_JOB)
        jobs.append({'mov': mov, 'fix': fix, 'script': str(sh),
                     'log': str(out_dir / 'log' / f'{mov}_tpl2tpl_to-{fix}.out'), 'job_id': _sbatch(sh)})
    return jobs


def chain_transforms(path, xfm_dir):
    """Transforms (antsApplyTransforms / ants.apply_transforms order, the last applied first) that move an image of
    template path[0] along the templates of `path` to path[-1] with the tpl2tpl registrations in xfm_dir; one
    interpolation however many steps. E.g. ['BCP03M', 'BCP04M', 'MNI152NLin2009cSym']."""
    tx = []
    for a, b in zip(path[:-1], path[1:]):
        edge = Path(xfm_dir) / f'{a}_tpl2tpl_to-{b}'
        tx = [f'{edge}_1Warp.nii.gz', f'{edge}_0GenericAffine.mat'] + tx
    return tx


TEMPLATE_SUITPY_RUNS = 5  # SUITPy runs per BCP template (bcp_template_cerebellum); the medoid run is its cerebellum
TEMPLATE_CEREB_JOB = {'cpus': 4, 'mem': '24G', 'time': '06:00:00'}  # Slurm resources of one template (TEMPLATE_SUITPY_RUNS runs)


def template_cerebellum_figure(t1, new, old, out_png, text, old_label='earlier template cerebellum'):
    """T1 of a template (or of a visit, brain_to_cerebellum) through the centre of the new cerebellum (three planes): new
    cerebellum filled, the other cerebellum `old` (if any; earlier template cerebellum) dashed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img = np.squeeze(np.asanyarray(nib.load(t1).dataobj)).astype(float)
    c = np.round(np.mean(np.nonzero(new), axis=1)).astype(int)
    vmax = np.percentile(img[img > 0], 99)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for axis, ax in enumerate(axes):
        sl = lambda v: np.take(v, c[axis], axis=axis).T
        ax.imshow(sl(img), cmap='gray', vmin=0, vmax=vmax, origin='lower')
        ax.contourf(sl(new).astype(float), levels=[0.5, 1.5], colors=['#1baf7a'], alpha=0.35)
        if old is not None:
            ax.contour(sl(old).astype(float), levels=[0.5], colors='#e4572e', linestyles='--', linewidths=1.2)
        ax.axis('off')
    handles = [plt.Line2D([], [], color='#1baf7a', lw=6, alpha=0.4, label='SUITPy cerebellum (medoid run)')]
    if old is not None:
        handles.append(plt.Line2D([], [], color='#e4572e', ls='--', lw=2, label=old_label))
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle(text, fontsize=9, ha='left', x=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return str(out_png)


def bcp_template_cerebellum(name, runs=TEMPLATE_SUITPY_RUNS, bcp_dir=None, qc_figure=True):
    """SUITPy cerebellum of a BCP template (name as template_by_name, e.g. 'BCP03M') from the template T1 with its T2
    and template mask (with the T1 alone, SUITPy cuts off the inferior cerebellar hemispheres of the 03-15M templates).
    SUITPy is not deterministic, so it runs `runs` times. The cerebellum is the medoid run: the highest mean Dice with
    the other runs (a real SUITPy output, no voting). 'needs_visual_check': fewer than half of the runs found a
    cerebellum, or the medoid's mean Dice with the other runs is below DICE_MIN (the run agreement of the skull strip
    QC).
    Saved in <template folder>/cerebellum_suitpy/: every run, <base>-T1_cerebellum_suitpy_from-<input>_run<i>_dseg.nii.gz
    (existing run files are used again), <base>-T1_cerebellum_suitpy.json (runs, input used, Dice between the runs,
    medoid, Dice and HD95 with the earlier template cerebellum -T1_cerebellum_dseg.nii.gz, NaN if it is no longer
    there) and, with qc_figure, <base>-T1_cerebellum_suitpy_qc.png; the medoid as <base>-T1_cerebellum_suitpy_dseg.nii.gz
    next to the template. Returns the json content."""
    tpl = template_by_name(name, bcp_dir)
    base = Path(tpl['t1'][:-len('-T1.nii.gz')])
    run_dir = base.parent / 'cerebellum_suitpy'
    run_dir.mkdir(exist_ok=True)
    stem = f'{base.name}-T1_cerebellum_suitpy'
    need = (runs + 1) // 2
    res = {'template': name, 't1': tpl['t1'], 'runs_per_input': runs, 'software': environment_info(), 'runs': []}
    tag, found = 'T1wT2wbrain', []
    for i in range(1, runs + 1):
        f = run_dir / f'{stem}_from-{tag}_run{i}_dseg.nii.gz'
        _section(f'{name}: SUITPy from {tag}, run {i}')
        ok = f.exists() or suitpy_cerebellum(tpl['t1'], f, t2=f'{base}-T2.nii.gz', brain_mask=tpl['mask']) is not None
        res['runs'].append({'input': tag, 'run': i, 'dseg': str(f) if ok else None})
        found += [f] if ok else []
    masks = [_load_bool(f) for f in found]
    dice = np.array([[_dice(a, b) for b in masks] for a in masks])
    mean = [(dice[i].sum() - 1) / (len(masks) - 1) if len(masks) > 1 else np.nan for i in range(len(masks))]
    k = int(np.nanargmax(mean)) if len(masks) > 1 else 0
    res.update(input_used=tag, n_found=len(found), runs_dice=dice.round(4).tolist(),
               medoid=str(found[k]) if found else None, medoid_mean_dice=float(mean[k]) if found else np.nan)
    res['needs_visual_check'] = len(found) < need or not res['medoid_mean_dice'] >= DICE_MIN
    if found:
        res['cerebellum'] = f'{base}-T1_cerebellum_suitpy_dseg.nii.gz'
        shutil.copy(found[k], res['cerebellum'])
        earlier = Path(f'{base}-T1_cerebellum_dseg.nii.gz')  # one SUITPy run, before bcp_template_cerebellum
        old = _load_bool(earlier) if earlier.exists() else None
        res['earlier_cerebellum'] = str(earlier) if old is not None else None
        res['dice_with_earlier'] = float(_dice(masks[k], old)) if old is not None else np.nan
        res['hd95_with_earlier'] = (_hd95(masks[k], old, nib.load(tpl['t1']).header.get_zooms()[:3])
                                    if old is not None else np.nan)
        if qc_figure:
            text = (f'{name} | SUITPy from {tag}: {len(found)}/{runs} runs found a cerebellum | medoid run '
                    f'{Path(found[k]).name.split("_run")[1][0]}, mean Dice with the other runs '
                    f'{res["medoid_mean_dice"]:.3f} | Dice with the earlier cerebellum {res["dice_with_earlier"]:.3f}')
            res['qc_figure'] = template_cerebellum_figure(tpl['t1'], masks[k], old, run_dir / f'{stem}_qc.png', text)
    (run_dir / f'{stem}.json').write_text(json.dumps(res, indent=2, default=str))
    return res


def submit_bcp_template_cerebellum(names, setup='', runs=TEMPLATE_SUITPY_RUNS):
    """bcp_template_cerebellum of each template of names as one Slurm job (TEMPLATE_CEREB_JOB; job tcb_<name>), script
    and output in <template folder>/cerebellum_suitpy/log/<base>-T1_cerebellum_suitpy.sh / .out. Returns one dict per
    template: template, script, log, job_id."""
    jobs = []
    for name in names:
        base = Path(template_by_name(name)['t1'][:-len('-T1.nii.gz')])
        log_dir = base.parent / 'cerebellum_suitpy' / 'log'
        stem = f'{base.name}-T1_cerebellum_suitpy'
        code = (f'import sys; sys.path.insert(0, {str(HERE)!r}); import neonorm; neonorm.SHOW_TOOL_OUTPUT = True; '
                f'neonorm.bcp_template_cerebellum({name!r}, runs={runs})')
        sh = _job_script(f'tcb_{name}', stem, log_dir, [['python', '-c', code]], setup, **TEMPLATE_CEREB_JOB)
        jobs.append({'template': name, 'script': str(sh), 'log': str(log_dir / f'{stem}.out'), 'job_id': _sbatch(sh)})
    return jobs


AVGSUBJ_JOB = {'cpus': 4, 'mem': '16G', 'time': '08:00:00'}  # Slurm resources of one avgsubj edge (16 visits: 3 min)


def avgsubj_visit_transforms(normalize_json, a, b):
    """Transforms (antsApplyTransforms order, the last applied first) and whichtoinvert that move an image of template
    a through a visit to template b with the direct registrations of the visit (normalize_json:
    <prefix>_direct_normalize.json, registered to a and to b): template a -> visit [a: 0GenericAffine inverted,
    1InverseWarp], then visit -> template b [b: 1Warp, 0GenericAffine]; no init.mat (it is in 0GenericAffine.mat)."""
    reg = json.loads(Path(normalize_json).read_text())['normalize']
    return [reg[b]['warp'], reg[b]['affine'], reg[a]['affine'], reg[a]['inverse_warp']], [False, False, True, False]


def _avgsubj_visit_field(normalize_json, a, b, ref, field):
    """Composite field template a -> visit -> template b (avgsubj_visit_transforms) on the grid of ref, written to
    field (antsApplyTransforms -o [field,1]); returns the nibabel image and its data (float64)."""
    tx, inv = avgsubj_visit_transforms(normalize_json, a, b)
    run_ants('antsApplyTransforms', '-d', 3, '-r', ref, '-o', f'[{field},1]',
             *[x for t, n in zip(tx, inv) for x in ('-t', f'[{t},1]' if n else t)])
    img = nib.load(field)
    return img, np.asarray(img.dataobj, dtype=np.float64)


def avgsubj_edge(a, b, visits, out_dir, bcp_dir=None, mni_dir=None, qc_figure=True):
    """avgsubj: the average over `visits` of template a -> visit -> template b (avgsubj_visit_transforms), as one
    displacement field on the grid of b: per visit the composite field (antsApplyTransforms -o [field,1]), then the
    voxelwise mean of the fields. visits: one dict per visit with 'normalize_json' (other keys, e.g. dataset,
    participant_id, child, are kept in the json). QC of template a moved with the average to b: registration_qc as in
    tpl2tpl_edge (T1 r; Dice, HD95 of the brain mask and of the template cerebellums; figure; 'needs_visual_check':
    cerebellum Dice below CEREB_DICE_MIN); the Jacobian is that of the whole average field (with the affine part, so
    also the growth from a to b), as tpl2tpl (full_jacobian).
    Saved in out_dir: <a>_avgsubj_to-<b>_Warp.nii.gz (the average: moves an image of a to b, antsApplyTransforms -t
    <field>), _sum.nii.gz (sum of the fields; the average without one child is (sum - fields of its visits) /
    (n - its visits), the fields made again with avgsubj_visit_transforms), _Warped.nii.gz (T1 of a in b), _qc.png and
    <a>_avgsubj_to-<b>.json (visits, QC, software), whose content it returns."""
    if not visits:
        raise ValueError(f'{a} -> {b}: no visits')
    if n_threads():
        os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(n_threads()))  # ANTs threads
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fa, fb = template_by_name(a, bcp_dir, mni_dir), template_by_name(b, bcp_dir, mni_dir)
    stem = out_dir / f'{a}_avgsubj_to-{b}'
    total = None
    with tempfile.TemporaryDirectory() as tmp:
        field = os.path.join(tmp, 'field.nii')
        for i, v in enumerate(visits, 1):
            _section(f'{a} -> {b}: visit {i}/{len(visits)}, {Path(v["normalize_json"]).parent.name}')
            img, d = _avgsubj_visit_field(v['normalize_json'], a, b, fb['t1'], field)
            total = d if total is None else total + d
            affine, header = img.affine, img.header.copy()
    header.set_data_dtype(np.float32)
    save = lambda data, f: nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header), f) or str(f)
    res = {'way': 'avgsubj', 'mov': a, 'fix': b, 'n_visits': len(visits), 'visits': visits,
           'average': save(total / len(visits), f'{stem}_Warp.nii.gz'), 'sum': save(total, f'{stem}_sum.nii.gz'),
           'warped': f'{stem}_Warped.nii.gz', 'mov_t1': fa['t1'], 'mov_mask': fa['mask'], 'mov_cerebellum': fa['cerebellum'],
           'tpl_t1': fb['t1'], 'tpl_mask': fb['mask'], 'tpl_cerebellum': fb['cerebellum']}
    _section(f'{a} -> {b}: template {a} moved with the average')
    run_ants('antsApplyTransforms', '-d', 3, '-i', fa['t1'], '-r', fb['t1'], '-o', res['warped'], '-t', res['average'])
    reg = {'t1_brain': fa['t1'], 'tpl_t1': fb['t1'], 'tpl_mask': fb['mask'], 'warp': res['average'],
           'transforms': [res['average']], 'warped': res['warped']}
    png = f'{stem}_qc.png' if qc_figure else None
    _section(f'QC of {a} -> {b}')
    qc = registration_qc(reg, fa['mask'], fa['cerebellum'], fb['cerebellum'], png, f'{a} -> {b} (avgsubj, {len(visits)} visits)',
                         full_jacobian=True)
    res.update(qc=qc, cerebellum_dice_min=CEREB_DICE_MIN, needs_visual_check=not qc['cerebellum_dice'] >= CEREB_DICE_MIN,
               software=environment_info())
    Path(f'{stem}.json').write_text(json.dumps(res, indent=2, default=str))
    return res


def submit_avgsubj(edges, out_dir, setup=''):
    """avgsubj_edge of each edge as one Slurm job (AVGSUBJ_JOB; job avg_<a>_<b>). edges: {(a, b): visits} (visits as
    in avgsubj_edge), written to <out_dir>/log/<a>_avgsubj_to-<b>_visits.json, which the job reads; script and output
    <out_dir>/log/<a>_avgsubj_to-<b>.sh / .out. Returns one dict per edge: mov, fix, n_visits, script, log, job_id."""
    out_dir = Path(out_dir)
    log_dir = out_dir / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for (a, b), visits in edges.items():
        name = f'{a}_avgsubj_to-{b}'
        visits_file = log_dir / f'{name}_visits.json'
        visits_file.write_text(json.dumps(visits, indent=2, default=str))
        code = (f'import sys, json; sys.path.insert(0, {str(HERE)!r}); import neonorm; neonorm.SHOW_TOOL_OUTPUT = True; '
                f'neonorm.avgsubj_edge({a!r}, {b!r}, json.load(open({str(visits_file)!r})), {str(out_dir)!r})')
        sh = _job_script(f'avg_{a}_{b}', name, log_dir, [['python', '-c', code]], setup, **AVGSUBJ_JOB)
        jobs.append({'mov': a, 'fix': b, 'n_visits': len(visits), 'script': str(sh), 'log': str(log_dir / f'{name}.out'),
                     'job_id': _sbatch(sh)})
    return jobs


TO_MNI_JOB = {'cpus': 4, 'mem': '16G', 'time': '04:00:00'}  # Slurm resources of one visit, all of its ways (visit_to_mni)


def path_months(path):
    """The <path> of the visit file names: ['BCP00M', 'BCP01M', 'BCP02M'] -> '00-01-02'."""
    return '-'.join(name[3:-1] for name in path)


def avgsubj_field_without(a, b, child, out, xfm_dir):
    """The avgsubj field of edge a -> b (<xfm_dir>/<a>_avgsubj_to-<b>.json) without the visits of `child` (way
    avgsubjl1o of visit_to_mni): the saved average if child is None or has no visit in the edge, else (sum - their fields) / (n - their number), the fields made again (avgsubj_visit_transforms),
    written to `out` (float32). Returns {'edge', 'field', 'n_visits' (in the field), 'left_out' (visits of the
    child)}."""
    e = json.loads((Path(xfm_dir) / f'{a}_avgsubj_to-{b}.json').read_text())
    own = [v for v in e['visits'] if v.get('child') == child] if child is not None else []  # None: the average
    res = {'edge': f'{a} -> {b}', 'field': e['average'], 'n_visits': e['n_visits'], 'left_out': len(own)}
    if not own:
        return res
    if len(own) == e['n_visits']:
        raise ValueError(f'{a} -> {b}: all visits are of child {child}')
    img = nib.load(e['sum'])
    total = np.asarray(img.dataobj, dtype=np.float64)
    with tempfile.TemporaryDirectory() as tmp:
        for v in own:
            _section(f'{a} -> {b}: average without {Path(v["normalize_json"]).parent.name}')
            total -= _avgsubj_visit_field(v['normalize_json'], a, b, e['tpl_t1'], os.path.join(tmp, 'field.nii'))[1]
    nib.save(nib.Nifti1Image((total / (e['n_visits'] - len(own))).astype(np.float32), img.affine, img.header), out)
    return {**res, 'field': str(out), 'n_visits': e['n_visits'] - len(own)}


VISIT_WAYS = ('direct', 'tpl2tpl', 'avgsubj', 'avgsubjl1o')  # ways of visit_to_mni


def visit_to_mni(normalize_json, way, path=(), child=None, xfm_root=None, mni_dir=None):
    """Step 3 of the pipeline, in three parts (3a the transforms composed into one field, 3b the images moved with it
    and its log Jacobian, 3c mni_metrics): one visit to MNI by one way (VISIT_WAYS), for the evaluation: 'direct' (its MNI registration of notebook 3, no
    path), or along `path` (BCP template names from the start template, e.g. the adjacent chain ['BCP03M', ...,
    'BCP60M'], then MNI) visit -> path[0] (its direct registration) + per edge: 'tpl2tpl' the registration
    (chain_transforms), 'avgsubj' the average of all visits of the edge (<a>_avgsubj_to-<b>_Warp.nii.gz), 'avgsubjl1o'
    the average without the visits of the visit's `child` (avgsubj_field_without, leave one child out: the visit's own
    registrations are then not in the average; e.g. for a test of a systematic difference between avgsubj and direct).
    All transforms are composed into one field on the MNI grid (antsApplyTransforms -o [field,1]; not kept), which
    moves, with one interpolation, the skull-stripped, bias-corrected T1 of the visit (t1_brain of normalize_json,
    linear) and its SUITPy cerebellum (nearest neighbour: the reference cerebellum of the evaluation,
    <prefix>_cerebellum_suitpy_from-brain-N4_dseg.nii.gz of brain_to_cerebellum, if it exists, else t1_cerebellum of
    <prefix>_direct_normalize_qc.json; which one: inputs.cerebellum of the json) and its brain mask (nearest
    neighbour); the log Jacobian determinant of the field (CreateJacobianDeterminantImage; MNI brain mask, 0 outside)
    is the local volume change. The evaluation values at MNI (mni_metrics) go into the json ('metrics').
    Saved next to normalize_json: <prefix>_<way>[-<path_months>]_to-MNI152NLin2009cSym_T1w.nii.gz (float32),
    _cerebellum_dseg.nii.gz (uint8), _brain_mask.nii.gz (uint8), _logjd.nii.gz (float32) and .json (way, path, n_steps,
    start_month, child, edges, transforms, the ANTs command lines in the order run and the steps in Python ('<tmp>':
    the temporary folder, deleted), files, metrics, software), whose content it returns."""
    global ANTS_COMMANDS
    if n_threads():
        os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(n_threads()))  # ANTs threads
    nj = Path(normalize_json)
    norm = json.loads(nj.read_text())
    qc = json.loads(nj.with_name(nj.name.replace('_direct_normalize.json', '_direct_normalize_qc.json')).read_text())
    prefix, path, mni = norm['inputs']['prefix'], list(path), mni_template(mni_dir)
    cereb = brain_cerebellum_files(nj.parent, prefix)['dseg']  # reference of the evaluation (brain_to_cerebellum)
    cereb = str(cereb) if cereb.exists() else qc['t1_cerebellum']
    if (way not in VISIT_WAYS or (way == 'direct') == bool(path) or (path and xfm_root is None)
            or (way == 'avgsubjl1o' and child is None)):
        raise ValueError(f'{way}, path {path}: direct takes no path, the others a path and xfm_root, avgsubjl1o a child')
    stem = nj.parent / (f'{prefix}_{way}' + (f'-{path_months(path)}' if path else '') + f'_to-{MNI_NAME}')
    steps = list(zip(path, path[1:] + [MNI_NAME]))
    reg = norm['normalize'][path[0] if path else MNI_NAME]
    files = {'t1': f'{stem}_T1w.nii.gz', 'cerebellum': f'{stem}_cerebellum_dseg.nii.gz',
             'brain_mask': f'{stem}_brain_mask.nii.gz', 'logjd': f'{stem}_logjd.nii.gz'}
    res = {'way': way, 'prefix': prefix, 'normalize_json': str(nj), 'path': path, 'n_steps': len(path),
           'start_month': int(path[0][3:-1]) if path else None, 'target': MNI_NAME, 'child': child,
           'edges': [{'edge': f'{a} -> {b}'} for a, b in steps]}
    python_steps = []
    ANTS_COMMANDS = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if way == 'tpl2tpl':
                tx = chain_transforms(path + [MNI_NAME], Path(xfm_root) / 'tpl2tpl')
            elif way in ('avgsubj', 'avgsubjl1o'):
                res['edges'] = [avgsubj_field_without(a, b, child if way == 'avgsubjl1o' else None,
                                                      os.path.join(tmp, f'{a}_to-{b}.nii'), Path(xfm_root) / 'avgsubj')
                                for a, b in steps]
                tx = [e['field'] for e in reversed(res['edges'])]
                python_steps += [f"{e['edge']}: <tmp>/{Path(e['field']).name} = (sum - the {e['left_out']} fields of the "
                                 f"child) / {e['n_visits']}" for e in res['edges'] if e['left_out']]
            else:
                tx = []
            tx += [reg['warp'], reg['affine']]
            field = os.path.join(tmp, 'field.nii')
            _section(f'{prefix}: {way} {path_months(path)} to MNI, {len(tx)} transforms composed')
            run_ants('antsApplyTransforms', '-d', 3, '-r', mni['t1'], '-o', f'[{field},1]', *[x for t in tx for x in ('-t', t)])
            run_ants('antsApplyTransforms', '-d', 3, '-i', norm['head2brain']['t1_brain'], '-r', mni['t1'], '-o',
                     files['t1'], '-n', 'Linear', '-u', 'float', '-t', field)
            run_ants('antsApplyTransforms', '-d', 3, '-i', cereb, '-r', mni['t1'], '-o',
                     files['cerebellum'], '-n', 'NearestNeighbor', '-u', 'uchar', '-t', field)
            run_ants('antsApplyTransforms', '-d', 3, '-i', norm['inputs']['brain_mask'], '-r', mni['t1'], '-o',
                     files['brain_mask'], '-n', 'NearestNeighbor', '-u', 'uchar', '-t', field)
            logjd = os.path.join(tmp, 'logjd.nii')
            run_ants('CreateJacobianDeterminantImage', 3, field, logjd, 1, 0)
            mask = nib.load(mni['mask'])
            jd = np.squeeze(np.asarray(nib.load(logjd).dataobj, dtype=np.float32)) * (np.asarray(mask.dataobj) > 0)
            nib.save(nib.Nifti1Image(jd.astype(np.float32), mask.affine), files['logjd'])
            python_steps.append(f"{files['logjd']} = <tmp>/logjd.nii x ({mni['mask']} > 0), float32")
            hide = lambda s: s.replace(tmp, '<tmp>')
            res.update(transforms=[hide(t) for t in tx], commands=[hide(c) for c in ANTS_COMMANDS],
                       python_steps=python_steps)
            for e in res['edges']:
                if 'field' in e:
                    e['field'] = hide(e['field'])
    finally:
        ANTS_COMMANDS = None
    res.update(inputs={'t1_brain': norm['head2brain']['t1_brain'], 'cerebellum': cereb,
                       'brain_mask': norm['inputs']['brain_mask']}, files=files, metrics=mni_metrics(files, mni_dir),
               software=environment_info())
    Path(f'{stem}.json').write_text(json.dumps(res, indent=2, default=str))
    return res


def submit_visits_to_mni(visits, run_dir, xfm_root, setup='', max_running=50):
    """visit_to_mni of many visits as one Slurm job array (TO_MNI_JOB; array to_mni), one task per visit with all of
    its ways in turn. visits: one dict per visit with normalize_json, child and ways ({way: path}, e.g. {'direct': [],
    'tpl2tpl': [...], 'avgsubj': [...]}); per visit the job script and output <visit folder>/log/<prefix>_to_mni.sh /
    .out. run_dir: the task list to_mni_tasks.tsv (line i = task i), the array script and the Slurm output per task
    (as submit_direct_normalize_array). Returns {'job_id', 'run_dir', 'visits': the dicts with task, script, log}."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, v in enumerate(visits, 1):
        nj = Path(v['normalize_json'])
        prefix = json.loads(nj.read_text())['inputs']['prefix']
        cmds = [['python', '-c', f'import sys; sys.path.insert(0, {str(HERE)!r}); import neonorm; '
                 f'neonorm.SHOW_TOOL_OUTPUT = True; neonorm.visit_to_mni({str(nj)!r}, {way!r}, {list(path)!r}, '
                 f'{v.get("child")!r}, {str(xfm_root)!r})'] for way, path in v['ways'].items()]
        sh = _job_script(f'to_mni_{prefix}', f'{prefix}_to_mni', nj.parent / 'log', cmds, setup, **TO_MNI_JOB)
        rows.append({**v, 'task': i, 'script': str(sh), 'log': str(nj.parent / 'log' / f'{prefix}_to_mni.out')})
    tasks = run_dir / 'to_mni_tasks.tsv'
    tasks.write_text(''.join(f"{r['script']}\t{r['log']}\n" for r in rows))
    sh = run_dir / 'to_mni_array.sh'
    sh.write_text('\n'.join([
        '#!/bin/bash', '#SBATCH --job-name=to_mni', f'#SBATCH --array=1-{len(rows)}%{max_running}',
        f'#SBATCH --cpus-per-task={TO_MNI_JOB["cpus"]}', f'#SBATCH --mem={TO_MNI_JOB["mem"]}',
        f'#SBATCH --time={TO_MNI_JOB["time"]}', f'#SBATCH --output={run_dir}/to_mni_%A_%a.out',
        f'IFS=$\'\\t\' read -r script out < <(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {shlex.quote(str(tasks))})',
        'bash "$script" > "$out" 2>&1']) + '\n')
    return {'job_id': _sbatch(sh), 'run_dir': str(run_dir), 'visits': rows}


# ---------------------------------------------------------------------------
# Evaluation (notebook 5): the reference cerebellum of each visit (brain input; whole-head input: skull_strip.py)
# ---------------------------------------------------------------------------

BRAIN_SUITPY_RUNS = 2  # SUITPy runs per visit (brain_to_cerebellum); one more if they differ (Dice < DICE_MIN)
HEAD_DICE_MIN = 0.80   # brain_to_cerebellum: below this best Dice with the whole-head SUITPy of notebook 2 -> visual check
BRAIN_CEREB_JOB = {'cpus': 4, 'mem': '24G', 'time': '02:00:00'}  # Slurm resources of one visit (T2 N4, 2-3 SUITPy runs)


def brain_cerebellum_files(visit_dir, prefix):
    """Files of brain_to_cerebellum of one visit: 'dseg' (the reference cerebellum, in the visit folder), 'json', 'qc'
    (figure) and the folder of the runs, 'runs' (cerebellum_suitpy/)."""
    d = Path(visit_dir)
    stem = f'{prefix}_cerebellum_suitpy_from-brain-N4'
    return {'dseg': d / f'{stem}_dseg.nii.gz', 'json': d / 'cerebellum_suitpy' / f'{stem}.json',
            'qc': d / 'cerebellum_suitpy' / f'{stem}_qc.png', 'runs': d / 'cerebellum_suitpy'}


def brain_to_cerebellum(normalize_json, t2=None, head_cerebellums=(), qc_figure=True, rotate=None):
    """Reference cerebellum of one visit for the evaluation: one file for every way to MNI (visit_to_mni), chosen
    without any registration. (The cerebellum of the direct QC, <prefix>_direct_normalize_qc.json, is the SUITPy run
    with the higher Dice with the registered template, so it favours the direct registration.)
    As the BCP template cerebellums (bcp_template_cerebellum): SUITPy on the skull-stripped, bias-corrected T1 (t1_brain
    of normalize_json) with the T2 and the brain mask. t2: the rigid T2 on the T1 grid (<prefix>_T2w_reorient_rigid2T1.nii.gz,
    whole head), skull-stripped and bias-corrected here as in head2brain_and_n4correction (<prefix>_T2w_brain_n4.nii.gz,
    used again if it exists); None: the T1 alone (with the brain mask). SUITPy is not deterministic: BRAIN_SUITPY_RUNS
    runs, one more if fewer than two found a cerebellum or they differ (Dice < DICE_MIN); the cerebellum is the medoid
    run (highest mean Dice with the other runs; of runs that agree, the first).
    rotate (a template name of normalize_json, e.g. 'BCP04M'; for a head that lies strongly rotated, where SUITPy
    misplaces the cerebellum): T1, T2 and brain mask are first moved by the rotation start of that registration (its
    init, rotation_search_init: rigid only, no registration to the template) onto the template grid, SUITPy runs there,
    and each run is moved back onto the T1 grid (inverse of init, nearest neighbour), so the result is in the space of
    the visit as without rotate; input names then end with -rot<template>.
    Check without registration: Dice with the whole-head SUITPy cerebellums of notebook 2 (head_cerebellums, e.g.
    skullstrip/<prefix>_cerebellum_suitpy_from-T1wReorientHead_dseg.nii.gz; missing files are skipped; the best Dice
    counts). 'needs_visual_check': fewer than two runs found a cerebellum, the medoid's mean Dice with the other runs is
    below DICE_MIN, or its best Dice with head_cerebellums is below HEAD_DICE_MIN. For information also the Dice with the
    cerebellum of the direct QC.
    Saved (brain_cerebellum_files) in <visit folder>/cerebellum_suitpy/: every run,
    <prefix>_cerebellum_suitpy_from-<input>_run<i>_dseg.nii.gz (existing run files are used again),
    <prefix>_cerebellum_suitpy_from-brain-N4.json and, with qc_figure, <prefix>_cerebellum_suitpy_from-brain-N4_qc.png (medoid
    filled, the best whole-head cerebellum dashed); the medoid as <prefix>_cerebellum_suitpy_from-brain-N4_dseg.nii.gz in
    the visit folder. The same SUITPy call (suitpy_cerebellum) as the whole-head references of the skull strip QC
    (skull_strip.head_to_cerebellum, <prefix>_cerebellum_suitpy_from-T1wReorientHead*_dseg.nii.gz in skullstrip/:
    reoriented T1 (+ T2), no brain mask, one run per input); only the input and the number of runs differ.
    Returns the json content."""
    nj = Path(normalize_json)
    norm = json.loads(nj.read_text())
    prefix, out_dir = norm['inputs']['prefix'], nj.parent
    t1_brain, brain_mask = norm['head2brain']['t1_brain'], str(norm['inputs']['brain_mask'])
    files = brain_cerebellum_files(out_dir, prefix)
    run_dir = files['runs']
    run_dir.mkdir(exist_ok=True)
    stem = f'{prefix}_cerebellum_suitpy'
    t2_brain = None
    if t2 is not None:
        t2_brain = out_dir / f'{prefix}_T2w_brain_n4.nii.gz'
        if not t2_brain.exists():
            ref, img = nib.load(t1_brain), nib.load(t2)
            if img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine, atol=1e-3):
                raise ValueError(f'{t2} is not on the T1 grid: give the rigid T2 (<prefix>_T2w_reorient_rigid2T1.nii.gz)')
            if n_threads():
                os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', str(n_threads()))  # ANTs threads
            with tempfile.TemporaryDirectory() as tmp:
                brain = os.path.join(tmp, 't2_brain.nii.gz')
                _section('T2: skull stripping (ImageMath) and bias correction (N4BiasFieldCorrection)')
                run_ants('ImageMath', 3, brain, 'm', str(t2), brain_mask)
                run_ants('N4BiasFieldCorrection', '-d', 3, '-i', brain, '-x', brain_mask, '-o', str(t2_brain))
    tag = 'T1wT2wbrain' if t2_brain else 'T1wbrain'
    rot_tmp = tempfile.TemporaryDirectory()
    rot = None
    if rotate:
        reg = norm['normalize'][rotate]
        if not reg.get('init'):
            raise ValueError(f'{prefix}: the registration to {rotate} has no rotation start (init)')
        rot = {'template': rotate, 'init': reg['init'], 'grid': reg['tpl_t1']}
        for key, f, interp in (('t1', t1_brain, 'Linear'), ('t2', t2_brain, 'Linear'), ('mask', brain_mask, 'NearestNeighbor')):
            if f:
                rot[key] = os.path.join(rot_tmp.name, f'{key}.nii.gz')
                _section(f'{prefix}: {key} rotated by the start transform of {rotate}')
                run_ants('antsApplyTransforms', '-d', 3, '-i', str(f), '-r', reg['tpl_t1'], '-o', rot[key], '-n', interp,
                         '-t', reg['init'])
        tag += f'-rot{rotate}'
    res = {'prefix': prefix, 'normalize_json': str(nj), 't1_brain': t1_brain, 't2_brain': str(t2_brain) if t2_brain else None,
           'brain_mask': brain_mask, 'input': tag, 'rotate': {k: v for k, v in rot.items() if k in ('template', 'init', 'grid')}
           if rot else None, 'software': environment_info(), 'runs': []}
    found, masks = [], []

    def suitpy_run(i):
        f = run_dir / f'{stem}_from-{tag}_run{i}_dseg.nii.gz'
        _section(f'{prefix}: SUITPy from {tag}, run {i}')
        if f.exists():
            ok = True
        elif rot:  # SUITPy on the rotated images, the result moved back onto the T1 grid
            rotated = os.path.join(rot_tmp.name, f'run{i}_dseg.nii.gz')
            ok = suitpy_cerebellum(rot['t1'], rotated, t2=rot.get('t2'), brain_mask=rot['mask']) is not None
            if ok:
                run_ants('antsApplyTransforms', '-d', 3, '-i', rotated, '-r', t1_brain, '-o', str(f), '-n', 'NearestNeighbor',
                         '-u', 'uchar', '-t', f"[{rot['init']},1]")
        else:
            ok = suitpy_cerebellum(t1_brain, f, t2=t2_brain, brain_mask=brain_mask) is not None
        res['runs'].append({'run': i, 'dseg': str(f) if ok else None})
        if ok:
            found.append(f)
            masks.append(_load_bool(f))

    for i in range(1, BRAIN_SUITPY_RUNS + 1):
        suitpy_run(i)
    if len(masks) < 2 or min(_dice(a, b) for j, a in enumerate(masks) for b in masks[j + 1:]) < DICE_MIN:
        suitpy_run(BRAIN_SUITPY_RUNS + 1)
    rot_tmp.cleanup()
    dice = np.array([[_dice(a, b) for b in masks] for a in masks])
    mean = [(dice[i].sum() - 1) / (len(masks) - 1) if len(masks) > 1 else np.nan for i in range(len(masks))]
    k = int(np.nanargmax(mean)) if len(masks) > 1 else 0
    res.update(n_runs=len(res['runs']), n_found=len(found), runs_dice=dice.round(4).tolist(),
               medoid=str(found[k]) if found else None, medoid_mean_dice=float(mean[k]) if found else np.nan,
               head_dice={}, head_best=None, head_dice_best=np.nan, dice_with_direct_qc=np.nan, cerebellum=None)
    if found:
        res['cerebellum'] = str(files['dseg'])
        shutil.copy(found[k], res['cerebellum'])
        heads = {str(f): _load_bool(f) for f in head_cerebellums if Path(f).exists()}
        res['head_dice'] = {f: float(_dice(masks[k], h)) for f, h in heads.items() if h.shape == masks[k].shape}
        if res['head_dice']:
            res['head_best'] = max(res['head_dice'], key=res['head_dice'].get)
            res['head_dice_best'] = res['head_dice'][res['head_best']]
        qc_json = nj.with_name(nj.name.replace('_direct_normalize.json', '_direct_normalize_qc.json'))
        direct = json.loads(qc_json.read_text()).get('t1_cerebellum') if qc_json.exists() else None
        if direct and Path(direct).exists():
            res['dice_with_direct_qc'] = float(_dice(masks[k], _load_bool(direct)))
    res['needs_visual_check'] = (len(found) < 2 or not res['medoid_mean_dice'] >= DICE_MIN
                                 or not res['head_dice_best'] >= HEAD_DICE_MIN)
    if qc_figure and found:
        text = (f'{prefix} | SUITPy from {tag}: {len(found)}/{len(res["runs"])} runs found a cerebellum | medoid run '
                f'{found[k].name.split("_run")[1][0]}, mean Dice with the other runs {res["medoid_mean_dice"]:.3f} | '
                f'best Dice with the whole-head SUITPy {res["head_dice_best"]:.3f}')
        res['qc_figure'] = template_cerebellum_figure(t1_brain, masks[k], heads.get(res['head_best']),
                                                      files['qc'], text,
                                                      old_label='whole-head SUITPy of notebook 2 (best Dice)')
    files['json'].write_text(json.dumps(res, indent=2, default=str))
    return res


def submit_brain_to_cerebellum(visits, run_dir, setup='', max_running=50):
    """brain_to_cerebellum of many visits as one Slurm job array (BRAIN_CEREB_JOB; array bcereb), one task per visit (as
    submit_visits_to_mni). visits: one dict per visit with normalize_json, t2 (or None), head_cerebellums and
    optionally rotate (brain_to_cerebellum); per visit
    the job script and output <visit folder>/log/<prefix>_cerebellum_suitpy_from-brain-N4.sh / .out. run_dir: the task list
    bcereb_tasks.tsv (line i = task i), the array script and the Slurm output per task. Returns {'job_id', 'run_dir',
    'visits': the dicts with task, script, log}."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, v in enumerate(visits, 1):
        nj = Path(v['normalize_json'])
        prefix = json.loads(nj.read_text())['inputs']['prefix']
        t2 = str(v['t2']) if v.get('t2') else None
        code = (f'import sys; sys.path.insert(0, {str(HERE)!r}); import neonorm; neonorm.SHOW_TOOL_OUTPUT = True; '
                f'neonorm.brain_to_cerebellum({str(nj)!r}, {t2!r}, {[str(f) for f in v.get("head_cerebellums", [])]!r}, '
                f'rotate={v.get("rotate")!r})')
        sh = _job_script(f'bcereb_{prefix}', f'{prefix}_cerebellum_suitpy_from-brain-N4', nj.parent / 'log',
                         [['python', '-c', code]], setup, **BRAIN_CEREB_JOB)
        rows.append({**v, 'task': i, 'script': str(sh),
                     'log': str(nj.parent / 'log' / f'{prefix}_cerebellum_suitpy_from-brain-N4.out')})
    tasks = run_dir / 'bcereb_tasks.tsv'
    tasks.write_text(''.join(f"{r['script']}\t{r['log']}\n" for r in rows))
    sh = run_dir / 'bcereb_array.sh'
    sh.write_text('\n'.join([
        '#!/bin/bash', '#SBATCH --job-name=bcereb', f'#SBATCH --array=1-{len(rows)}%{max_running}',
        f'#SBATCH --cpus-per-task={BRAIN_CEREB_JOB["cpus"]}', f'#SBATCH --mem={BRAIN_CEREB_JOB["mem"]}',
        f'#SBATCH --time={BRAIN_CEREB_JOB["time"]}', f'#SBATCH --output={run_dir}/bcereb_%A_%a.out',
        f'IFS=$\'\\t\' read -r script out < <(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {shlex.quote(str(tasks))})',
        'bash "$script" > "$out" 2>&1']) + '\n')
    return {'job_id': _sbatch(sh), 'run_dir': str(run_dir), 'visits': rows}


FOLD_LOGJD = float(np.log(0.001)) + 1e-4  # CreateJacobianDeterminantImage sets J below 0.001 to 0.001 before the log:
                                          # logJD at this floor = folded (J <= 0) or collapsed voxel


def mni_metrics(files, mni_dir=None):
    """Evaluation values of one visit at MNI from the files of visit_to_mni ('t1', 'brain_mask', 'cerebellum',
    'logjd'), as registration_qc: correlation of the T1 with the MNI T1 inside the MNI brain mask (t1_corr); Dice, HD95
    (mm) and log10(HD95) of the brain mask with the MNI brain mask and of the cerebellum with the MNI cerebellum
    (<...>_res-1_cerebellum_mask; NaN if a file is missing); mean and minimum of the log Jacobian determinant inside the
    MNI brain mask (logjd_mean, logjd_min; finite values) and the share of folded voxels (logjd_fold_pct, % of the MNI
    brain mask at the floor FOLD_LOGJD). Returns a dict."""
    mni = mni_template(mni_dir)
    img = lambda f: np.squeeze(np.asanyarray(nib.load(f).dataobj))
    tm = img(mni['mask']) > 0
    spacing = nib.load(mni['mask']).header.get_zooms()[:3]
    res = {'t1_corr': float(np.corrcoef(img(files['t1']).astype(np.float32)[tm], img(mni['t1']).astype(np.float32)[tm])[0, 1])}
    for name, f, ref in (('brain', files.get('brain_mask'), mni['mask']), ('cerebellum', files.get('cerebellum'), mni['cerebellum'])):
        res.update({f'{name}_dice': np.nan, f'{name}_hd95': np.nan, f'{name}_log10_hd95': np.nan})
        if f and ref and Path(f).exists():
            m, r = img(f) > 0, img(ref) > 0
            res[f'{name}_dice'] = float(_dice(m, r))
            res[f'{name}_hd95'] = _hd95(m, r, spacing)
            with np.errstate(divide='ignore'):  # HD95 0 (less than one voxel): -inf, as registration_qc
                res[f'{name}_log10_hd95'] = float(np.log10(res[f'{name}_hd95']))
    jd = img(files['logjd']).astype(np.float32)[tm]
    jd = jd[np.isfinite(jd)]
    res.update(logjd_mean=float(jd.mean()), logjd_min=float(jd.min()), logjd_fold_pct=float(100 * (jd <= FOLD_LOGJD).mean()))
    return res


def logjd_fold_table(files, out_tsv, mni_dir=None):
    """logjd_fold_pct of mni_metrics for logJD maps at MNI (`files`, e.g. the _logjd.nii.gz of visit_to_mni written
    before mni_metrics had it): out_tsv, one row per file (file_logjd, logjd_fold_pct). Returns out_tsv."""
    tm = np.squeeze(np.asanyarray(nib.load(mni_template(mni_dir)['mask']).dataobj)) > 0
    rows = []
    for f in files:
        jd = np.squeeze(np.asarray(nib.load(f).dataobj, dtype=np.float32))[tm]
        rows.append({'file_logjd': f, 'logjd_fold_pct': float(100 * (jd[np.isfinite(jd)] <= FOLD_LOGJD).mean())})
    pd.DataFrame(rows).to_csv(out_tsv, sep='\t', index=False)
    return str(out_tsv)


def adult_mni_metrics(adults, out_tsv, mni_dir=None):
    """mni_metrics of adults registered directly to MNI elsewhere (the adult reference of the evaluation, e.g. HCP
    S1200, as the earlier evaluation): per adult (dict with participant_id; t1: its T1 at MNI; brain_mask and cerebellum
    on its own grid; transforms: its ANTs transforms to MNI, warp first) the transforms composed into one field on the
    MNI grid, the brain mask and cerebellum moved with it (nearest neighbour) and its log Jacobian, as visit_to_mni. Only
    the values are kept: out_tsv, one row per adult (participant_id + mni_metrics), written after every adult. Returns
    out_tsv."""
    mni = mni_template(mni_dir)
    rows = []
    for a in adults:
        with tempfile.TemporaryDirectory() as tmp:
            field = os.path.join(tmp, 'field.nii')
            files = {'t1': a['t1'], **{k: os.path.join(tmp, f'{k}.nii') for k in ('brain_mask', 'cerebellum', 'logjd')}}
            run_ants('antsApplyTransforms', '-d', 3, '-r', mni['t1'], '-o', f'[{field},1]',
                     *[x for t in a['transforms'] for x in ('-t', t)])
            for k in ('brain_mask', 'cerebellum'):
                run_ants('antsApplyTransforms', '-d', 3, '-i', a[k], '-r', mni['t1'], '-o', files[k], '-n',
                         'NearestNeighbor', '-u', 'uchar', '-t', field)
            run_ants('CreateJacobianDeterminantImage', 3, field, files['logjd'], 1, 0)
            rows.append({'participant_id': a['participant_id'], **mni_metrics(files, mni_dir)})
        pd.DataFrame(rows).to_csv(out_tsv, sep='\t', index=False)
    return str(out_tsv)


FOLD_JOB = {'cpus': 1, 'mem': '8G', 'time': '04:00:00'}   # Slurm resources of logjd_fold_table (a few thousand maps)
ADULT_JOB = {'cpus': 2, 'mem': '16G', 'time': '04:00:00'}  # Slurm resources of one chunk of adult_mni_metrics (~1 min per adult)


def _submit_python(job, name, log_dir, func, kwargs, resources, setup=''):
    """neonorm.<func>(**kwargs) as one Slurm job: kwargs in <log_dir>/<name>.json, script and output
    <log_dir>/<name>.sh / .out. Returns the job id."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    args = log_dir / f'{name}.json'
    args.write_text(json.dumps(kwargs, indent=1, default=str))
    code = (f'import sys, json; sys.path.insert(0, {str(HERE)!r}); import neonorm; '
            f'neonorm.{func}(**json.load(open({str(args)!r})))')
    return _sbatch(_job_script(job, name, log_dir, [['python', '-c', code]], setup, **resources))


def submit_logjd_fold_table(files, out_tsv, setup=''):
    """logjd_fold_table as one Slurm job (FOLD_JOB; job logjd_fold; log in <folder of out_tsv>/logs). Returns the job id."""
    out = Path(out_tsv)
    return _submit_python('logjd_fold', out.stem, out.parent / 'logs', 'logjd_fold_table',
                          {'files': [str(f) for f in files], 'out_tsv': str(out)}, FOLD_JOB, setup)


def submit_adult_mni_metrics(adults, out_dir, setup='', n_chunks=20):
    """adult_mni_metrics in n_chunks Slurm jobs (ADULT_JOB; jobs adult_<k>): <out_dir>/adult_mni_metrics_chunk<k>.tsv,
    logs in <out_dir>/logs. Returns one dict per chunk: chunk, n, tsv, job_id."""
    jobs = []
    for k in range(n_chunks):
        part = list(adults)[k::n_chunks]
        if part:
            tsv = Path(out_dir) / f'adult_mni_metrics_chunk{k:02d}.tsv'
            jobs.append({'chunk': k, 'n': len(part), 'tsv': str(tsv),
                         'job_id': _submit_python(f'adult_{k}', tsv.stem, Path(out_dir) / 'logs', 'adult_mni_metrics',
                                                  {'adults': part, 'out_tsv': str(tsv)}, ADULT_JOB, setup)})
    return jobs


WAY_COLORS = {'direct': 'gray', 'tpl2tpl': '#1B2A7B', 'avgsubj': '#8B1A1A', 'avgsubjl1o': '#CD5C5C'}  # earlier notebooks
EVAL_METRICS = {  # mni_metrics value -> axis label, sign (1: higher is better, -1: lower is better, 0: neither), y limits
    'brain_dice': ('Brain Dice', 1, None),
    'brain_log10_hd95': ('Brain HD95 (log10 mm)', -1, None),
    'cerebellum_dice': ('Cerebellum Dice', 1, None),
    'cerebellum_log10_hd95': ('Cerebellum HD95 (log10 mm)', -1, None),
    't1_corr': ('T1 r with MNI', 1, (0, 1)),
    'logjd_mean': ('Mean logJD', 0, None),
    'logjd_fold_pct': ('Folded voxels (%)', -1, None),
}
OLDER_AGE_EDGES = (96, 132, 168, 216)  # months: age groups of the datasets other than BCP (HCPD: 8-11, 11-14, 14-18 y)


def eval_groups(dataset, age_months):
    """Age group of each visit for plot_by_month: BCP the month of its nearest BCP template (as bcp_template; the start
    of its indirect path), e.g. '3M'; other datasets '<dataset> 8-11y' (OLDER_AGE_EDGES). Returns the labels and the
    groups in plotting order (BCP months, then the other datasets)."""
    e = OLDER_AGE_EDGES
    labels = []
    for d, a in zip(dataset, age_months):
        if d == 'BCP':
            labels.append(f'{min(BCP_MONTHS, key=lambda m: (abs(m - a), -m))}M')
        else:
            i = int(np.searchsorted(e, a, side='right')) - 1
            labels.append(f'{d} {e[i] // 12}-{e[i + 1] // 12}y' if 0 <= i < len(e) - 1 else f'{d} other')
    others = sorted({(d, l) for d, l in zip(dataset, labels) if d != 'BCP'},
                    key=lambda dl: (dl[0], dl[1].endswith('other'), int(dl[1].split()[-1].split('-')[0])
                                    if not dl[1].endswith('other') else 0))
    order = [f'{m}M' for m in BCP_MONTHS if f'{m}M' in labels] + [l for _, l in others]
    return labels, order


def plot_by_month(df, metrics=tuple(EVAL_METRICS), ways=('direct', 'tpl2tpl', 'avgsubj'), adult_ref=None,
                  significance=True, alpha=0.05, adult_label='adult reference', out_png=None):
    """Values at MNI per age group (eval_groups), as the figures of the earlier evaluation
    (2.2nd_postnatal_step_in.ipynb, plot_metrics_grid): one panel per metric, stacked; per way the mean +/- SD of each
    group, connected within a dataset (BCP template months; the other datasets apart, after a dotted line), the ways
    side by side; x labels with the number of visits.
    df: one row per visit x way (dataset, participant_id, age_months, way and the metric columns; 5.2_mni_metrics.tsv).
    adult_ref: None, or one row per adult with the metric columns (adult_mni_metrics): grey band mean +/- SD, dotted
    line at the mean, over the whole x axis.
    significance: per group and pair of ways a paired Wilcoxon signed-rank test (visits with both values), FDR
    (Benjamini-Hochberg) over the groups and pairs of a metric; above the data a grey bracket between the two ways of
    each pair with q < alpha (stars: * q < 0.05, ** < 0.01, *** < 0.001; non-significant pairs are not drawn), the
    pairs of a metric on fixed levels, one per pair (the shorter ones lower). The better way of each pair (EVAL_METRICS sign) is in
    the tests table.
    Returns the figure, the summary (metric, way, group, n, mean, sd) and the tests (metric, group, pair, n, p, q,
    better)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    ways, metrics = list(ways), list(metrics)
    d = df[df['way'].isin(ways)].copy()
    d['group'], groups = eval_groups(d['dataset'], d['age_months'])
    d[metrics] = d[metrics].replace([np.inf, -np.inf], np.nan)
    long = d.melt(id_vars=['way', 'group'], value_vars=metrics, var_name='metric').dropna(subset=['value'])
    summary = long.groupby(['metric', 'way', 'group'])['value'].agg(n='count', mean='mean', sd='std').reset_index()
    pairs = [(a, b) for i, a in enumerate(ways) for b in ways[i + 1:]]
    tests = []
    for m in metrics if significance else []:
        wide = d.pivot_table(index=['dataset', 'participant_id', 'group'], columns='way', values=m).reset_index()
        rows = []
        for g in groups:
            x = wide[wide['group'] == g]
            for a, b in pairs:
                v = x[[a, b]].dropna() if {a, b} <= set(x) else pd.DataFrame(columns=[a, b])
                diff = v[a] - v[b]
                p = wilcoxon(v[a], v[b]).pvalue if (diff != 0).sum() > 0 else np.nan
                sign = EVAL_METRICS.get(m, (m, 0, None))[1]
                rows.append({'metric': m, 'group': g, 'pair': f'{a} vs {b}', 'n': len(v), 'p': p,
                             'better': (a if diff.mean() * sign > 0 else b) if sign and len(v) else None})
        r = pd.DataFrame(rows)
        r['q'] = np.nan
        ok = r['p'].notna()
        if ok.any():
            r.loc[ok, 'q'] = multipletests(r.loc[ok, 'p'], method='fdr_bh')[1]
        tests.append(r)
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame(
        columns=['metric', 'group', 'pair', 'n', 'p', 'q', 'better'])

    x_of = {g: i for i, g in enumerate(groups)}
    dataset_of = d.groupby('group')['dataset'].first()
    segments = [[g for g in groups if dataset_of[g] == ds] for ds in dict.fromkeys(dataset_of[groups])]
    n_of = d.groupby('group')['participant_id'].nunique()
    off = {w: (k - (len(ways) - 1) / 2) * 0.2 for k, w in enumerate(ways)}
    span = {(a, b): abs(ways.index(a) - ways.index(b)) for a, b in pairs}
    level = {q: i for i, q in enumerate(sorted(pairs, key=lambda q: (span[q], ways.index(q[0]))))}  # one per pair, short low
    n_levels = len(pairs)
    stars = lambda q: '***' if q < 0.001 else '**' if q < 0.01 else '*'
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(10, 0.6 * len(groups)), 3.4 * len(metrics)), squeeze=False)
    for ax, m in zip(axes[:, 0], metrics):
        label, sign, ylim = EVAL_METRICS.get(m, (m, 0, None))
        s = summary[summary['metric'] == m]
        for w in ways:
            sw = s[s['way'] == w].set_index('group')
            for seg in segments:
                p = sw.reindex([g for g in seg if g in sw.index])
                x = [x_of[g] + off[w] for g in p.index]
                ax.plot(x, p['mean'], color=WAY_COLORS.get(w), lw=2, marker='o', ms=4)
                ax.errorbar(x, p['mean'], yerr=p['sd'], fmt='none', color=WAY_COLORS.get(w), capsize=3, capthick=1,
                            elinewidth=0.8)
        if adult_ref is not None and m in adult_ref:
            v = adult_ref[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v):
                ax.axhspan(v.mean() - v.std(), v.mean() + v.std(), color='gray', alpha=0.2, lw=0, zorder=0)
                ax.axhline(v.mean(), color='gray', ls=':', lw=1, zorder=0)
        for seg in segments[1:]:
            ax.axvline(x_of[seg[0]] - 0.5, color='#8a8984', ls=':', lw=1)
        if ylim:
            ax.set_ylim(*ylim)
        if significance:
            lo, hi = ax.get_ylim()
            step = 0.08 * (hi - lo)
            t = tests[(tests['metric'] == m) & (tests['q'] < alpha)]
            for g, pair, q in zip(t['group'], t['pair'], t['q']):
                a, b = pair.split(' vs ')
                x1, x2 = sorted((x_of[g] + off[a], x_of[g] + off[b]))
                y = hi + (level[(a, b)] + 0.3) * step
                ax.plot([x1, x1, x2, x2], [y, y + 0.2 * step, y + 0.2 * step, y], color='gray', lw=0.8, clip_on=False)
                ax.text((x1 + x2) / 2, y + 0.15 * step, stars(q), ha='center', va='bottom', fontsize=6, color='gray')
            ax.set_yticks([v for v in ax.get_yticks() if lo - 1e-9 <= v <= hi + 1e-9])  # no ticks at the brackets
            ax.set_ylim(lo, hi + (n_levels + 0.3) * step)
            ax.spines['left'].set_bounds(lo, hi)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks(range(len(groups)), [f"{g.replace(' ', chr(10))}\nn={n_of[g]}" for g in groups], fontsize=7)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.2, axis='x')
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1, 0].set_xlabel('BCP: age-specific template (nearest to the age, months) | other datasets: age group',
                           fontsize=8)
    handles = [Line2D([0], [0], color=WAY_COLORS.get(w), lw=2, marker='o', ms=4, label=f'{w} (mean +/- SD)')
               for w in ways]
    if adult_ref is not None:
        handles.append(Patch(color='gray', alpha=0.2, label=f'{adult_label} (mean +/- SD, n={len(adult_ref)})'))
    if significance:
        handles.append(Line2D([0], [0], color='gray', lw=0.8,
                              label='paired Wilcoxon, FDR: * q < 0.05, ** < 0.01, *** < 0.001'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / fig.get_figheight()))
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig, summary, tests


LONGI_JOB = {'cpus': 4, 'mem': '32G', 'time': '04:00:00'}  # Slurm resources of one longitudinal_r (a few hundred maps)


def longitudinal_r(early, late, out_tsv, mni_dir=None):
    """Pearson r between every map of `early` and every map of `late` (lists of {'key', 'file'}, e.g. the logJD at MNI
    of visit_to_mni), inside the MNI brain mask (as the earlier longitudinal evaluation, 4.eval_longitudinal.ipynb):
    out_tsv, one row per pair (early, late, r); the notebook takes the pairs of the same child (longitudinal
    reliability) and of different children (null). Missing files are skipped. Returns out_tsv."""
    mni = mni_template(mni_dir)
    tm = np.squeeze(np.asanyarray(nib.load(mni['mask']).dataobj)) > 0

    def standardised(items):
        items = [it for it in items if Path(it['file']).exists()]
        x = np.empty((len(items), int(tm.sum())), dtype=np.float32)
        for i, it in enumerate(items):
            v = np.squeeze(np.asarray(nib.load(it['file']).dataobj, dtype=np.float32))[tm]
            v -= v.mean()
            x[i] = v / np.linalg.norm(v)  # then r = dot product
        return [it['key'] for it in items], x

    ek, e = standardised(early)
    lk, l = standardised(late)
    r = e @ l.T
    pd.DataFrame({'early': np.repeat(ek, len(lk)), 'late': np.tile(lk, len(ek)), 'r': r.ravel()}).to_csv(
        out_tsv, sep='\t', index=False)
    return str(out_tsv)


def submit_longitudinal_r(tasks, out_dir, setup=''):
    """longitudinal_r of each task (a dict with name, early, late) as one Slurm job (LONGI_JOB; job longi_<name>): the
    maps in <out_dir>/<name>_maps.json, the result <out_dir>/<name>_r.tsv, script and output in
    <out_dir>/log/<name>.sh / .out. Returns one dict per task: name, r_tsv, log, job_id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t in tasks:
        maps = out_dir / f"{t['name']}_maps.json"
        maps.write_text(json.dumps({'early': t['early'], 'late': t['late']}, indent=1, default=str))
        r_tsv = out_dir / f"{t['name']}_r.tsv"
        code = (f'import sys, json; sys.path.insert(0, {str(HERE)!r}); import neonorm; '
                f'm = json.load(open({str(maps)!r})); neonorm.longitudinal_r(m["early"], m["late"], {str(r_tsv)!r})')
        sh = _job_script(f"longi_{t['name']}", t['name'], out_dir / 'log', [['python', '-c', code]], setup, **LONGI_JOB)
        jobs.append({'name': t['name'], 'r_tsv': str(r_tsv), 'log': str(out_dir / 'log' / f"{t['name']}.out"),
                     'job_id': _sbatch(sh)})
    return jobs


def main():
    p = argparse.ArgumentParser(description='neonorm for one visit: step 1 (preprocessing: orientation, skull '
                                            'stripping + QC), or step 2 (direct normalisation) with --direct_normalize')
    p.add_argument('--check', action='store_true', help='check the software setup and exit')
    p.add_argument('t1', nargs='?')
    p.add_argument('age_months', nargs='?', type=float, help='age at scan in months')
    p.add_argument('out_dir', nargs='?')
    p.add_argument('--t2', help='T2 (with --direct_normalize, optional: the T2 on the T1 grid, '
                                '<prefix>_T2w_rigid2T1.nii.gz of step 1; skull-stripped and bias corrected, not used for '
                                'the registration)')
    p.add_argument('--prefix', help='file name prefix (default: the T1 file name without _T1w.nii.gz)')
    p.add_argument('--save_brain', action='store_true', help='also save the skull-stripped T1 (and T2 on the T1 grid)')
    p.add_argument('--t1_brain', help='save the skull-stripped T1 to this path')
    p.add_argument('--qc_figure', action='store_true', help='write the QC figure also when nothing is flagged')
    p.add_argument('--direct_normalize', action='store_true',
                   help='step 2 instead of step 1: t1 = the T1 of step 1 (orientation), needs --brain_mask; out_dir '
                        'optional (default: the folder of t1)')
    p.add_argument('--brain_mask', help='step 2: brain mask of step 1')
    p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3], choices=[1, 2, 3],
                   help='with --direct_normalize, the parts of step 2: 1 = 2a skull stripping + N4 (T1, T2), '
                        '2 = 2b registrations, 3 = 2c QC (default: all)')
    p.add_argument('--templates', nargs='+',
                   help='with --direct_normalize, 2b: template names (e.g. BCP05M MNI152NLin2009cSym; default: the '
                        'BCP template of the nearest atlas month and MNI); templates registered before are kept')
    p.add_argument('--no_rotation_init', dest='rotation_init', action='store_false',
                   help='with --direct_normalize, 2b: the centre-of-mass start of antsRegistrationSyN.sh instead of '
                        'the default, a rotation search (antsAI) against the BCP template of the age')
    p.add_argument('--rotation_init', action='store_true', help=argparse.SUPPRESS)  # the default; older job scripts
    p.set_defaults(rotation_init=True)
    p.add_argument('--bcp_templates', help='step 2: UNC BCP 4D atlas folder (sets NEONORM_BCP_TEMPLATES)')
    p.add_argument('--mni_template', help='step 2: templateflow folder tpl-MNI152NLin2009cSym (sets NEONORM_MNI_TEMPLATE)')
    p.add_argument('--synthstrip_models', help='folder with the SynthStrip models (sets NEONORM_SYNTHSTRIP_MODELS)')
    p.add_argument('--threads', type=int, help='CPU threads (sets NEONORM_NTHREADS)')
    p.add_argument('--antspath', help='ANTs bin folder with antsRegistrationSyN.sh (sets ANTSPATH; step 2)')
    a = p.parse_args()
    global SHOW_TOOL_OUTPUT
    SHOW_TOOL_OUTPUT = True  # command line (e.g. a Slurm job): the full output of every tool goes to the log
    if a.antspath:
        os.environ['ANTSPATH'] = a.antspath
    if a.synthstrip_models:
        os.environ['NEONORM_SYNTHSTRIP_MODELS'] = a.synthstrip_models
    if a.threads:
        os.environ['NEONORM_NTHREADS'] = str(a.threads)
    if a.bcp_templates:
        os.environ['NEONORM_BCP_TEMPLATES'] = a.bcp_templates
    if a.mni_template:
        os.environ['NEONORM_MNI_TEMPLATE'] = a.mni_template
    if a.check:
        problems = check_environment(verbose=True)
        print('\nready' if not problems else '\nnot ready:\n  - ' + '\n  - '.join(problems))
        sys.exit(1 if problems else 0)
    if a.direct_normalize:
        if not (a.t1 and a.age_months is not None and a.brain_mask):
            p.error('--direct_normalize needs t1, age_months and --brain_mask')
        out = direct_normalize(a.t1, a.age_months, a.brain_mask, a.out_dir, t2=a.t2, prefix=a.prefix, steps=a.steps,
                               rotation_init=a.rotation_init, templates=a.templates)
        _section(f'done: --steps {a.steps}')
        summary = {'head2brain': out.get('head2brain') if 1 in a.steps else None,
                   'normalize': {name: {k: r[k] for k in ('affine', 'warp', 'inverse_warp', 'warped')}
                                 for name, r in out.get('normalize', {}).items()} if 2 in a.steps else None}
        if 3 in a.steps:
            q = out['qc']
            summary['qc'] = {'cerebellum': q['t1_cerebellum'], 'needs_visual_check': q['needs_visual_check'],
                             'qc': {name: {k: v for k, v in x.items() if k != 'qc_figure'} for name, x in q['qc'].items()}}
        print(json.dumps({k: v for k, v in summary.items() if v is not None}, indent=2, default=str))
        return
    if not (a.t1 and a.age_months is not None and a.out_dir):
        p.error('t1, age_months and out_dir are required (or use --check)')
    out = preprocess(a.t1, a.age_months, a.out_dir, a.t2, a.prefix, a.t1_brain, a.save_brain, a.qc_figure)
    print(json.dumps(out['skull_strip'].get('qc', {}), indent=2, default=str))


if __name__ == '__main__':
    main()
