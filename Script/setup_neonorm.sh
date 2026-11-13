#!/bin/bash
# One-time setup of neonorm on a server (Linux x86-64) or a Mac (arm64 or x86-64) with the versions validated on the BCP
# data (README.md):
#   bash Script/setup_neonorm.sh <software folder> [<templates source>]
# <software folder>: where ANTs, the SynthStrip models and neonorm_env.sh go (e.g. /project/<id>/toolbox/neonorm).
# <templates source>: the templates/ folder of a computer that has them (a path or host:path, rsync); it is copied to
#   <repository>/templates. Without it the templates must already be there.
# Steps, each skipped if already done:
#   1. conda environment `neonorm` (environment.yml)
#   2. ANTs release binaries (ANTS_VERSION, ANTS_PLATFORM)
#   3. SynthStrip: the default model from FreeSurfer, the infant model from the repository; without FreeSurfer
#      (FREESURFER_HOME not set, e.g. on a Mac) the mri_synthstrip script of FreeSurfer 8.0.0 in a conda environment
#      of its own (`neonorm-synthstrip`: PyTorch and surfa as in FreeSurfer 8.0.0) and the default model from the
#      SynthStrip website
#   4. templates, checked against Script/data/templates_md5.txt (BCP and MNI templates, their cerebellums, the
#      tpl2tpl and avgsubj edges); without the edges (tpl_xfm_build/, 16 GB, not in git) steps 1-2 only
#   5. <software folder>/neonorm_env.sh: the settings of this server; `source` it in the shell, in ~/.bashrc and in
#      every job script (the `setup` of the neonorm.submit_* functions)
#   6. neonorm --check and the versions against the validated ones
# FSL 6.0.6.2 and FreeSurfer 8.0.0 are not installed by this script (licences, size): set FSLDIR (and FREESURFER_HOME,
# if FreeSurfer is there) before running it (e.g. `module load fsl freesurfer`, or the folders under /opt). neonorm uses
# only fslreorient2std and mri_synthstrip from them.
set -eo pipefail  # no -u: conda's activation scripts use unset variables
export PYTHONNOUSERSITE=1  # pip installs every package into the environment, none counts as there from ~/.local
SOFT=${1:?usage: bash Script/setup_neonorm.sh <software folder> [<templates source>]}
TEMPLATES_SRC=${2:-}
REPO=$(cd "$(dirname "$0")/.." && pwd)
ANTS_VERSION=${ANTS_VERSION:-2.6.2}
case "$(uname -s)-$(uname -m)" in  # release file ants-<version>-<platform>.zip (e.g. ubuntu-22.04-X64-gcc)
    Darwin-arm64) ANTS_PLATFORM=${ANTS_PLATFORM:-macos-14-ARM64-clang} ;;
    Darwin-*) ANTS_PLATFORM=${ANTS_PLATFORM:-macos-13-X64-clang} ;;
    *) ANTS_PLATFORM=${ANTS_PLATFORM:-almalinux8-X64-gcc} ;;
esac
: "${FSLDIR:?set FSLDIR (FSL 6.0.6.2) first}"

md5_check() {  # md5sum -c of the lines '<md5>  <file>' on stdin, also on macOS (no md5sum there: md5)
    if command -v md5sum > /dev/null; then
        md5sum --quiet -c -
    else
        local sum file bad=0
        while read -r sum file; do
            [ "$(md5 -q "$file" 2> /dev/null)" = "$sum" ] || { echo "$file: FAILED"; bad=1; }
        done
        return $bad
    fi
}
mkdir -p "$SOFT"
SOFT=$(cd "$SOFT" && pwd)

echo "== 1. conda environment neonorm"
command -v conda > /dev/null || { echo 'conda not found: install Miniforge (or load the conda module) first'; exit 1; }
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda env list | grep -qE '^neonorm\s'; then  # exists: brought to environment.yml (also after a failed creation)
    conda env update -n neonorm -f "$REPO/environment.yml" --prune
else
    conda env create -f "$REPO/environment.yml"
fi

echo "== 2. ANTs $ANTS_VERSION"
ANTSPATH=$SOFT/ants-$ANTS_VERSION/bin
if [ ! -x "$ANTSPATH/antsRegistrationSyN.sh" ]; then
    zip=ants-$ANTS_VERSION-$ANTS_PLATFORM.zip
    curl -fL -o "$SOFT/$zip" "https://github.com/ANTsX/ANTs/releases/download/v$ANTS_VERSION/$zip"
    unzip -q "$SOFT/$zip" -d "$SOFT" && rm "$SOFT/$zip"
fi
"$ANTSPATH/antsRegistration" --version | head -1

echo "== 3. SynthStrip"
MODELS=$SOFT/synthstrip_models
mkdir -p "$MODELS"
if [ -n "$FREESURFER_HOME" ]; then
    [ -f "$MODELS/synthstrip.1.pt" ] || cp "$FREESURFER_HOME/models/synthstrip.1.pt" "$MODELS/"
    FS_ENV="export FREESURFER_HOME=$FREESURFER_HOME"
else  # no FreeSurfer: its mri_synthstrip script (8.0.0) with the packages of FreeSurfer 8.0.0 in an environment of its own
    BIN=$SOFT/synthstrip/bin
    if [ ! -x "$BIN/mri_synthstrip" ]; then
        conda env list | grep -qE '^neonorm-synthstrip\s' || conda create -y -n neonorm-synthstrip -c conda-forge python=3.8 pip
        torch_index=$([ "$(uname -s)" = Linux ] && echo '--index-url https://download.pytorch.org/whl/cpu' || true)
        conda run -n neonorm-synthstrip pip install $torch_index torch==2.1.2
        conda run -n neonorm-synthstrip pip install numpy==1.24.3 scipy==1.10.1 nibabel==5.2.1 surfa==0.6.1
        mkdir -p "$BIN"
        curl -fL -o "$SOFT/synthstrip/mri_synthstrip.py" \
            https://raw.githubusercontent.com/freesurfer/freesurfer/v8.0.0/mri_synthstrip/mri_synthstrip
        py=$(conda run -n neonorm-synthstrip python -c 'import sys; print(sys.executable)')
        printf '#!/bin/bash\nexec %s %s "$@"\n' "$py" "$SOFT/synthstrip/mri_synthstrip.py" > "$BIN/mri_synthstrip"
        chmod +x "$BIN/mri_synthstrip"
    fi
    [ -f "$MODELS/synthstrip.1.pt" ] || curl -fL -o "$MODELS/synthstrip.1.pt" \
        https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt
    FS_ENV="export PATH=$BIN:\$PATH  # mri_synthstrip of FreeSurfer 8.0.0 without FreeSurfer"
fi
[ -f "$MODELS/synthstrip.infant.1.pt" ] || cp "$REPO/Script/skullstrip/synthstrip/model/synthstrip.infant.1.pt" "$MODELS/"
(cd "$MODELS" && md5_check <<'EOF'
27f86a16f396ac6db3023c2f70b97ae1  synthstrip.1.pt
37a6e0fa34adec4ab4ffba22de551595  synthstrip.infant.1.pt
EOF
) && echo 'both models as validated'

echo "== 4. templates"
if [ -n "$TEMPLATES_SRC" ]; then
    rsync -a --info=progress2 "$TEMPLATES_SRC/" "$REPO/templates/"
fi
if [ -d "$REPO/templates/tpl_xfm_build" ]; then
    (cd "$REPO/templates" && md5_check < "$REPO/Script/data/templates_md5.txt") && echo 'all template files as validated'
else  # the templates of git only: steps 1-2
    (cd "$REPO/templates" && grep -v '  tpl_xfm_build/' "$REPO/Script/data/templates_md5.txt" | md5_check) &&
        echo 'the BCP and MNI templates as validated; no tpl_xfm_build/ (tpl2tpl and avgsubj edges): steps 1-2 only,' \
             'for step 3 run this again with <templates source>'
fi

echo "== 5. $SOFT/neonorm_env.sh"
cat > "$SOFT/neonorm_env.sh" <<EOF
# neonorm on $(hostname -s), written by Script/setup_neonorm.sh on $(date +%F). Source it in the shell, in ~/.bashrc
# and in job scripts: source $SOFT/neonorm_env.sh
export FSLDIR=$FSLDIR
export FSLOUTPUTTYPE=NIFTI_GZ
$FS_ENV
export ANTSPATH=$ANTSPATH
export NEONORM_SYNTHSTRIP_MODELS=$MODELS
export NEONORM_BCP_TEMPLATES=$REPO/templates/BCP-atlas-for_release-Ver2.0.0
export NEONORM_MNI_TEMPLATE=$REPO/templates/tpl-MNI152NLin2009cSym
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate neonorm
export PYTHONNOUSERSITE=1  # no Python packages from ~/.local
EOF
cat "$SOFT/neonorm_env.sh"

echo "== 6. check"
source "$SOFT/neonorm_env.sh"
python "$REPO/Script/neonorm.py" --check
python - "$REPO/Script" <<'EOF'
# The versions found against those validated on the BCP data (README.md); a difference is no error, but results
# may differ from those of the validation
import sys
sys.path.insert(0, sys.argv[1])
import neonorm
VALIDATED = {'python': '3.12.2', 'antspyx': '0.6.1', 'SUITPy': '2.0.1', 'nibabel': '5.3.2', 'numpy': '2.4.6',
             'pandas': '2.3.3', 'scipy': '1.15.3', 'fsl_version': '6.0.6.2', 'ants_version': '2.6.2',
             'freesurfer_version': 'freesurfer-linux-centos8_x86_64-8.0.0-20250203',
             'synthstrip_model_md5': '27f86a16f396ac6db3023c2f70b97ae1',
             'synthstrip_infant_model_md5': '37a6e0fa34adec4ab4ffba22de551595'}
info = neonorm.environment_info()
diff = {k: (info.get(k), v) for k, v in VALIDATED.items() if not str(info.get(k) or '').startswith(v)}
print('\nall versions as validated' if not diff else '\ndifferent from the validation (found, validated):')
for k, (found, v) in diff.items():
    print(f'  {k}: {found}, {v}')
EOF
