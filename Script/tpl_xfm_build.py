"""
neonate_tpl_clean.py

This module provides two ways to submit ANTs-based image processing pipelines:
1. SLURM-based cluster job submission (submit_slurm_job)
2. Local bash execution using nohup (submit_bash_job)

Functions:
- submit_slurm_job(): Submits a SLURM job script for remote execution on HPC clusters
- submit_bash_job(): Executes the same job locally via bash+nohup
- multimodal_register_pipeline(): High-level multi-step ANTs registration workflow (example usage)

Requirements:
- ANTs installed and accessible via ANTSPATH
- SLURM system (for submit_slurm_job)
- bash + nohup (for local execution)

Usage:
Import this module in your pipeline script, configure paths and job parameters,
and call one of the submission functions based on your environment.
"""

# ===== Default configuration =====
DEFAULT_ANTSPATH = "/project/4290000.01/yapwan/toolbox/ANTs/ants-2.6.2" # TODO: [User] Set this to the path where ANTs is installed on your machine
DEFAULT_MEM = "30G"
DEFAULT_TIME = "36:00:00"

import os
import sys
import subprocess
import nibabel as nib
import ants
import re
import numpy as np
import nitools
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import time
from typing import Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)



def ants_to_nib(ants_img):
    data = ants_img.numpy()
    spacing = np.array(ants_img.spacing)
    direction = np.array(ants_img.direction).reshape(3, 3)
    origin = np.array(ants_img.origin)

    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin

    return nib.Nifti1Image(data, affine)


def translate_pair_to_template_space(
    t1_img_path: str,
    t2_img_path: str,
    fix_img_path: str,
    t1_output_path: str,
    t2_output_path: str,
):
    """
    Just translate T1 + T2 to template space WITHOUT resampling.
    T1 defines the transform, T2 follows exactly.

    Steps:
    1. Reorient both to template orientation
    2. Compute COM from T1
    3. Build shared voxel shift + canvas
    4. Construct shared affine
    5. Apply same transform to both T1 and T2
    """

    # Step 0: Reorient both images to template orientation if needed
    t1_ants = ants.image_read(t1_img_path)
    t2_ants = ants.image_read(t2_img_path)
    fix_ants = ants.image_read(fix_img_path)

    fix_dir = ants.get_orientation(fix_ants)
    t1_dir = ants.get_orientation(t1_ants)
    t2_dir = ants.get_orientation(t2_ants)

    if t1_dir != fix_dir:
        print(f"[INFO] T1 reorient: {t1_dir} -> {fix_dir}")
        t1_ants = ants.reorient_image2(t1_ants, orientation=fix_dir)
        t1_img = ants_to_nib(t1_ants)
    else:
        t1_img = nib.load(t1_img_path)
    if t2_dir != fix_dir:
        print(f"[INFO] T2 reorient: {t2_dir} -> {fix_dir}")
        t2_ants = ants.reorient_image2(t2_ants, orientation=fix_dir)
        t2_img = ants_to_nib(t2_ants)
    else:
        t2_img = nib.load(t2_img_path)
    fix_img = nib.load(fix_img_path)
    
    t1_data = t1_img.get_fdata()
    t2_data = t2_img.get_fdata()
    fix_data = fix_img.get_fdata()

    t1_affine = t1_img.affine.copy()
    t2_affine = t2_img.affine.copy()
    fix_affine = fix_img.affine.copy()

    t1_shape = np.array(t1_data.shape)
    t2_shape = np.array(t2_data.shape)
    fix_shape = np.array(fix_data.shape)

    t1_spacing = np.linalg.norm(t1_affine[:3, :3], axis=0)
    t2_spacing = np.linalg.norm(t2_affine[:3, :3], axis=0)
    fix_spacing = np.linalg.norm(fix_affine[:3, :3], axis=0)

    print("T1 shape:", t1_shape, "spacing:", t1_spacing)
    # Warning if T1 and T2 affine different
    if not np.allclose(t1_affine, t2_affine):
        print("[WARNING] T1 and T2 affines differ! This may cause issues with alignment.")
        print("T1 affine:\n", t1_affine)
        print("T2 affine:\n", t2_affine)
    
    # Step 1: Target shape (world-space match)
    fix_world_size = fix_shape * fix_spacing
    target_shape = np.round(fix_world_size / t1_spacing).astype(int)
    print("target shape:", target_shape)

    
    # Step 2: COM: center of mass (only from T1)
    mask = t1_data > np.percentile(t1_data[t1_data > 0], 20)
    coords = np.array(np.nonzero(mask))
    t1_com_vox = coords.mean(axis=1)
    target_center_vox = (target_shape - 1) / 2.0
    shift_vox = np.round(target_center_vox - t1_com_vox).astype(int)
    print("T1 COM:", t1_com_vox)
    print("shift_vox:", shift_vox)

    
    # Step 3: Apply SAME voxel shift to T1 & T2
    def place_into_canvas(data, shape):
        canvas = np.zeros(target_shape, dtype=data.dtype)
        src_slices = []
        dst_slices = []

        for d in range(3):
            if shift_vox[d] >= 0:
                src_start = 0
                dst_start = shift_vox[d]
            else:
                src_start = -shift_vox[d]
                dst_start = 0

            length = min(
                shape[d] - src_start,
                target_shape[d] - dst_start
            )
            src_slices.append(slice(src_start, src_start + length))
            dst_slices.append(slice(dst_start, dst_start + length))

        canvas[
            dst_slices[0],
            dst_slices[1],
            dst_slices[2]
        ] = data[
            src_slices[0],
            src_slices[1],
            src_slices[2]
        ]

        return canvas

    t1_adjusted = place_into_canvas(t1_data, t1_shape)
    t2_adjusted = place_into_canvas(t2_data, t2_data.shape)
    print("placed into canvas")

    
    # Step 4: Construct shared affine
    fix_center_voxel = (fix_shape - 1) / 2.0
    fix_center_world = fix_affine[:3, :3] @ fix_center_voxel + fix_affine[:3, 3]
    fix_dir_norm = fix_affine[:3, :3] / np.linalg.norm(fix_affine[:3, :3], axis=0)
    new_affine = np.eye(4)
    new_affine[:3, :3] = fix_dir_norm * t1_spacing
    new_center_voxel = (target_shape - 1) / 2.0
    new_affine[:3, 3] = fix_center_world - new_affine[:3, :3] @ new_center_voxel
    print("affine constructed")

    
    # Step 5: Save outputs
    t1_img_out = nib.Nifti1Image(t1_adjusted, new_affine)
    t1_img_out.set_qform(new_affine, code=1)
    t1_img_out.set_sform(new_affine, code=1)
    nib.save(t1_img_out, t1_output_path)

    t2_img_out = nib.Nifti1Image(t2_adjusted, new_affine)
    t2_img_out.set_qform(new_affine, code=1)
    t2_img_out.set_sform(new_affine, code=1)
    nib.save(t2_img_out, t2_output_path)
    print(f"[SAVE] T1 -> {t1_output_path}")
    print(f"[SAVE] T2 -> {t2_output_path}")



def multimodal_register_pipeline(modalities, input_files, tpl_root, tpl_month, output_dir, **kwargs):
    """
    Pipeline to perform multi-modal nonlinear image registration to a template and compute Jacobian determinant maps.

    This function:
    1. Registers multi-modal brain images (e.g., T1, T2) to a specified neonatal template using ANTs.
    2. Combines and saves forward and inverse transforms (Affine + SyN).
    3. Computes Jacobian determinant maps from the forward and inverse deformation fields.

    Args:
        modalities (list of str): Modalities to register, typically ["T1", "T2"].
            Each modality must have a corresponding preprocessed input file in `input_files`.

        input_files (dict): Dictionary containing paths to the preprocessed brain images.
            Required keys match the `modalities` list. E.g.:
                {
                    "T1": "/path/to/T1_Brain.nii.gz",
                    "T2": "/path/to/T2_Brain.nii.gz"
                }

        tpl_root (str): Root directory containing BCP template images.
            Should include subfolders like "00Month", "01Month", etc.

        tpl_month (str): Two-digit string representing the template month (e.g., "00", "03", "06").
            Used to locate the corresponding template (e.g., "BCP-00M-T1.nii.gz").

        output_dir (str): Directory where both input images are stored and output results will be saved.
            This folder will store registration outputs, transform files, warped images, and Jacobian maps.

    Keyword Args (**kwargs):
        steps (list of int): Stages to run. Default: [1, 2, 3].
            - 1: Registration using `antsRegistration`
            - 2: Combine transform fields and save inverse fields
            - 3: Compute Jacobian determinant maps from fields
    """

    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    

    # tpl
    register_tpls = {
            "T1": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T1.nii.gz'),
            "T2": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T2.nii.gz')
        }

    # Generate commands
    commands = []
    

    # Step 1: Registration
    if 1 in steps:
        brain_img_T1 = input_files['T1']
        brain_img_T2 = input_files['T2']
        tpl_T1 = register_tpls['T1']
        tpl_T2 = register_tpls['T2']
        fix_mask = tpl_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')
        print("[INFO] No moving mask used")
        mask_param = fix_mask
        
        out_prefix = os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_')
        warped_output = os.path.join(output_dir, f'T1_Brain_pad_Norm_to_{tpl_month}Mtpl_Warped.nii.gz')

        cmd = f"""
        antsRegistration --verbose 1 -d 3 \\
        -o [{out_prefix}, {warped_output}] \\
        -x {mask_param} \\
        \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -t Rigid[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -t Affine[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -t SyN[0.1, 3, 0] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        -u 1 -z 1
        """
        commands.append(cmd.strip())
    
    # Step 2: Combine transforms
    if 2 in steps:
        tpl_T1 = register_tpls['T1']
        combine_xfm_cmd = f"""
        antsApplyTransforms -d 3 \\
        -r {tpl_T1} \\
        -o [{os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")},1] \\
        -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_1Warp.nii.gz')} \\
        -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_0GenericAffine.mat')}
        """
        # Only need the forward combined transform for JD (direct),
        # inverse is optional and can generated based on 0Affine.mat and 1InverseWarp.nii.gz if needed, 
        # So we keep the intermediate xfm files for now and skip the inverse combined 
        # brain_img_T1 = input_files['T1']
        # combine_xfm_cmd = f"""
        # antsApplyTransforms -d 3 \\
        # -r {brain_img_T1} \\
        # -o [{os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp_Inverse.nii.gz")},1] \\
        # -t [{os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_0GenericAffine.mat')},1] \\
        # -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_1InverseWarp.nii.gz')} 
        # """
        commands.append(combine_xfm_cmd.strip())

    # Step 3: Jacobian
        # Usage:
            # CreateJacobianDeterminantImage
            # imageDimension
            # deformationField
            # outputImage
            # [doLogJacobian=0]
            # [useGeometric=0]
            # [deformationGradient=0]
    if 3 in steps:
       
        out_field = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")
        jd_file = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_log_JD.nii.gz")
        cmd = f"""
        CreateJacobianDeterminantImage 3 \\
        {out_field} \\
        {jd_file} 1 0
        """

        # out_field = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp_Inverse.nii.gz")
        # jd_file = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_Inverse_log_geometric_JD.nii.gz")
        # cmd = f"""
        # CreateJacobianDeterminantImage 3 \\
        # {out_field} \\
        # {jd_file} 1 1
        # """
        commands.append(cmd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"T1T2_{tpl_month}Mtpl"
    if slurm:
        submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=kwargs.get("num_threads", 6),
            time_limit=kwargs.get("time_limit", "36:00:00"),
            mem=kwargs.get("mem", "30G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            gpu_type=kwargs.get("gpu_type", None),
            email=kwargs.get("email", None),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
    else:
        job_script = os.path.join(log_dir, f'{job_prefix}.sh')
        output_log = os.path.join(log_dir, f'{job_prefix}.out')
        error_log = os.path.join(log_dir, f'{job_prefix}.err')
        submit_bash_job(full_cmd, job_script, job_prefix, output_log, error_log, num_threads, verbose)

    return True


def multimodal_register_pipeline_qc(
    modalities,
    input_files,
    tpl_root,
    tpl_month,
    output_dir,
    **kwargs
):

    num_threads = kwargs.get('num_threads', 6)
    verbose = kwargs.get('verbose', False)
    max_iter = kwargs.get('max_iterations', 5)
    threshold = kwargs.get('qc_threshold', 0.5)

    # templates
    tpl_T1 = os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T1.nii.gz')
    tpl_T2 = os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T2.nii.gz')
    brain_mask = tpl_T1.replace('-T1.nii.gz', '-Mask.nii.gz')

    brain_img_T1 = input_files['T1']
    brain_img_T2 = input_files['T2']

    out_prefix = os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_')
    warped_output = os.path.join(output_dir, f'T1_Brain_pad_Norm_to_{tpl_month}Mtpl_Warped.nii.gz')

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    sub_id = os.path.basename(output_dir)

    # ======================
    # FULL BASH PIPELINE
    # ======================

    cmd = f"""
MAX_ITER={max_iter}
THRESH={threshold}

tpl_T1="{tpl_T1}"
tpl_T2="{tpl_T2}"
brain_img_T1="{brain_img_T1}"
brain_img_T2="{brain_img_T2}"
brain_mask="{brain_mask}"
out_prefix="{out_prefix}"
warped_output="{warped_output}"

for ((i=1;i<=MAX_ITER;i++))
do
    echo "=============================="
    echo "Trial $i / $MAX_ITER"
    echo "=============================="

    rm -f ${{warped_output}}

    antsRegistration --verbose 1 -d 3 \\
    -o [${{out_prefix}}, ${{warped_output}}] \\
    -x ${{brain_mask}} \\
    \\
    -m MI[${{tpl_T1}}, ${{brain_img_T1}}, 1, 32, Regular, 0.25] \\
    -m MI[${{tpl_T2}}, ${{brain_img_T2}}, 1, 32, Regular, 0.25] \\
    -t Rigid[0.1] \\
    -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
    \\
    -m MI[${{tpl_T1}}, ${{brain_img_T1}}, 1, 32, Regular, 0.25] \\
    -m MI[${{tpl_T2}}, ${{brain_img_T2}}, 1, 32, Regular, 0.25] \\
    -t Affine[0.1] \\
    -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
    \\
    -m MI[${{tpl_T1}}, ${{brain_img_T1}}, 1, 32, Regular, 0.25] \\
    -m MI[${{tpl_T2}}, ${{brain_img_T2}}, 1, 32, Regular, 0.25] \\
    -t SyN[0.1, 3, 0] \\
    -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
    -u 1 -z 1

    # ======================
    # QC: Pearson correlation (via Python)
    # ======================
    corr=$(python - <<END
import nibabel as nib
import numpy as np

tpl = nib.load("{tpl_T1}").get_fdata()
warped = nib.load("{warped_output}").get_fdata()
mask = nib.load("{brain_mask}").get_fdata() > 0

tpl = tpl[mask]
warped = warped[mask]

if tpl.std() == 0 or warped.std() == 0:
    print(0)
else:
    print(np.corrcoef(tpl, warped)[0,1])
END
)

    echo "Correlation = $corr"

    pass=$(echo "$corr >= $THRESH" | bc -l)

    if [ "$pass" -eq 1 ]; then
        echo "PASS QC"

        # ======================
        # Step 2: Combine transform
        # ======================
        antsApplyTransforms -d 3 \\
        -r {tpl_T1} \\
        -o [{os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")},1] \\
        -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_1Warp.nii.gz')} \\
        -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_0GenericAffine.mat')}

        # ======================
        # Step 3: Jacobian
        # ======================
        CreateJacobianDeterminantImage 3 \\
        {os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")} \\
        {os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_log_JD.nii.gz")} 1 0

        echo "Pipeline finished successfully"
        exit 0
    else
        echo "FAIL QC"
    fi

    if [ "$i" -eq "$MAX_ITER" ]; then
        echo "FAILED after $MAX_ITER attempts"
        exit 1
    fi
done
"""

    # ======================
    # SUBMIT SLURM JOB
    # ======================
    job_prefix = f"{sub_id.replace('sub-','')}_q_{tpl_month}"

    submit_slurm_job(
        full_cmd=cmd,
        log_dir=log_dir,
        job_prefix=job_prefix,
        num_threads=num_threads,
        time_limit=kwargs.get("time_limit", "36:00:00"),
        mem=kwargs.get("mem", "30G"),
        ntasks=1,
        use_gpu=kwargs.get("use_gpu", False),
        gpu_type=kwargs.get("gpu_type", None),
        email=kwargs.get("email", None),
        ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
        verbose=verbose,
    )

    return True
    


def unimodal_register_pipeline(modalities, input_files, tpl_root, tpl_month, output_dir, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    

    # tpl
    register_tpls = {
            "T1": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T1.nii.gz'),
            "T2": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T2.nii.gz')
        }

    
    

    # Step 1: Registration
    for modality in modalities:
        # Generate commands
        commands = []
    
        brain_img = input_files[modality]
        tpl = register_tpls[modality]
        fix_mask = tpl.replace(f'-{modality}.nii.gz', '-Mask.nii.gz')
        print("[INFO] No moving mask used")
        mask_param = fix_mask
        out_prefix = os.path.join(output_dir, f'{modality}only_Brain_pad_Norm_to_{tpl_month}Mtpl_')
        warped_output = os.path.join(output_dir, f'{modality}only_Brain_pad_Norm_to_{tpl_month}Mtpl_Warped.nii.gz')

        cmd = f"""
        antsRegistration --verbose 1 -d 3 \\
        -o [{out_prefix}, {warped_output}] \\
        -x {mask_param} \\
        \\
        -m MI[{tpl}, {brain_img}, 1, 32, Regular, 0.25] \\
        -t Rigid[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl}, {brain_img}, 1, 32, Regular, 0.25] \\
        -t Affine[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl}, {brain_img}, 1, 32, Regular, 0.25] \\
        -t SyN[0.1, 3, 0] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        -u 1 -z 1
        """
        commands.append(cmd.strip())
        

        # Combine all commands
        full_cmd = "\n\n".join(commands)
        # print(full_cmd)

        # Submit
        log_dir = os.path.join(output_dir, "log")
        job_prefix = f"{modality}o_{tpl_month}Mtpl"
        if slurm:
            submit_slurm_job(
                full_cmd=full_cmd,
                log_dir=log_dir,
                job_prefix=job_prefix,
                num_threads=kwargs.get("num_threads", 6),
                time_limit=kwargs.get("time_limit", "36:00:00"),
                mem=kwargs.get("mem", "30G"),
                ntasks=kwargs.get("ntasks", 1),
                use_gpu=kwargs.get("use_gpu", False),
                gpu_type=kwargs.get("gpu_type", None),
                email=kwargs.get("email", None),
                ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
                dependency_jobid=kwargs.get("dependency_jobid", None),
                verbose=verbose,
            )
        else:
            job_script = os.path.join(log_dir, f'{job_prefix}.sh')
            output_log = os.path.join(log_dir, f'{job_prefix}.out')
            error_log = os.path.join(log_dir, f'{job_prefix}.err')
            submit_bash_job(full_cmd, job_script, job_prefix, output_log, error_log, num_threads, verbose)

    return True

def multimodal_register_lesion_pipel(modalities, input_files, tpl_root, tpl_month, output_dir, **kwargs):

    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    

    # tpl
    register_tpls = {
            "T1": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T1.nii.gz'),
            "T2": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T2.nii.gz')
        }

    # Generate commands
    commands = []
    

    # Step 1: Registration

    brain_img_T1 = input_files['T1']
    brain_img_T2 = input_files['T2']
    tpl_T1 = register_tpls['T1']
    tpl_T2 = register_tpls['T2']
    fix_mask = tpl_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')
    # mov mask options:
    for mov_mask in [True, False]:
        default_mov_mask = brain_img_T1.replace('.nii.gz', '_mask.nii.gz')
        if mov_mask is False:
            print("[INFO] No moving mask used")
            mask_param = fix_mask
            out_key = f'T1T2_Brain_pad_lesion_Norm_withoutmovmask_to'
        elif mov_mask is True:
            print("[INFO] Using default subject moving mask")
            mov_mask_path = default_mov_mask
            mask_param = f"[{fix_mask}, {mov_mask_path}]"
            out_key = f'T1T2_Brain_pad_lesion_Norm_withmovmask_to'
            

        out_prefix = os.path.join(output_dir, f'{out_key}_{tpl_month}Mtpl_')
        warped_output = os.path.join(output_dir, f'{out_key}_{tpl_month}Mtpl_Warped.nii.gz')

        cmd = f"""
        antsRegistration --verbose 1 -d 3 \\
        -o [{out_prefix}, {warped_output}] \\
        -x {mask_param} \\
        \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -t Rigid[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -t Affine[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -t SyN[0.1, 3, 0] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        -u 1 -z 1
        """
        commands.append(cmd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"les_{tpl_month}Mtpl"
    if slurm:
        submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=kwargs.get("num_threads", 6),
            time_limit=kwargs.get("time_limit", "36:00:00"),
            mem=kwargs.get("mem", "30G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            gpu_type=kwargs.get("gpu_type", None),
            email=kwargs.get("email", None),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
    else:
        job_script = os.path.join(log_dir, f'{job_prefix}.sh')
        output_log = os.path.join(log_dir, f'{job_prefix}.out')
        error_log = os.path.join(log_dir, f'{job_prefix}.err')
        submit_bash_job(full_cmd, job_script, job_prefix, output_log, error_log, num_threads, verbose)

    return True


def multimodal_tpl_register(modalities, tpl_root, tpl_mov_month, tpl_fix_month, output_dir, **kwargs):
    """
    Performs multi-modal registration and JD calculation for neiborhood templates (00M->01M, 01M->02M).
    
    Args:
        modalities (list of str): Modalities to process (e.g., ["T1", "T2"])
        tpl_root (str): Root directory of templates
        tpl_mov_month (str): Moving template month (e.g., "00", "01", "02", etc.)
        tpl_fix_month (str): Fixed template month (e.g., "00", "01", "02", etc.)
        Output_dir (str): Directory that contains all output files (e.g., transforms, Jacobians, logs)

    Optional keyword arguments (**kwargs):
        - steps, slurm, num_threads, etc.
    """
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])

    # tpl
    tpl_mov = {
            "T1": os.path.join(tpl_root, f'{tpl_mov_month}Month/BCP-{tpl_mov_month}M-T1.nii.gz'),
            "T2": os.path.join(tpl_root, f'{tpl_mov_month}Month/BCP-{tpl_mov_month}M-T2.nii.gz')  
            }
    tpl_fix = {
            "T1": os.path.join(tpl_root, f'{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T1.nii.gz'),
            "T2": os.path.join(tpl_root, f'{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T2.nii.gz')
            
        }
    tpl_mov_T1 = tpl_mov['T1']
    tpl_mov_T2 = tpl_mov['T2']
    tpl_fix_T1 = tpl_fix['T1']
    tpl_fix_T2 = tpl_fix['T2']
    brain_mask = tpl_fix_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')

    commands = []


    # Step 1: Registration
    if 1 in steps:
        out_prefix = os.path.join(output_dir, f'T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_')
        warped_output = os.path.join(output_dir, f'T1_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_Warped.nii.gz')
        cmd = f"""
        antsRegistration -d 3 \\
        -o [{out_prefix}, {warped_output}] \\
        -x {brain_mask} \\
        \\
        -m MI[{tpl_fix_T1}, {tpl_mov_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_fix_T2}, {tpl_mov_T2}, 1, 32, Regular, 0.25] \\
        -t Rigid[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl_fix_T1}, {tpl_mov_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_fix_T2}, {tpl_mov_T2}, 1, 32, Regular, 0.25] \\
        -t Affine[0.1] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        \\
        -m MI[{tpl_fix_T1}, {tpl_mov_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_fix_T2}, {tpl_mov_T2}, 1, 32, Regular, 0.25] \\
        -t SyN[0.1, 3, 0] \\
        -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
        -u 1 -z 1
        """
        commands.append(cmd.strip())

    # # Step 2: Combine transforms
    # if 2 in steps:
    #     combine_xfm_cmd = f"""
    #     antsApplyTransforms -d 3 \\
    #     -r {tpl_fix_T1} \\
    #     -o [{os.path.join(output_dir, f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_AffWarp.nii.gz")},1] \\
    #     -t {os.path.join(output_dir, f'T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz')} \\
    #     -t {os.path.join(output_dir, f'T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat')}
    #     """
    #     commands.append(combine_xfm_cmd.strip())

    # # Step 3: Jacobian
    # if 3 in steps:
    #     out_field = os.path.join(output_dir, f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_AffWarp.nii.gz")
    #     jd_file = os.path.join(output_dir, f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_geometric_JD.nii.gz")
    #     cmd = f"""
    #     CreateJacobianDeterminantImage 3 \\
    #     {out_field} \\
    #     {jd_file} 0 1
    #     """
    #     commands.append(cmd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"T1T2_{tpl_mov_month}Mtpl_{tpl_fix_month}Mtpl"
    if slurm:
        submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=kwargs.get("num_threads", 6),
            time_limit=kwargs.get("time_limit", "36:00:00"),
            mem=kwargs.get("mem", "30G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            gpu_type=kwargs.get("gpu_type", None),
            email=kwargs.get("email", None),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
    else:
        job_script = os.path.join(log_dir, f'{job_prefix}.sh')
        output_log = os.path.join(log_dir, f'{job_prefix}.out')
        error_log = os.path.join(log_dir, f'{job_prefix}.err')
        submit_bash_job(full_cmd, job_script, f"T1T2_{tpl_mov_month}_{tpl_fix_month}", output_log, error_log, num_threads, verbose)

    return True



def generate_xfm_between_tpl_viasubj(tpl_mov_month, tpl_fix_month, output_dir, tpl_root, **kwargs):  
    """
    Concatenate the displacement field tpl_mov->subj and subj->tpl_fix to get tpl_mov->tpl_fix
    and calculate the Jacobian determinant.

    Args:
        tpl_mov_month (str): Moving template month (e.g., "00", "01", "02", etc.)
        tpl_fix_month (str): Fixed template month (e.g., "00", "01", "02", etc.)
        output_dir (str): Directory that contains all output files (e.g., transforms, Jacobians, logs)
        tpl_root (str): Root directory of templates

    """
    
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)

    print(f"Registering from month {tpl_mov_month} ➜ {tpl_fix_month}...")

    commands = []
    tpl_mov_img_path = f'{tpl_root}/{tpl_mov_month}Month/BCP-{tpl_mov_month}M-T1.nii.gz'
    tpl_fix_img_path = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T1.nii.gz'

    xfm_chain = []
    
    tpl_mov_to_subj_warp = os.path.join(
        output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_1InverseWarp.nii.gz")
    xfm_chain.append(f"-t {tpl_mov_to_subj_warp}")
    tpl_mov_to_subj_aff = os.path.join(
        output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_0GenericAffine.mat")
    xfm_chain.append(f"-t [{tpl_mov_to_subj_aff},1]")
    
    subj_to_tpl_fix_aff = os.path.join(
        output_dir,f"T1T2_Brain_pad_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat")
    xfm_chain.append(f"-t {subj_to_tpl_fix_aff}")
    subj_to_tpl_fix_warp = os.path.join(
        output_dir,f"T1T2_Brain_pad_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz")
    xfm_chain.append(f"-t {subj_to_tpl_fix_warp}")
    xfm_chain.reverse()
    # Format lines: add "\" to all except last
    xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
    xfm_lines.append(xfm_chain[-1])
    xfm_chain_str = "\n    ".join(xfm_lines)  

    # Step 1: compose transforms
    composed_path = os.path.join(
        output_dir,
        f"Displacement_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_viasubj.nii.gz"
    )
    # No indentation for the multi-line string to avoid leading spaces in the command
    cmd_compose = f"""
    antsApplyTransforms -d 3 \\
    -r {tpl_fix_img_path} \\
    -o [{composed_path},1] \\
    {xfm_chain_str} 
    """
    commands.append(cmd_compose.strip())


    # # Step 2: Jacobian determinant
    # jd_out = composed_path.replace(".nii.gz", "_geometric_JD.nii.gz")
    # cmd_jd = f"""
    # CreateJacobianDeterminantImage 3 \\
    # {composed_path} \\
    # {jd_out} 0 1
    # """
    # commands.append(cmd_jd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    print(full_cmd)

    # Submit
    job_prefix = f"avgsubj_{tpl_mov_month}M_{tpl_fix_month}M"
    log_dir = os.path.join(output_dir, "log")
    if slurm:
        submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=num_threads,
            time_limit=kwargs.get("time_limit", "36:00:00"),
            mem=kwargs.get("mem", "30G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            gpu_type=kwargs.get("gpu_type", None),
            email=kwargs.get("email", None),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
    else:
        job_script = os.path.join(log_dir, f'{job_prefix}.sh')
        output_log = os.path.join(log_dir, f'{job_prefix}.out')
        error_log = os.path.join(log_dir, f'{job_prefix}.err')
        submit_bash_job(full_cmd, job_script, job_prefix, output_log, error_log, num_threads, verbose)

    return True



def tpl_transf_concate_resli(transf_type, tpl_mov_month, tpl_fix_month, tpl_root, **kwargs):
    """
    Reslice the tpl_mov to tpl_fix using two types displacement field: 'directtpl', 'averagesubj'

    Args:
        transf_type (str): 'directtpl' or 'averagesubj'
        tpl_mov_month (str): Moving template month (e.g., "00", "01", "02", etc.)
        tpl_fix_month (str): Fixed template month (e.g., "00", "01", "02", etc.)
        tpl_root (str): Root directory of templates
    """
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)

    # Tpl paths
    tpl_mov_img = f'{tpl_root}/{tpl_mov_month}Month/BCP-{tpl_mov_month}M-T1.nii.gz'
    tpl_fix_img = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T1.nii.gz'
    tpl_trans_dir = f"{tpl_root}/tpl_xfm_build"

    # Get the stepwise tpl order to concanate for tpls not neighbors
    # For exmple, 00M->02M = 00M->01M + 01M->02M
    tpl_dirs = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
    tpl_months = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)
    month_order = tpl_months
    idx_mov = month_order.index(tpl_mov_month)
    idx_fix = month_order.index(tpl_fix_month)
    requires_chaining = abs(idx_mov - idx_fix) > 1

    output_xfm = os.path.join(tpl_trans_dir, 'tpl_trans', f"Displacement_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_by_{transf_type}.nii.gz")
    tpl_warped_path = os.path.join(tpl_trans_dir, 'tpl_trans', f"Tpl_{tpl_mov_month}Mtpl_warp_to_{tpl_fix_month}Mtpl_by_{transf_type}.nii.gz")

    commands = []
    if requires_chaining:
        path_months = month_order[idx_mov:idx_fix + 1]

        xfm_chain = []
        for k in range(len(path_months) - 1):
            m_from = path_months[k]
            m_to = path_months[k + 1]
            if transf_type=='tplonly':
                aff_path = os.path.join(
                    tpl_trans_dir, transf_type, 
                    f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat"
                )
                xfm_chain.append(f"-t {aff_path}")
                warp_path = os.path.join(
                    tpl_trans_dir, transf_type, 
                    f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz"
                )
                xfm_chain.append(f"-t {warp_path}")
            elif transf_type=='avgsubj_train':
                xfm_path = os.path.join(tpl_trans_dir, transf_type.replace("train", ""), f'{m_from}-{m_to}M',
                f"Avg_Displacement_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_viasubj_halfsplit_train.nii.gz")
                xfm_chain.append(f"-t {xfm_path}")
        xfm_chain.reverse()
        # Format lines: add "\" to all except last
        xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
        xfm_lines.append(xfm_chain[-1])
        xfm_chain_str = "\n    ".join(xfm_lines)  
        
        combine_xfm_cmd = f"""
            antsApplyTransforms \\
            -d 3 \\
            -r {tpl_fix_img} \\
            -o [{output_xfm}, 1] \\
            {xfm_chain_str} 
            """
        commands.append(combine_xfm_cmd.strip())
        
        reslice_cmd = f"""
            antsApplyTransforms \\
            -d 3 \\
            -i {tpl_mov_img} \\
            -r {tpl_fix_img} \\
            -o {tpl_warped_path} \\
            {xfm_chain_str}
            """
        commands.append(reslice_cmd.strip())
        
    else:
        xfm_chain = []
        if transf_type=='tplonly':
            aff_path = os.path.join(
                tpl_trans_dir, transf_type, 
                f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat"
            )
            xfm_chain.append(f"-t {aff_path}")
            warp_path = os.path.join(
                tpl_trans_dir, transf_type, 
                f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz"
            )
            xfm_chain.append(f"-t {warp_path}")
        elif transf_type=='avgsubj_train':
            xfm_path = os.path.join(tpl_trans_dir, transf_type.replace("_train", ""), f'{tpl_mov_month}-{tpl_fix_month}M',
            f"Avg_Displacement_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_viasubj_halfsplit_train.nii.gz")
            xfm_chain.append(f"-t {xfm_path}")
        # # copy xfm_path to output_xfm
        # copy_cmd = f"cp {xfm_path} {output_xfm}"
        # commands.append(copy_cmd.strip())  
        xfm_chain.reverse()
        # Format lines: add "\" to all except last
        xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
        xfm_lines.append(xfm_chain[-1])
        xfm_chain_str = "\n    ".join(xfm_lines)    
        
        # combine_xfm_cmd = f"""
        #     antsApplyTransforms \\
        #     -d 3 \\
        #     -r {tpl_fix_img} \\
        #     -o [{output_xfm}, 1] \\
        #     {xfm_chain_str} 
        #     """
        # commands.append(combine_xfm_cmd.strip())

        reslice_cmd = f"""
            antsApplyTransforms \\
            -d 3 \\
            -i {tpl_mov_img} \\
            -r {tpl_fix_img} \\
            -o {tpl_warped_path} \\
            {xfm_chain_str} 
            """
        commands.append(reslice_cmd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    print(full_cmd)

    # Submit
    log_dir = os.path.join(tpl_root, "tpl_xfm_build/log")
    job_prefix = f"resl_{tpl_mov_month}_{tpl_fix_month}_{transf_type}_tpl"
    if slurm:
        submit_slurm_job(
                full_cmd=full_cmd,
                log_dir=log_dir,
                job_prefix=job_prefix,
                num_threads=kwargs.get("num_threads", 6),
                time_limit=kwargs.get("time_limit", "36:00:00"),
                mem=kwargs.get("mem", "30G"),
                ntasks=kwargs.get("ntasks", 1),
                use_gpu=kwargs.get("use_gpu", False),
                gpu_type=kwargs.get("gpu_type", None),
                email=kwargs.get("email", None),
                ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
                dependency_jobid=kwargs.get("dependency_jobid", None),
                verbose=verbose,
            )
    else:
        job_script = os.path.join(log_dir, f'{job_prefix}.sh')
        output_log = os.path.join(log_dir, f'{job_prefix}.out')
        error_log = os.path.join(log_dir, f'{job_prefix}.err')
        submit_bash_job(full_cmd, job_script, job_prefix, output_log, error_log, num_threads, verbose)
    return True




def subj_concate_xfm_JD_and_resli(transf_type, data_dir, sub_list, tpl_mov_month, tpl_fix_month, tpl_root, **kwargs):
    """
    Reslice the subject to tpl_fix space via tpl_mov using two types displacement field: 'directtpl', 'averagesubj'
    Args:
        transf_type (str): 'directtpl' or 'averagesubj'
        pipel_dir (str): Pipeline root directory
        dataset_name (str): Dataset name
        sub_list (list of str): List of subject IDs
        tpl_mov_month (str): Moving template month (e.g., "00", "01", "02", etc.)
        tpl_fix_month (str): Fixed template month (e.g., "00", "01", "02", etc.)
        tpl_root (str): Root directory of templates
    """


    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)

    # Tpl paths
    tpl_mov_img = f'{tpl_root}/{tpl_mov_month}Month/BCP-{tpl_mov_month}M-T1.nii.gz'
    tpl_fix_img = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T1.nii.gz'
    tpl_trans_dir = f"{tpl_root}/tpl_xfm_build"
    # Get the stepwise tpl order to concanate for tpls not neighbors
    # For exmple, 00M->02M = 00M->01M + 01M->02M
    tpl_dirs = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
    tpl_months = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)
    month_order = tpl_months
    idx_mov = month_order.index(tpl_mov_month)
    idx_fix = month_order.index(tpl_fix_month)

    commands = []

    for subid in sub_list:
        output_dir = os.path.join(data_dir, subid)
        output_xfm = os.path.join(output_dir, f"Displacement_T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type}.nii.gz")
        jd_file = os.path.join(output_dir, f"Displacement_T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type}_log_JD.nii.gz")
        geometric_jd_file = os.path.join(output_dir, f"Displacement_T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type}_log_geometric_JD.nii.gz")
        out_warped = os.path.join(output_dir, f"T1_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type}_warped.nii.gz")
        
        # Step 1: Combine transforms
        # Determine template warp path
        path_months = month_order[idx_mov:idx_fix + 1]

        xfm_chain = []
        # First: subject ➝ mov template
        # might change to aff and warp 2 files to keep less intermediate files, but for now keep it simple with one combined aff+warp file
        subject_to_mov_aff = os.path.join(
            output_dir,f"T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_0GenericAffine.mat") 
        xfm_chain.append(f"-t {subject_to_mov_aff}")
        subject_to_mov_warp = os.path.join(
            output_dir,f"T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_1Warp.nii.gz") 
        xfm_chain.append(f"-t {subject_to_mov_warp}")

        # Then: mov ➝ fix through intermediate template warps
        for k in range(len(path_months) - 1):
            m_from = path_months[k]
            m_to = path_months[k + 1]
            if transf_type == 'tplonly':
                aff_path = os.path.join(
                    tpl_trans_dir, transf_type, 
                    f"T1T2_{m_from}Mtpl_Norm_to_{m_to}Mtpl_0GenericAffine.mat"
                )
                xfm_chain.append(f"-t {aff_path}")
                warp_path = os.path.join(
                    tpl_trans_dir, transf_type, 
                    f"T1T2_{m_from}Mtpl_Norm_to_{m_to}Mtpl_1Warp.nii.gz"
                )
                xfm_chain.append(f"-t {warp_path}")
            elif transf_type == 'avgsubj':
                xfm_path = os.path.join(
                    tpl_trans_dir, f"Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_viasubj.nii.gz"
                )
                xfm_chain.append(f"-t {xfm_path}")
            elif transf_type == 'avgsubj_train':
                xfm_path = os.path.join(
                    tpl_trans_dir, transf_type.replace("_train", ""), f'{m_from}-{m_to}M',
                    f"Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_viasubj_halfsplit_train.nii.gz"
                )
                xfm_chain.append(f"-t {xfm_path}")
            # elif transf_type == 'avgsubj_train_onlyBCP':
            #     xfm_path = os.path.join(
            #         tpl_trans_dir, transf_type.replace("_train_onlyBCP", ""), f'{m_from}-{m_to}M',
            #         f"Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_viasubj_halfsplit_train_onlyBCP.nii.gz"
            #     )
            #     xfm_chain.append(f"-t {xfm_path}")
            elif transf_type.startswith('avgsubj_train_only'):
                base_type = transf_type.split('_train_')[0]

                xfm_path = os.path.join(
                    tpl_trans_dir,
                    base_type,
                    f'{m_from}-{m_to}M',
                    f"Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_viasubj_halfsplit_train_{transf_type.split('_train_')[1]}.nii.gz"
                )
                xfm_chain.append(f"-t {xfm_path}")
            elif transf_type.startswith('avgsubj_l1o'):
                base_type = transf_type.split('_train_')[0]

                xfm_path = os.path.join(
                    output_dir,
                    f"Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_via_other_subjs.nii.gz"
                )

                xfm_chain.append(f"-t {xfm_path}")

        # Reverse for antsApplyTransforms (last → first)
        xfm_chain.reverse()

        # Format lines: add "\" to all except last
        xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
        xfm_lines.append(xfm_chain[-1])
        xfm_chain_str = "\n    ".join(xfm_lines)  

        # Combine into final command
        combine_xfm_cmd = f"""
            antsApplyTransforms -d 3 \\
            -r {tpl_fix_img} \\
            -o [{output_xfm},1] \\
            {xfm_chain_str}
            """
        commands.append(combine_xfm_cmd.strip())

        generate_jd_cmd = f"""
            CreateJacobianDeterminantImage 3 \\
            {output_xfm} \\
            {jd_file} 1 0
            """
        commands.append(generate_jd_cmd.strip())

        generate_jd_cmd = f"""
            CreateJacobianDeterminantImage 3 \\
            {output_xfm} \\
            {geometric_jd_file} 1 1
            """
        commands.append(generate_jd_cmd.strip())

        reslice_cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {os.path.join(output_dir, f'T1_Brain_pad.nii.gz')} \\
            -r {tpl_fix_img} \\
            -o {out_warped} \\
           {xfm_chain_str}
            """
        commands.append(reslice_cmd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    print(full_cmd)
    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"resl_{tpl_mov_month}_{tpl_fix_month}_{transf_type}_subj"
    if slurm:
        submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=kwargs.get("num_threads", 6),
            time_limit=kwargs.get("time_limit", "36:00:00"),
            mem=kwargs.get("mem", "30G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            gpu_type=kwargs.get("gpu_type", None),
            email=kwargs.get("email", None),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
    else:
        job_script = os.path.join(log_dir, f'{job_prefix}.sh')
        output_log = os.path.join(log_dir, f'{job_prefix}.out')
        error_log = os.path.join(log_dir, f'{job_prefix}.err')
        submit_bash_job(full_cmd, job_script, job_prefix, output_log, error_log, num_threads, verbose)
    return True




def submit_slurm_job(
    full_cmd,
    log_dir,
    job_prefix="Job",
    name_components=None,
    num_threads=16,
    time_limit="36:00:00",
    mem="30G",
    ntasks=1,
    use_gpu=False,
    gpu_type=None,
    email=None,  
    ants_path=DEFAULT_ANTSPATH,  
    dependency_jobid=None,
    verbose=True
    ):

    os.makedirs(log_dir, exist_ok=True)
    name_components = name_components or []
    """
    submit_slurm_job

    Submits a job to SLURM with optional GPU, dependency, and email notifications.

    Args:
        full_cmd (str): The full command to be executed inside the job.
        log_dir (str): Directory to store job script and logs.
        job_prefix (str): Prefix for job name and script files.
        num_threads (int): Number of CPU threads requested.
        time_limit (str): Max runtime (format HH:MM:SS).
        mem (str): RAM requested (e.g., "30G").
        use_gpu (bool): Whether to request a GPU.
        gpu_type (str or None): GPU type constraint (e.g., "A100").
        email (str or None): Email address for SLURM notifications.
        ants_path (str): Path to ANTs installation root.
        dependency_jobid (str or None): SLURM job ID to depend on.
        verbose (bool): If True, prints submission info.

    Returns:
        (str, str): Tuple of (job name, job script path).
    """
    # Unique name with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name_full = f"{job_prefix}"
    job_name_short = log_name_full[:50]  # Limit for safety
    os.makedirs(log_dir, exist_ok=True)
    job_script = os.path.join(log_dir, f"{log_name_full}.sh")
    output_log = os.path.join(log_dir, f"{log_name_full}.out")
    error_log = os.path.join(log_dir, f"{log_name_full}.err")

    # Slurm headers
    slurm_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name_short}",
        f"#SBATCH --output={output_log}",
        f"#SBATCH --error={error_log}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={num_threads}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --exclude=dccn-c[046-047,052-054,063,076-085]"
    ]

    if use_gpu:
        slurm_lines.append("#SBATCH --gres=gpu:1")
        if gpu_type:
            slurm_lines.append(f"#SBATCH --constraint={gpu_type}")

    if email:
        slurm_lines.append(f"#SBATCH --mail-user={email}")
        slurm_lines.append("#SBATCH --mail-type=BEGIN,END,FAIL")

    if dependency_jobid:
        slurm_lines.append(f"#SBATCH --dependency=afterok:{dependency_jobid}")

    # Commands
    slurm_lines += [
        "",
        f"echo 'Starting job: {log_name_full}'",
        f"export ANTSPATH={ants_path}",
        "export PATH=$ANTSPATH/bin:$PATH",
        "export LD_LIBRARY_PATH=$ANTSPATH/lib:$LD_LIBRARY_PATH",
        "echo 'Checking antsBrainExtraction.sh path: '",
        "which antsBrainExtraction.sh || echo '[WARNING] antsBrainExtraction.sh not found in PATH'",
        "echo 'ANTs Version:'",
        "antsRegistration --version || echo '[WARNING] antsRegistration not found'",
        f"{full_cmd}",
        "",
        "if [ $? -ne 0 ]; then",
        f"    echo \"[ERROR] Job failed. See {error_log}\" >> {error_log}",
        "    exit 1",
        "fi",
        "exit 0"
    ]

    # Write script
    with open(job_script, "w") as f:
        f.write("\n".join(slurm_lines))

    os.chmod(job_script, 0o755)

    # Submit job
    cmd_sbatch = ["sbatch", job_script]
    if verbose:
        print(f"[INFO] Submitting job: {log_name_full}")
        print(f"[INFO] Job script: {job_script}")
        print("[INFO] Command:", " ".join(cmd_sbatch))

    result = subprocess.run(cmd_sbatch,capture_output=True,text=True)

    # stdout example: "Submitted batch job 51546474\n"
    stdout = result.stdout.strip()

    if verbose:
        print("[INFO] sbatch output:", stdout)

    # extract job id
    try:
        job_id = stdout.split()[-1]
    except:
        raise RuntimeError(f"Failed to parse job ID from sbatch output: {stdout}")
    if verbose:
        print(f"[INFO] Job ID: {job_id}")
    return job_id
    



def submit_bash_job(full_cmd, job_script, log_name, output_log, error_log, num_threads, verbose):
    """
    Runs a Bash job using `nohup`.

    Args:
        full_cmd (str): The command string that runs the ANTs pipeline.
        job_script (str): Path to the job script.
        log_name (str): Job name.
        output_log (str): Path to output log.
        error_log (str): Path to error log.
        num_threads (int): Number of CPU threads to allocate.
        verbose (bool): If True, prints job execution details.
    """
    with open(job_script, "w") as f:
        f.write(f"""#!/bin/bash
echo "Starting {log_name} processing..."

# Export ANTs binary path (custom system path)
export ANTSPATH={DEFAULT_ANTSPATH}
export PATH=$ANTSPATH/bin:$PATH
export LD_LIBRARY_PATH=$ANTSPATH/lib:$LD_LIBRARY_PATH

START_TIME=$(date +%s)   # Start timer
cd {os.path.dirname(job_script)}

# Ensure all threads are used
export OMP_NUM_THREADS={num_threads}

(
echo "Checking antsBrainExtraction.sh path..."
which antsBrainExtraction.sh || echo "[WARNING] antsBrainExtraction.sh not found in PATH"

echo "ANTs Version:"
antsRegistration --version || echo "[WARNING] antsRegistration not found"

echo "Executing ANTs commands..."
{full_cmd}
) > {output_log} 2> {error_log}

if [ $? -ne 0 ]; then
    echo "[ERROR] Processing failed. Check {error_log}" >> {error_log}
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Processing complete. Total time: $ELAPSED_TIME seconds."
exit 0
""")

    os.chmod(job_script, 0o755)

    if verbose:
        print("[INFO] Running job using bash:", job_script)

    subprocess.Popen(["nohup", "bash", job_script, "&"], stdout=open(output_log, "a"), stderr=open(error_log, "a"))



def pad_to_match_world_space(
    mov_img_path: str,
    fix_img_path: str,
    output_img_path: str
    ) -> None:
    """
    Pad or crop a 3D NIfTI image so that its *world-space dimensions* match a given template,
    while preserving the moving image's voxel resolution (spacing) and content integrity. 
    Re-orients the moving image to RAS if necessary.

    This function does NOT resample intensity values — it only adjusts the voxel grid size
    (by padding or cropping symmetrically) and then redefines the affine matrix so that the
    world-space alignment and orientation match the fixed image.

    Parameters
    ----------
    mov_img_path : str
        Path to the moving image (e.g., "subject_T1.nii.gz").
    fix_img_path : str
        Path to the fixed/template image (e.g., "template_T1.nii.gz").
    output_img_path : str
        Path to save the adjusted image.

    Output
    ------
    A new NIfTI image will be saved to `output_img_path` that:
        • Has approximately the same world-space size and center as the fixed image.
        • Retains the moving image’s original resolution (spacing).
        • Has zero-padding or symmetric cropping applied to match world dimensions.
    """

    # === Step 1: Load input images ===
    mov_img = nib.load(mov_img_path)
    # check if the moving image is in RAS orientation, if not, reorient it
    from nibabel.orientations import axcodes2ornt, ornt_transform
    if nib.aff2axcodes(mov_img.affine) != ('R', 'A', 'S'):
        print(f"[INFO] Moving image is not in RAS orientation. Reorienting...")
        mov_img = nib.as_closest_canonical(mov_img)
    fix_img = nib.load(fix_img_path)
    mov_data = mov_img.get_fdata()
    fix_data = fix_img.get_fdata()

    mov_affine = mov_img.affine
    fix_affine = fix_img.affine

    mov_spacing = np.linalg.norm(mov_affine[:3, :3], axis=0)
    fix_spacing = np.linalg.norm(fix_affine[:3, :3], axis=0)

    mov_shape = np.array(mov_data.shape)
    fix_shape = np.array(fix_data.shape)

    # === Step 2: Compute world-space size of fixed image ===
    fix_world_size = fix_shape * fix_spacing
    print(f"[INFO] Fixed image shape: {fix_shape}, spacing: {fix_spacing}")
    print(f"[INFO] Fixed image world-space size: {fix_world_size}")

    # === Step 3: Determine new voxel shape to match world size ===
    target_shape = np.round(fix_world_size / mov_spacing).astype(int)
    print(f"[INFO] Moving image original shape: {mov_shape}, spacing: {mov_spacing}")
    print(f"[INFO] Target shape for moving image: {target_shape}")

    # === Step 4: Symmetric pad or crop to target shape ===
    adjusted_data = mov_data.copy()
    current_shape = mov_data.shape
    mov_slices = []

    for dim in range(3):
        diff = target_shape[dim] - current_shape[dim]
        if diff == 0:
            mov_slices.append(slice(0, current_shape[dim]))
            print(f"[No act] Axis {dim}")
            continue
        elif diff > 0:
            # Padding
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width = [(0, 0)] * 3
            pad_width[dim] = (pad_before, pad_after)
            adjusted_data = np.pad(adjusted_data, pad_width, mode='constant', constant_values=0)
            mov_slices.append(slice(pad_before, pad_before + current_shape[dim]))
            print(f"[PAD] Axis {dim}: before={pad_before}, after={pad_after}")
        else:
            # Cropping
            crop_total = -diff
            crop_before = crop_total // 2
            crop_after = crop_total - crop_before
            slc = slice(crop_before, current_shape[dim] - crop_after)
            adjusted_data = adjusted_data.take(indices=range(slc.start, slc.stop), axis=dim)
            mov_slices.append(slice(0, slc.stop - slc.start))
            print(f"[CROP] Axis {dim}: before={crop_before}, after={crop_after}")
        current_shape = adjusted_data.shape  # Update shape after each dim

    # === Step 5: Final shape check ===
    assert adjusted_data.shape == tuple(target_shape), \
        f"[ERROR] Final shape mismatch: got {adjusted_data.shape}, expected {target_shape}"
    print(f"[INFO] Adjusted moving image shape: {adjusted_data.shape}")

    # === Step 6: Try to verify internal data consistency (for padded case) ===
    try:
        retained_data = adjusted_data[mov_slices[0], mov_slices[1], mov_slices[2]]
        if retained_data.shape == mov_data.shape and np.allclose(retained_data, mov_data):
            print("[DEBUG] Original moving data preserved in padded/cropped result ✅")
        else:
            print("[WARNING] Retained data differs from original moving image ⚠️")
    except Exception as e:
        print(f"[WARNING] Could not verify retained region: {e}")

    # === Step 7: Construct new affine ===
    fix_center_voxel = fix_shape / 2.0
    fix_center_world = fix_affine[:3, :3] @ fix_center_voxel + fix_affine[:3, 3]
    fix_dir_norm = fix_affine[:3, :3] / np.linalg.norm(fix_affine[:3, :3], axis=0)

    new_affine = np.eye(4)
    new_affine[:3, :3] = fix_dir_norm * mov_spacing
    mov_center_voxel = target_shape / 2.0
    new_affine[:3, 3] = fix_center_world - new_affine[:3, :3] @ mov_center_voxel

    print(f"[INFO] New affine matrix:\n{new_affine}")

    # === Step 8: Save adjusted image ===
    new_img = nib.Nifti1Image(adjusted_data, affine=new_affine)
    new_img.set_qform(new_affine)
    new_img.set_sform(new_affine)
    nib.save(new_img, output_img_path)

    print(f"[SAVE] Recentered image saved to: {output_img_path}")
