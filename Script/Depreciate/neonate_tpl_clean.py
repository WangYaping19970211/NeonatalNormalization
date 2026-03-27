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
import re
import numpy as np
import nitools
from scipy.ndimage import map_coordinates
from tqdm import tqdm
from typing import Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

import nibabel as nib
import numpy as np

def pad_to_match_world_space(
    mov_img_path: str,
    fix_img_path: str,
    output_img_path: str
) -> None:
    """
    Pad or crop a 3D NIfTI image so that its *world-space dimensions* match a given template,
    while preserving the moving image's voxel resolution (spacing) and content integrity.

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
        brain_mask = tpl_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')

        out_prefix = os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_')
        warped_output = os.path.join(output_dir, f'T1_Brain_pad_Norm_to_{tpl_month}Mtpl_Warped.nii.gz')

        cmd = f"""
        antsRegistration --verbose 1 -d 3 \\
        -o [{out_prefix}, {warped_output}] \\
        -x {brain_mask} \\
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
        brain_img_T1 = os.path.join(output_dir, f'T1_Brain_pad.nii.gz')
        combine_xfm_cmd = f"""
        antsApplyTransforms -d 3 \\
        -r {tpl_T1} \\
        -o [{os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")},1] \\
        -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_1Warp.nii.gz')} \\
        -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_0GenericAffine.mat')}
        """
        commands.append(combine_xfm_cmd.strip())
        combine_xfm_cmd_inv = f"""
        antsApplyTransforms -d 3 \\
        -r {brain_img_T1} \\
        -o [{os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp_Inverse.nii.gz")},1] \\
        -t [{os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_0GenericAffine.mat')},1] \\
        -t {os.path.join(output_dir, f'T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_1InverseWarp.nii.gz')} 
        """
        commands.append(combine_xfm_cmd_inv.strip())

    # Step 3: Jacobian
    if 3 in steps:
        out_field = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")
        jd_file = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_Jacobian.nii.gz")
        cmd = f"""
        CreateJacobianDeterminantImage 3 \\
        {out_field} \\
        {jd_file} 1 0
        """
        commands.append(cmd.strip())

        out_field = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp_Inverse.nii.gz")
        jd_file = os.path.join(output_dir, f"T1T2_Brain_pad_Norm_to_{tpl_month}Mtpl_Inverse_Jacobian.nii.gz")
        cmd_inv = f"""
        CreateJacobianDeterminantImage 3 \\
        {out_field} \\
        {jd_file} 1 0
        """
        commands.append(cmd_inv.strip())

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

    # Step 2: Combine transforms
    if 2 in steps:
        combine_xfm_cmd = f"""
        antsApplyTransforms -d 3 \\
        -r {tpl_fix_T1} \\
        -o [{os.path.join(output_dir, f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_AffWarp.nii.gz")},1] \\
        -t {os.path.join(output_dir, f'T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz')} \\
        -t {os.path.join(output_dir, f'T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat')}
        """
        commands.append(combine_xfm_cmd.strip())

    # Step 3: Jacobian
    if 3 in steps:
        out_field = os.path.join(output_dir, f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_AffWarp.nii.gz")
        jd_file = os.path.join(output_dir, f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_Jacobian.nii.gz")
        cmd = f"""
        CreateJacobianDeterminantImage 3 \\
        {out_field} \\
        {jd_file} 1 0
        """
        commands.append(cmd.strip())

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


    tpl_mov_to_subj_field = os.path.join(
        output_dir,
        f"T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_AffWarp_Inverse.nii.gz"
    )
    subj_to_tpl_fix_field = os.path.join(
        output_dir,
        f"T1T2_Brain_pad_Norm_to_{tpl_fix_month}Mtpl_AffWarp.nii.gz"
    )



    # Step 1: compose transforms
    composed_path = os.path.join(
        output_dir,
        f"Displacement_field_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_viasubj.nii.gz"
    )
    # No indentation for the multi-line string to avoid leading spaces in the command
    cmd_compose = f"""
    antsApplyTransforms -d 3 \\
    -r {tpl_fix_img_path} \\
    -o [{composed_path},1] \\
    -t {subj_to_tpl_fix_field} \\
    -t {tpl_mov_to_subj_field}
    """
    commands.append(cmd_compose.strip())


    # Step 2: Jacobian determinant
    jd_out = composed_path.replace(".nii.gz", "_Jacobian.nii.gz")
    cmd_jd = f"""
    CreateJacobianDeterminantImage 3 \\
    {composed_path} \\
    {jd_out} 1 0
    """
    commands.append(cmd_jd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)

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
    tpl_trans_dir = f"{tpl_root}/neonate_tpl_transfer"

    # Get the stepwise tpl order to concanate for tpls not neighbors
    # For exmple, 00M->02M = 00M->01M + 01M->02M
    tpl_dirs = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
    tpl_months = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)
    month_order = tpl_months
    idx_mov = month_order.index(tpl_mov_month)
    idx_fix = month_order.index(tpl_fix_month)
    requires_chaining = abs(idx_mov - idx_fix) > 1

    output_xfm = os.path.join(tpl_trans_dir, f"Displacement_field_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_by_{transf_type}.nii.gz")
    tpl_warped_path = os.path.join(tpl_trans_dir, f"Tpl_{tpl_mov_month}Mtpl_warp_to_{tpl_fix_month}Mtpl_by_{transf_type}.nii.gz")

    commands = []
    if requires_chaining:
        path_months = month_order[idx_mov:idx_fix + 1]

        xfm_chain = []
        for k in range(len(path_months) - 1):
            m_from = path_months[k]
            m_to = path_months[k + 1]
            if transf_type=='directtpl':
                xfm_path = os.path.join(tpl_trans_dir,f"T1T2_{m_from}Mtpl_Norm_to_{m_to}Mtpl_AffWarp.nii.gz")
            elif transf_type=='averagesubj':
                xfm_path = os.path.join(tpl_trans_dir,f"Avg_Displacement_field_{m_from}Mtpl_to_{m_to}Mtpl_viasubj.nii.gz")
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
        if transf_type=='directtpl':
            xfm_path = os.path.join(tpl_trans_dir,f"T1T2_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_AffWarp.nii.gz")
        elif transf_type=='averagesubj':
            xfm_path = os.path.join(tpl_trans_dir,f"Avg_Displacement_field_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_viasubj.nii.gz")
        # copy xfm_path to output_xfm
        copy_cmd = f"cp {xfm_path} {output_xfm}"
        commands.append(copy_cmd.strip())    
        reslice_cmd = f"""
            antsApplyTransforms \\
            -d 3 \\
            -i {tpl_mov_img} \\
            -r {tpl_fix_img} \\
            -o {tpl_warped_path} \\
            -t {output_xfm} 
            """
        commands.append(reslice_cmd.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    print(full_cmd)

    # Submit
    log_dir = os.path.join(tpl_root, "neonate_tpl_transfer/log")
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
    tpl_trans_dir = f"{tpl_root}/neonate_tpl_transfer"
    # Get the stepwise tpl order to concanate for tpls not neighbors
    # For exmple, 00M->02M = 00M->01M + 01M->02M
    tpl_dirs = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
    tpl_months = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)
    month_order = tpl_months
    idx_mov = month_order.index(tpl_mov_month)
    idx_fix = month_order.index(tpl_fix_month)

    commands = []

    for subid in sub_list:
        if "Example" in data_dir:
            output_dir = os.path.join(data_dir, subid)
        else:
            output_dir = os.path.join(data_dir, subid, "WB", "T1_T2_neonate_tpl_transfer")
        output_xfm = os.path.join(output_dir, f"Displacement_T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type}.nii.gz")
        jd_file = os.path.join(output_dir, f"Displacement_T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type}_Jacobian.nii.gz")
        out_warped = os.path.join(output_dir, f"T1_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type}.nii.gz")
        
        # Step 1: Combine transforms
        # Determine template warp path
        path_months = month_order[idx_mov:idx_fix + 1]

        xfm_chain = []
        # First: subject ➝ mov template
        subject_to_mov = os.path.join(
            output_dir,
            f"T1T2_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_AffWarp.nii.gz"
        )
        xfm_chain.append(f"-t {subject_to_mov}")

        # Then: mov ➝ fix through intermediate template warps
        for k in range(len(path_months) - 1):
            m_from = path_months[k]
            m_to = path_months[k + 1]
            if transf_type == 'directtpl':
                xfm_path = os.path.join(
                    tpl_trans_dir, f"T1T2_{m_from}Mtpl_Norm_to_{m_to}Mtpl_AffWarp.nii.gz"
                )
            elif transf_type == 'averagesubj':
                xfm_path = os.path.join(
                    tpl_trans_dir, f"Avg_Displacement_field_{m_from}Mtpl_to_{m_to}Mtpl_viasubj.nii.gz"
                )
            elif transf_type == 'averagesubj_train':
                xfm_path = os.path.join(
                    tpl_trans_dir, 'halfsplit', f"Avg_Displacement_field_{m_from}Mtpl_to_{m_to}Mtpl_viasubj_halfsplit_train.nii.gz"
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
    log_dir = os.path.join(tpl_trans_dir, "log")
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

    subprocess.run(cmd_sbatch)

    return log_name_full, job_script
    



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

