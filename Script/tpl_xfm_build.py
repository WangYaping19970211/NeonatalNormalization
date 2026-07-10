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

# ===== Default configuration depends on your server =====
DEFAULT_ANTSPATH = "/hpf/largeprojects/smiller/tools/rcp_pipeline/conda_environments/rcp_env" # TODO: [User] Set this to the path where ANTs is installed on your machine
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





def com_initialize_to_template_space(
    t1_img_path: str,
    fix_img_path: str,
    t1_output_path: str,
    t2_img_path: str = None,
    t2_output_path: str = None,
    mask_img_path: str = None,
    mask_output_path: str = None,
):
    """
    Coarse initialization of images into a template-aligned space using a T1-driven voxel shift.

    This function performs a lightweight spatial normalization step that places input images
    (T1, optional T2, optional mask) into a common field-of-view aligned with a template.
    It is designed as a robust preprocessing step prior to rigid or nonlinear registration.

    Key characteristics:
    --------------------
    - No resampling or interpolation is performed.
      All operations are based on integer voxel shifts and direct data placement,
      preserving the original image intensities.

    - T1-driven transformation.
      The center-of-mass (COM) of the T1 image is used to compute a voxel shift that
      moves the brain approximately to the center of a template-sized grid.
      The same voxel shift is applied to all additional inputs (T2, mask).

    - Template-aligned grid construction.
      A target grid (canvas) is defined based on the template world size and T1 spacing.
      All images are embedded into this grid after shifting.

    - Affine reconstruction.
      A new affine is constructed such that the center voxel of the target grid corresponds
      to the center of the template in world coordinates.
      If additional images (e.g., T2) have different voxel spacing, their affine is rebuilt
      using their own spacing while preserving the same orientation and center alignment.

    What this step achieves:
    ------------------------
    - Roughly centers the subject brain in template space
    - Ensures consistent orientation across modalities
    - Places all modalities into a shared field-of-view
    - Provides a stable initialization for subsequent registration (e.g., rigid, affine, SyN)

    Important notes:
    ----------------
    - This is NOT a full registration step. Anatomical alignment to the template is not achieved here.
    - When voxel spacing differs between modalities, the shared voxel shift may introduce a small
      mismatch in world coordinates; however, this is acceptable for same-subject data and is typically
      corrected by subsequent rigid alignment.
    - The accuracy of centering depends on the COM estimation, which is computed from non-zero voxels.

    In summary:
    -----------
    This function performs a T1-guided, voxel-based initialization that aligns images into a
    template-oriented space without modifying image intensities, serving as a reliable starting
    point for downstream multi-modal registration.
    """
    
    # ---------- Load & reorient ----------
    fix_ants = ants.image_read(fix_img_path)
    fix_dir = ants.get_orientation(fix_ants)

    def load_and_reorient(path, name):
        img_ants = ants.image_read(path)
        if ants.get_orientation(img_ants) != fix_dir:
            # print(f"{name} reorient")
            img_ants = ants.reorient_image2(img_ants, orientation=fix_dir)
            return ants_to_nib(img_ants)
        return nib.load(path)

    t1_img = load_and_reorient(t1_img_path, "T1")
    fix_img = nib.load(fix_img_path)

    if t2_img_path:
        t2_img = load_and_reorient(t2_img_path, "T2")
    if mask_img_path:
        mask_img = load_and_reorient(mask_img_path, "MASK")

    # ---------- Extract ----------
    t1_data = t1_img.get_fdata()
    t1_affine = t1_img.affine.copy()

    fix_affine = fix_img.affine.copy()
    fix_shape = np.array(fix_img.shape[:3])
    fix_spacing = np.linalg.norm(fix_affine[:3, :3], axis=0)

    # ---------- Target grid ----------
    t1_spacing = np.linalg.norm(t1_affine[:3, :3], axis=0)
    fix_world_size = fix_shape * fix_spacing
    target_shape = np.round(fix_world_size / t1_spacing).astype(int)
    target_center_vox = (target_shape - 1) / 2.0

    # print("Target shape:", target_shape)

    # ---------- T1 COM ----------
    coords = np.array(np.nonzero(t1_data))
    t1_com_vox = coords.mean(axis=1)

    shift_vox = np.round(target_center_vox - t1_com_vox).astype(int)

    # print("T1 COM:", t1_com_vox)
    # print("shift_vox:", shift_vox)

    # ---------- Canvas placement ----------
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

            length = min(shape[d] - src_start, target_shape[d] - dst_start)

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

    # ---------- Shared affine base ----------
    fix_center_vox = (fix_shape - 1) / 2.0
    fix_center_world = fix_affine[:3, :3] @ fix_center_vox + fix_affine[:3, 3]
    fix_dir_norm = fix_affine[:3, :3] / np.linalg.norm(fix_affine[:3, :3], axis=0)

    def build_affine(spacing):
        A = np.eye(4)
        A[:3, :3] = fix_dir_norm * spacing
        new_center_voxel = (target_shape - 1) / 2.0
        A[:3, 3] = fix_center_world - A[:3, :3] @ new_center_voxel
        return A

    # ---------- Process function ----------
    def process(img, name, output_path, ref_spacing):
        data = img.get_fdata()
        affine = img.affine.copy()
        shape = np.array(data.shape)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)

        canvas = place_into_canvas(data, shape)

        # 🔥 
        if np.allclose(spacing, ref_spacing, atol=1e-3):
            # print(f"{name}: use shared affine")
            new_affine = build_affine(ref_spacing)
        else:
            # print(f"{name}: use own affine (spacing mismatch)")
            new_affine = build_affine(spacing)

        out = nib.Nifti1Image(canvas, new_affine, header=img.header.copy())
        out.set_qform(new_affine, code=1)
        out.set_sform(new_affine, code=1)
        nib.save(out, output_path)

        # print(f"{name} saved → {output_path}")

    # ---------- Run ----------
    process(t1_img, "T1", t1_output_path, t1_spacing)

    if t2_img_path and t2_output_path:
        process(t2_img, "T2", t2_output_path, t1_spacing)

    if mask_img_path and mask_output_path:
        process(mask_img, "MASK", mask_output_path, t1_spacing)

def com_initialize_pair_to_template_space_old(
    t1_img_path: str,
    t2_img_path: str,
    fix_img_path: str,
    t1_output_path: str,
    t2_output_path: str,
):
    """
    Initialize paired T1 and T2 images into a common template-aligned space without resampling.

    A center-of-mass (COM)-based translation derived from the T1 image is applied to both modalities,
    ensuring consistent spatial alignment. The images are reoriented to match the template orientation,
    embedded into a shared voxel grid, and assigned a common affine aligned with the template space
    (in orientation and center).

    This procedure does not involve interpolation or deformation and preserves the original image intensities.
    It provides a robust and consistent initialization for subsequent multi-modal registration.
    """

    # Step 0: Reorient both images to template orientation if needed
    t1_ants = ants.image_read(t1_img_path)
    t2_ants = ants.image_read(t2_img_path)
    fix_ants = ants.image_read(fix_img_path)

    fix_dir = ants.get_orientation(fix_ants)
    t1_dir = ants.get_orientation(t1_ants)
    t2_dir = ants.get_orientation(t2_ants)

    if t1_dir != fix_dir:
        print(f"T1 reorient: {t1_dir} -> {fix_dir}")
        t1_ants = ants.reorient_image2(t1_ants, orientation=fix_dir)
        t1_img = ants_to_nib(t1_ants)
    else:
        t1_img = nib.load(t1_img_path)
    if t2_dir != fix_dir:
        print(f"T2 reorient: {t2_dir} -> {fix_dir}")
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
    print("T2 shape:", t2_shape, "spacing:", t2_spacing)
    # Warning if T1 and T2 affine different
    if not np.allclose(t1_spacing, t2_spacing, atol=1e-3):
        print("[ERROR] Spacing mismatch → rigid will NOT fix this")
    elif not np.allclose(t1_affine, t2_affine, atol=1e-2):
        print("[INFO] Affine differs (rigid mismatch) → will be corrected by T1-T2 rigid registration later")
    
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
    print(f"T1 -> {t1_output_path}")
    print(f"T2 -> {t2_output_path}")




def t1_n4_SyN_pipeline(t1_file, tpl_root, tpl_month, output_dir, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])

    # tpl
    tpl = os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T1.nii.gz')
    fix_mask = tpl_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')
    mask_param = fix_mask

    # Generate commands
    commands = []

    # Step 1: Bias correction
    if 1 in steps:
        cmd = f"""
            N4BiasFieldCorrection -d 3 \\
            -i {t1_file} \\
            -o {t1_file.replace('.nii.gz','_N4.nii.gz')} \\
            """
        commands.append(cmd.strip())
    
    # Step 2: Registration
    if 2 in steps:
        out_prefix = t1_file.replace('.nii.gz','_N4_Norm_to_') + f'{tpl_month}Mtpl_'
        cmd = f"""
            antsRegistrationSyN.sh -d 3 \\
            -f {tpl} \\
            -m {t1_file.replace('.nii.gz','_N4.nii.gz')} \\
            -o {out_prefix} \\
            -x {mask_param} \\
            -n {num_threads} 
            """
        commands.append(cmd.strip())

    # Step 3: Combine transforms
    if 3 in steps:
        xfm_chain = []
        aff_path = t1_file.replace('.nii.gz','_N4_Norm_to_') + f'{tpl_month}Mtpl_0GenericAffine.mat'
        xfm_chain.append(f"-t {aff_path}")
        warp_path = t1_file.replace('.nii.gz','_N4_Norm_to_') + f'{tpl_month}Mtpl_1Warp.nii.gz'
        xfm_chain.append(f"-t {warp_path}")
        # Reverse for antsApplyTransforms (last → first)
        xfm_chain.reverse()
        # Format lines: add "\" to all except last
        xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
        xfm_lines.append(xfm_chain[-1])
        xfm_chain_str = "\n    ".join(xfm_lines)  
        cmd = f"""
            antsApplyTransforms -d 3 \\
            -r {tpl} \\
            -o [{t1_file.replace('.nii.gz','_N4_Norm_to_') + f'{tpl_month}Mtpl_AffWarp.nii.gz'},1] \\
            {xfm_chain_str}
            """
        commands.append(cmd.strip())
    if 3 in steps:
        out_field = t1_file.replace('.nii.gz','_N4_Norm_to_') + f'{tpl_month}Mtpl_AffWarp.nii.gz'
        jd_file = t1_file.replace('.nii.gz','_N4_Norm_to_') + f'{tpl_month}Mtpl_log_geometric_JD.nii.gz'
        cmd = f"""
        CreateJacobianDeterminantImage 3 \\
        {out_field} \\
        {jd_file} 1 1
        """
        commands.append(cmd.strip())


        # Combine all commands
        full_cmd = "\n\n".join(commands)
        # print(full_cmd)

        # Submit
        log_dir = os.path.join(output_dir, "log")
        job_prefix = f"t1_{tpl_month}Mtpl"
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


def skullstripe_and_N4(input_files, tpl_files, output_dir, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    
    
    # Generate commands
    commands = []
    
    # Step 1: Re-orientation
    if 1 in steps:
        for modality in ['T1', 'T2']:
            brain_extraction_cmd = f"""
                fslreorient2std {input_files[modality]} \\
                {input_files[modality].replace('.nii.gz', '_reoriented.nii.gz')}
                """
            commands.append(brain_extraction_cmd.strip())
        input_files['T1'] = input_files['T1'].replace('.nii.gz', '_reoriented.nii.gz')
        input_files['T2'] = input_files['T2'].replace('.nii.gz', '_reoriented.nii.gz')
        


    # Step 2: Brain Extraction
    if 2 in steps:
        for modality in ['T1', 'T2']:
            brain_extraction_cmd = f"""
                antsBrainExtraction.sh -d 3 -k 1 \\
                -a {input_files[modality]} \\
                -e {tpl_files[modality]} \\
                -m {tpl_files['Mask']} \\
                -o {os.path.join(output_dir, f'{modality}_')}
                """
            commands.append(brain_extraction_cmd.strip())
            # Then remove the intermediate files except the final warped output
            cmd_cleanup = f"""
                find {output_dir} -maxdepth 1 -type f -name "{modality}_BrainExtraction*" \
                ! -name "{modality}_BrainExtractionBrain.nii.gz" -delete

                rm -f {output_dir}/{modality}_N4Truncated0.nii.gz \
                    {output_dir}/{modality}_N4Corrected0.nii.gz
                """
            commands.append(cmd_cleanup.strip())
    
    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    # get subid from output_dir
    subid = os.path.basename(output_dir).replace("sub-", "")
    job_prefix = f"ss_{subid}"
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


def t1_t2_rigid_and_N4(input_files, output_dir, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    

    brain_img_T1 = input_files['T1']
    brain_img_T2 = input_files['T2']
    
    # Generate commands
    commands = []
    
    # Step 1: Rigid align T2 to T1
    if 1 in steps:
        out_prefix = brain_img_T2.replace('.nii.gz','_rigid2T1_')
        cmd = f"""
            antsRegistrationSyN.sh -d 3 \\
            -f {brain_img_T1} \\
            -m {brain_img_T2} \\
            -o {out_prefix} \\
            -t r \\
            -n {num_threads} 
            """
        commands.append(cmd.strip())
        # Rename the final warped output to a consistent name for the next steps
        cmd_rename = f"""
        cp -r {out_prefix}Warped.nii.gz {brain_img_T2.replace('.nii.gz','_rigid2T1.nii.gz')}
        """
        commands.append(cmd_rename.strip())
        # Then remove the intermediate files except the final warped output
        cmd_cleanup = f"""
        rm -f {out_prefix}0GenericAffine.mat {out_prefix}InverseWarped.nii.gz {out_prefix}Warped.nii.gz
        """
        commands.append(cmd_cleanup.strip())
    brain_img_T2 = brain_img_T2.replace('.nii.gz','_rigid2T1.nii.gz')  
    
    # Step 2: Bias correction (N4) on both T1 and T2
    if 2 in steps:
        cmd = f"""
        N4BiasFieldCorrection -d 3 \\
        -i {brain_img_T1} \\
        -o {brain_img_T1.replace('.nii.gz','_N4.nii.gz')} 
        """
        commands.append(cmd.strip())
        cmd = f"""
        N4BiasFieldCorrection -d 3 \\
        -i {brain_img_T2} \\
        -o {brain_img_T2.replace('.nii.gz','_N4.nii.gz')} 
        """
        commands.append(cmd.strip())
    
    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    # get subid from output_dir
    subid = os.path.basename(output_dir).replace("sub-", "")
    job_prefix = f"rigN4_{subid}"
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


def N4_bias_correction(input_file, output_file, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)

    # Generate commands
    commands = []
    cmd = f"""
    N4BiasFieldCorrection -d 3 \\
    -i {input_file} \\
    -o {output_file}
    """
    commands.append(cmd.strip())
    
    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    output_dir = os.path.dirname(output_file)
    log_dir = os.path.join(output_dir, "log")
    # get subid from output_dir
    fname = os.path.basename(output_file)
    m = re.search(r'(sub-[A-Za-z0-9]+|bc\d+)', fname)
    subid = m.group(0).replace("sub-", "") if m else "unknown"
    job_prefix = f"N4_{subid}"
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

def multimodal_register_pipeline(modalities, input_files, tpl_root, tpl_month,  output_dir, mov_mask=False, **kwargs):
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
            - 2: Apply transforms to reslice T1 into template space
            - 3: Apply transforms to reslice T2 into template space
            - 4: Combine transform fields and save inverse fields
            - 5: Compute Jacobian determinant maps from fields
    """

    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3, 4])
    

    # tpl
    register_tpls = {
            "T1": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T1.nii.gz'),
            "T2": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T2.nii.gz')
        }
   


    # Generate commands
    commands = []
    
    
    # Step 1: Registration
    if 1 in steps:
        fix_mask = register_tpls['T1'].replace(f'-T1.nii.gz', '-Mask.nii.gz')
        if mov_mask:
            # print("[INFO] Using moving mask for registration")
            t1_mask = input_files['T1'].replace('.nii.gz', '_mask.nii.gz')
            cmd = f"""
            ThresholdImage 3 {input_files['T1']} {t1_mask} 0.01 Inf
            """
            commands.append(cmd.strip())
            moving_mask = t1_mask  
            if modalities == "T1T2":
                mask_param = f"[{fix_mask},{moving_mask}]"   # antsRegistration format
            else:
                mask_param = f"{fix_mask},{moving_mask}"     # antsRegistrationSyN.sh format
        else:
            # print("[INFO] No moving mask used")
            mask_param = fix_mask

        out_prefix = os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_')
        if modalities == "T1T2":
            # print(f"{modalities} joint registration ...")
            cmd = f"""
                antsRegistration --verbose 1 -d 3 \\
                --float 0 -z 1 -u 0 --winsorize-image-intensities [0.005,0.995] \\
                -o {out_prefix} \\
                -x {mask_param} \\
                -t Rigid[0.1] \\
                -m MI[{register_tpls['T1']}, {input_files['T1']}, 1, 32, Regular, 0.25] \\
                -m MI[{register_tpls['T2']}, {input_files['T2']}, 1, 32, Regular, 0.25] \\
                -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
                -t Affine[0.1] \\
                -m MI[{register_tpls['T1']}, {input_files['T1']}, 1, 32, Regular, 0.25] \\
                -m MI[{register_tpls['T2']}, {input_files['T2']}, 1, 32, Regular, 0.25] \\
                -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
                -t SyN[0.2, 3, 0] \\
                -m CC[{register_tpls['T1']}, {input_files['T1']}, 1, 2] \\
                -m CC[{register_tpls['T2']}, {input_files['T2']}, 1, 2] \\
                -c [100x100x70x50x20,1e-6,10] -s 5x3x2x1x0vox -f 10x6x4x2x1 \\
                """
        else:
            # print(f"{modalities} only registration ...")
            cmd = f"""
                antsRegistrationSyN.sh -d 3 \\
                -f {register_tpls[modalities]} \\
                -m {input_files[modalities]} \\
                -o {out_prefix} \\
                -x {mask_param} \\
                -n {num_threads} 
                """

        commands.append(cmd.strip())
    
    # Orgnize the xfm
    xfm_chain = []
    out_prefix = os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_')
    aff_path  = f"{out_prefix}0GenericAffine.mat"
    warp_path = f"{out_prefix}1Warp.nii.gz"
    xfm_chain.append(f"-t {aff_path}")
    xfm_chain.append(f"-t {warp_path}")
    # Reverse for antsApplyTransforms (last → first)
    xfm_chain.reverse()
    # Format lines: add "\" to all except last
    xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
    xfm_lines.append(xfm_chain[-1])
    xfm_chain_str = "\n    ".join(xfm_lines) 

    # Step 2: Reslice
    if 2 in steps:
        if modalities == "T1T2":
            # print(f"{modalities} joint reslicing ...")
            cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {input_files['T1']} \\
            -r {register_tpls['T1']} \\
            -o {os.path.join(output_dir, f"T1_resliced_to_{tpl_month}Mtpl_by_direct_{modalities}_xfm.nii.gz")} \\
            {xfm_chain_str}
            """
            commands.append(cmd.strip())

            cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {input_files['T2']} \\
            -r {register_tpls['T2']} \\
            -o {os.path.join(output_dir, f"T2_resliced_to_{tpl_month}Mtpl_by_direct_{modalities}_xfm.nii.gz")} \\
            {xfm_chain_str}
            """
            commands.append(cmd.strip())
        else:
            # print(f"{modalities} only reslicing ...")
            cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {input_files[modalities]} \\
            -r {register_tpls[modalities]} \\
            -o {os.path.join(output_dir, f"{modalities}_resliced_to_{tpl_month}Mtpl_by_direct_{modalities}_xfm.nii.gz")} \\
            {xfm_chain_str}
            """
            commands.append(cmd.strip())

    # Step 3: Combine transforms
    if 3 in steps:
        combine_xfm_cmd = f"""
        antsApplyTransforms -d 3 \\
        -r {register_tpls[modalities] if modalities != "T1T2" else register_tpls['T1']} \\
        -o [{os.path.join(output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")},1] \\
        {xfm_chain_str}
        """
        commands.append(combine_xfm_cmd.strip())

    # Step 4: Jacobian
        # Usage:
            # CreateJacobianDeterminantImage
            # imageDimension
            # deformationField
            # outputImage
            # [doLogJacobian=0]
            # [useGeometric=0]
            # [deformationGradient=0]
    if 4 in steps:
        out_field = os.path.join(output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")
        jd_file = os.path.join(output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_log_geometric_JD.nii.gz")
        cmd = f"""
        CreateJacobianDeterminantImage 3 \\
        {out_field} \\
        {jd_file} 1 1
        """
        commands.append(cmd.strip())
        # Remove the AffWarp intermediate file to save space
        cmd_cleanup = f"""
        rm -f {os.path.join(output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")}
        """
        commands.append(cmd_cleanup.strip())

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"{modalities}_{tpl_month}Mtpl"
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



def multimodal_register_pipeline_T1T2(modalities, input_files, tpl_root, tpl_month,  output_dir, mov_mask=False, **kwargs):
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
    brain_img_T1 = input_files['T1']
    brain_img_T2 = input_files['T2']
    tpl_T1 = register_tpls['T1']
    tpl_T2 = register_tpls['T2']
    # Generate commands
    commands = []
    
    
    # Step 1: Registration
    if 1 in steps:
        fix_mask = tpl_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')
        if mov_mask:
            print("[INFO] Using moving mask for registration")
            t1_mask = brain_img_T1.replace('.nii.gz', '_mask.nii.gz')
            cmd = f"""
            ThresholdImage 3 {brain_img_T1} {t1_mask} 0.01 Inf
            """
            commands.append(cmd.strip())
            moving_mask = t1_mask  
            mask_param = f"[{fix_mask},{moving_mask}]"
        else:
            print("[INFO] No moving mask used")
            mask_param = fix_mask
        
        out_prefix = os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_')
        
        cmd = f"""
        antsRegistration --verbose 1 -d 3 \\
        --float 0 -z 1 -u 0 --winsorize-image-intensities [0.005,0.995] \\
        -o {out_prefix} \\
        -x {mask_param} \\
        -t Rigid[0.1] \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
        -t Affine[0.1] \\
        -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
        -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
        -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
        -t SyN[0.2, 3, 0] \\
        -m CC[{tpl_T1}, {brain_img_T1}, 1, 2] \\
        -m CC[{tpl_T2}, {brain_img_T2}, 1, 2] \\
        -c [100x100x70x50x20,1e-6,10] -s 5x3x2x1x0vox -f 10x6x4x2x1 \\
        """
        commands.append(cmd.strip())
    
    # Orgnize the xfm
    xfm_chain = []
    out_prefix = os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_')
    aff_path = os.path.join(output_dir, f"{out_prefix}0GenericAffine.mat")
    xfm_chain.append(f"-t {aff_path}")
    warp_path = os.path.join(output_dir, f"{out_prefix}1Warp.nii.gz")
    xfm_chain.append(f"-t {warp_path}")
    # Reverse for antsApplyTransforms (last → first)
    xfm_chain.reverse()
    # Format lines: add "\" to all except last
    xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
    xfm_lines.append(xfm_chain[-1])
    xfm_chain_str = "\n    ".join(xfm_lines) 

    # Step 2: Reslice T1
    if 2 in steps:
        cmd = f"""
        antsApplyTransforms -d 3 \\
        -i {brain_img_T1} \\
        -r {tpl_T1} \\
        -o {os.path.join(output_dir, f"T1_resliced_to_{tpl_month}Mtpl_by_direct_{modalities}_xfm.nii.gz")} \\
        {xfm_chain_str}
        """
        commands.append(cmd.strip())
    # Step 3: Combine transforms
    if 3 in steps:
        tpl_T1 = register_tpls['T1']
        combine_xfm_cmd = f"""
        antsApplyTransforms -d 3 \\
        -r {tpl_T1} \\
        -o [{os.path.join(output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")},1] \\
        {xfm_chain_str}
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

    # Step 4: Jacobian
        # Usage:
            # CreateJacobianDeterminantImage
            # imageDimension
            # deformationField
            # outputImage
            # [doLogJacobian=0]
            # [useGeometric=0]
            # [deformationGradient=0]
    if 4 in steps:
        out_field = os.path.join(output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_AffWarp.nii.gz")
        jd_file = os.path.join(output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_month}Mtpl_log_geometric_JD.nii.gz")
        cmd = f"""
        CreateJacobianDeterminantImage 3 \\
        {out_field} \\
        {jd_file} 1 1
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


def n4bias_pipeline(subid, output_dir, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    

    commands = []
    brain_img_T1 = f"{output_dir}/T1_Brain_pad.nii.gz"
    cmd = f"""
    N4BiasFieldCorrection -d 3 \\
    -i {brain_img_T1} \\
    -o {brain_img_T1.replace('.nii.gz','_N4.nii.gz')} \\
    """
    commands.append(cmd.strip())

    brain_img_T2 = f"{output_dir}/T2_Brain_pad_rigid2T1.nii.gz"
    cmd = f"""
    N4BiasFieldCorrection -d 3 \\
    -i {brain_img_T2} \\
    -o {brain_img_T2.replace('.nii.gz','_N4.nii.gz')} \\
    """
    commands.append(cmd.strip())
    
    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"n4_{subid.replace('sub-','')}"
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



def t1_t2_rigid(subid, output_dir, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    
    commands = []
    fix_T1 = f"{output_dir}/T1_Brain_pad.nii.gz"
    mov_T2 = f"{output_dir}/T2_Brain_pad.nii.gz"
    out_prefix = os.path.join(output_dir, f'T2_Brain_pad_rigid2T1_')
    cmd = f"""
        antsRegistrationSyN.sh -d 3 \\
        -f {fix_T1} \\
        -m {mov_T2} \\
        -o {out_prefix} \\
        -t r \\
        -n {num_threads} 
        """
    commands.append(cmd.strip())
    # Then remove the intermediate files except the final warped output
    cmd_cleanup = f"""
    rm -f {out_prefix}0GenericAffine.mat {out_prefix}InverseWarped.nii.gz
    """
    commands.append(cmd_cleanup.strip())
    # # rename the warped image to the expected name for the next step
    # cmd_rename = f"""
    # mv {out_prefix}Warped.nii.gz {mov_T2}
    # """
    # commands.append(cmd_rename.strip())
    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"rig_{subid.replace('sub-','')}"
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


def unimodal_register_pipeline(xfm_keys, input_files, tpl_root, tpl_month, output_dir, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    steps = kwargs.get('steps', [1, 2, 3])
    

    # tpl
    register_tpls = {
            "T1": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T1.nii.gz'),
            "T2": os.path.join(tpl_root, f'{tpl_month}Month/BCP-{tpl_month}M-T2.nii.gz')
        }
    tpl_T1 = register_tpls['T1']
    tpl_T2 = register_tpls['T2']
    fix_mask = tpl_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')
    print("[INFO] No moving mask used")
    mask_param = fix_mask
    # Step 1: Registration
    for xfm_key in xfm_keys:
        # Generate commands
        commands = []
        out_prefix = os.path.join(output_dir, f'Brain_pad_{xfm_key}_Norm_to_{tpl_month}Mtpl_')
        # contains "T1T2"
        if "T1T2" in xfm_key:
            print("[INFO] Running multimodal registration for T1 and T2 together")
            if xfm_key == "N4_T1T2":
                brain_img_T1 = input_files['T1'].replace('.nii.gz','_N4.nii.gz')
                brain_img_T2 = input_files['T2'].replace('.nii.gz','_N4.nii.gz')
            else:
                brain_img_T1 = input_files['T1']
                brain_img_T2 = input_files['T2']

            cmd = f"""
            antsRegistration --verbose 1 -d 3 --float 0\\
            -o {out_prefix} \\
            -x {mask_param} \\
            -t Rigid[0.1] \\
            -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
            -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
            -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
            -t Affine[0.1] \\
            -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
            -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
            -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
            -t SyN[0.2, 3, 0] \\
            -m CC[{tpl_T1}, {brain_img_T1}, 1, 2] \\
            -m CC[{tpl_T2}, {brain_img_T2}, 1, 2] \\
            -c [100x100x70x50x20,1e-6,10] -s 5x3x2x1x0vox -f 10x6x4x2x1 \\
            -u 1 -z 1
            """
            commands.append(cmd.strip())
        elif "T12" in xfm_key:
            # Naybe longer time because CC
            print("[INFO] Testing the time longer")
            if xfm_key == "N4_T12":
                brain_img_T1 = input_files['T1'].replace('.nii.gz','_N4.nii.gz')
                brain_img_T2 = input_files['T2'].replace('.nii.gz','_N4.nii.gz')
            else:
                brain_img_T1 = input_files['T1']
                brain_img_T2 = input_files['T2']
            cmd = f"""
            antsRegistration --verbose 1 -d 3 \\
            -o {out_prefix} \\
            -x {mask_param} \\
            -t Rigid[0.1] \\
            -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
            -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
            -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
            -t Affine[0.1] \\
            -m MI[{tpl_T1}, {brain_img_T1}, 1, 32, Regular, 0.25] \\
            -m MI[{tpl_T2}, {brain_img_T2}, 1, 32, Regular, 0.25] \\
            -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
            -t SyN[0.2, 3, 0] \\
            -m CC[{tpl_T1}, {brain_img_T1}, 1, 2] \\
            -m CC[{tpl_T2}, {brain_img_T2}, 1, 2] \\
            -c [100x100x70x50x20,1e-6,10] -s 5x3x2x1x0vox -f 10x6x4x2x1 \\
            -u 1 -z 1
            """
            commands.append(cmd.strip())
        
            # cmd = f"""
            # antsRegistration --verbose 1 -d 3 \\
            # -o {out_prefix} \\
            # -x {mask_param} \\
            # -t Rigid[0.1] \\
            # -m MI[{tpl_T1}, {brain_img_T1.replace('.nii.gz','_N4.nii.gz')}, 1, 32, Regular, 0.25] \\
            # -m MI[{tpl_T2}, {brain_img_T2.replace('.nii.gz','_N4.nii.gz')}, 1, 32, Regular, 0.25] \\
            # -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
            # -t Affine[0.1] \\
            # -m MI[{tpl_T1}, {brain_img_T1.replace('.nii.gz','_N4.nii.gz')}, 1, 32, Regular, 0.25] \\
            # -m MI[{tpl_T2}, {brain_img_T2.replace('.nii.gz','_N4.nii.gz')}, 1, 32, Regular, 0.25] \\
            # -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
            # -t SyN[0.1, 3, 0] \\
            # -m MI[{tpl_T1}, {brain_img_T1.replace('.nii.gz','_N4.nii.gz')}, 1, 32, Regular, 0.25] \\
            # -m MI[{tpl_T2}, {brain_img_T2.replace('.nii.gz','_N4.nii.gz')}, 1, 32, Regular, 0.25] \\
            # -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
            # -u 1 -z 1
            # """
            # commands.append(cmd.strip())
            # -t SyN[0.2, 3, 0] \\
            # -c [100x100x70x50x20,1e-6,10] -s 5x3x2x1x0 -f 10x6x4x2x1 \\

        else:
            if "N4" in xfm_key:
                brain_img = input_files[xfm_key.replace('N4_','')].replace('.nii.gz','_N4.nii.gz')
                tpl = register_tpls[xfm_key.replace('N4_','')]
            else:
                brain_img = input_files[xfm_key]
                tpl = register_tpls[xfm_key]
            
            # cmd = f"""
            # antsRegistration --verbose 1 -d 3 \\
            # -o [{out_prefix}, {warped_output}] \\
            # -x {mask_param} \\
            # -m MI[{tpl}, {brain_img}, 1, 32, Regular, 0.25] \\
            # -t Rigid[0.1] \\
            # -c [1000x500x250x100,1e-8,10] -s 4x2x1x0 -f 8x4x2x1 \\
            # -m MI[{tpl}, {brain_img}, 1, 32, Regular, 0.25] \\
            # -t Affine[0.1] \\
            # -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
            # -m MI[{tpl}, {brain_img}, 1, 32, Regular, 0.25] \\
            # -t SyN[0.1, 3, 0] \\
            # -c [100x100x70x20,1e-6,10] -s 4x2x1x0 -f 6x4x2x1 \\
            # -u 1 -z 1
            # """
            # commands.append(cmd.strip())

            # cmd = f"""
            # N4BiasFieldCorrection -d 3 \\
            # -i {brain_img} \\
            # -o {brain_img.replace('.nii.gz','_N4.nii.gz')} \\
            # """
            # commands.append(cmd.strip())

            cmd = f"""
            antsRegistrationSyN.sh -d 3 \\
            -f {tpl} \\
            -m {brain_img} \\
            -o {out_prefix} \\
            -x {mask_param} \\
            -n {num_threads} 
            """
            commands.append(cmd.strip())
    
        xfm_chain = []
        aff_path = os.path.join(output_dir, f"Brain_pad_{xfm_key}_Norm_to_{tpl_month}Mtpl_0GenericAffine.mat")
        xfm_chain.append(f"-t {aff_path}")
        warp_path = os.path.join(output_dir, f"Brain_pad_{xfm_key}_Norm_to_{tpl_month}Mtpl_1Warp.nii.gz")
        xfm_chain.append(f"-t {warp_path}")
        # Reverse for antsApplyTransforms (last → first)
        xfm_chain.reverse()
        # Format lines: add "\" to all except last
        xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
        xfm_lines.append(xfm_chain[-1])
        xfm_chain_str = "\n    ".join(xfm_lines) 
    
        reslice_cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {input_files['T1'].replace('.nii.gz','_N4.nii.gz')} \\
            -r {tpl_T1} \\
            -o {os.path.join(output_dir, f'T1_N4_resliced_to_{tpl_month}Mtpl_by_{xfm_key}_xfm.nii.gz')} \\
            {xfm_chain_str}
            """
        commands.append(reslice_cmd.strip())
        reslice_cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {input_files['T2'].replace('.nii.gz','_N4.nii.gz')} \\
            -r {tpl_T2} \\
            -o {os.path.join(output_dir, f'T2_N4_resliced_to_{tpl_month}Mtpl_by_{xfm_key}_xfm.nii.gz')} \\
            {xfm_chain_str}
            """
        commands.append(reslice_cmd.strip()) 
        reslice_cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {input_files['T1']} \\
            -r {tpl_T1} \\
            -o {os.path.join(output_dir, f'T1_resliced_to_{tpl_month}Mtpl_by_{xfm_key}_xfm.nii.gz')} \\
            {xfm_chain_str}
            """
        commands.append(reslice_cmd.strip())
        reslice_cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {input_files['T2']} \\
            -r {tpl_T2} \\
            -o {os.path.join(output_dir, f'T2_resliced_to_{tpl_month}Mtpl_by_{xfm_key}_xfm.nii.gz')} \\
            {xfm_chain_str}
            """
        commands.append(reslice_cmd.strip())

        # Combine all commands
        full_cmd = "\n\n".join(commands)
        # print(full_cmd)

        # Submit
        log_dir = os.path.join(output_dir, "log")
        job_prefix = f"{xfm_key}_{tpl_month}M" # f"{xfm_key.replace('T','')}_{tpl_month}M"
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


def multimodal_tpl_register(modalities, tpl_root, tpl_mov_month, tpl_fix_month, output_dir,  mov_mask=False, **kwargs):
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
    fix_mask = tpl_fix_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')

    commands = []


    # Step 1: Registration
    if 1 in steps:
        if mov_mask:
            print("[INFO] Using default moving mask")
            moving_mask = tpl_mov_T1.replace(f'-T1.nii.gz', '-Mask.nii.gz')  
            if modalities == "T1T2":
                mask_param = f"[{fix_mask},{moving_mask}]"   # antsRegistration formate
            else:
                mask_param = f"{fix_mask},{moving_mask}"      # antsRegistrationSyN.sh
        else:
            print("[INFO] No moving mask used")
            mask_param = fix_mask

        out_prefix = os.path.join(output_dir, f'{modalities}_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_')
        if modalities == "T1T2":
            print("[INFO] Running multimodal registration for T1 and T2 together")
            cmd = f"""
            antsRegistration --verbose 1 -d 3 \\
            --float 0 -z 1 -u 0 --winsorize-image-intensities [0.005,0.995] \\
            -o {out_prefix} \\
            -x {mask_param} \\
            -t Rigid[0.1] \\
            -m MI[{tpl_fix_T1}, {tpl_mov_T1}, 1, 32, Regular, 0.25] \\
            -m MI[{tpl_fix_T2}, {tpl_mov_T2}, 1, 32, Regular, 0.25] \\
            -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
            -t Affine[0.1] \\
            -m MI[{tpl_fix_T1}, {tpl_mov_T1}, 1, 32, Regular, 0.25] \\
            -m MI[{tpl_fix_T2}, {tpl_mov_T2}, 1, 32, Regular, 0.25] \\
            -c [1000x500x250x100,1e-6,10] -s 4x3x2x1vox -f 12x8x4x2 \\
            -t SyN[0.2, 3, 0] \\
            -m CC[{tpl_fix_T1}, {tpl_mov_T1}, 1, 2] \\
            -m CC[{tpl_fix_T2}, {tpl_mov_T2}, 1, 2] \\
            -c [100x100x70x50x20,1e-6,10] -s 5x3x2x1x0vox -f 10x6x4x2x1 \\
            """
        else:
            print(f"[INFO] Running registration for {modalities} only")
            cmd = f"""
            antsRegistrationSyN.sh -d 3 \\
            -f {tpl_fix[modalities]} \\
            -m {tpl_mov[modalities]} \\
            -o {out_prefix} \\
            -x {mask_param} \\
            -n {num_threads}
            """

        commands.append(cmd.strip())

    # Orgnize the xfm
    xfm_chain = []
    out_prefix = os.path.join(output_dir, f'{modalities}_{tpl_mov_month}Mtpl_Norm_to_{tpl_fix_month}Mtpl_')
    aff_path  = f"{out_prefix}0GenericAffine.mat"
    warp_path = f"{out_prefix}1Warp.nii.gz"
    xfm_chain.append(f"-t {aff_path}")
    xfm_chain.append(f"-t {warp_path}")
    # Reverse for antsApplyTransforms (last → first)
    xfm_chain.reverse()
    # Format lines: add "\" to all except last
    xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
    xfm_lines.append(xfm_chain[-1])
    xfm_chain_str = "\n    ".join(xfm_lines) 

    # Step 2: Reslice 
    if 2 in steps:
        if modalities == "T1T2":
            print("[INFO] Reslicing both T1 and T2 using the same transform")
            cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {tpl_mov['T1']} \\
            -r {tpl_fix['T1']} \\
            -o {os.path.join(output_dir, f"T1_{tpl_mov_month}Mtpl_resliced_to_{tpl_fix_month}Mtpl_by_direct_{modalities}_xfm.nii.gz")} \\
            {xfm_chain_str}
            """
            commands.append(cmd.strip())

            cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {tpl_mov['T2']} \\
            -r {tpl_fix['T2']} \\
            -o {os.path.join(output_dir, f"T2_{tpl_mov_month}Mtpl_resliced_to_{tpl_fix_month}Mtpl_by_direct_{modalities}_xfm.nii.gz")} \\
            {xfm_chain_str}
            """
            commands.append(cmd.strip())
        else:
            print(f"[INFO] Reslicing {modalities} using the transform from {modalities} registration")
            cmd = f"""
            antsApplyTransforms -d 3 \\
            -i {tpl_mov[modalities]} \\
            -r {tpl_fix[modalities]} \\
            -o {os.path.join(output_dir, f"{modalities}_{tpl_mov_month}Mtpl_resliced_to_{tpl_fix_month}Mtpl_by_direct_{modalities}_xfm.nii.gz")} \\
            {xfm_chain_str}
            """
            commands.append(cmd.strip())
    

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    # print(full_cmd)

    # Submit
    log_dir = os.path.join(output_dir, "log")
    job_prefix = f"{modalities}_{tpl_mov_month}Mtpl_{tpl_fix_month}Mtpl"
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
        submit_bash_job(full_cmd, job_script, f"{modalities}_{tpl_mov_month}_{tpl_fix_month}", output_log, error_log, num_threads, verbose)

    return True



def generate_xfm_between_tpl_viasubj(tpl_mov_month, tpl_fix_month, output_dir, tpl_root, modalities_dict=None, **kwargs):  
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
    
    if modalities_dict is None:
        modalities_dict = ['T1', 'T2', 'T1T2']

    commands = []
    for modalities in modalities_dict:
            
        print(f"{modalities} registering from month {tpl_mov_month} ➜ {tpl_fix_month}...")
        ref_mod = 'T1' if modalities == 'T1T2' else modalities
        
        tpl_fix_img_path = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-{ref_mod}.nii.gz'
        xfm_chain = []
        
        tpl_mov_to_subj_warp = os.path.join(
            output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_1InverseWarp.nii.gz")
        xfm_chain.append(f"-t {tpl_mov_to_subj_warp}")
        tpl_mov_to_subj_aff = os.path.join(
            output_dir, f"{modalities}_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_0GenericAffine.mat")
        xfm_chain.append(f"-t [{tpl_mov_to_subj_aff},1]")
        
        subj_to_tpl_fix_aff = os.path.join(
            output_dir,f"{modalities}_Brain_pad_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat")
        xfm_chain.append(f"-t {subj_to_tpl_fix_aff}")
        subj_to_tpl_fix_warp = os.path.join(
            output_dir,f"{modalities}_Brain_pad_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz")
        xfm_chain.append(f"-t {subj_to_tpl_fix_warp}")
        xfm_chain.reverse()
        # Format lines: add "\" to all except last
        xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
        xfm_lines.append(xfm_chain[-1])
        xfm_chain_str = "\n    ".join(xfm_lines)  

        # Step 1: compose transforms
        composed_path = os.path.join(
            output_dir,
            f"Displacement_{tpl_mov_month}Mtpl_to_{tpl_fix_month}Mtpl_viasubj_by_{modalities}_xfm.nii.gz"
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





def tpl_concate_resli(transf_type, tpl_mov_month, tpl_fix_month, tpl_root, modalities_dict=None, **kwargs):
    num_threads = kwargs.get('num_threads', 6)
    slurm       = kwargs.get('slurm', True)
    verbose     = kwargs.get('verbose', True)
    if modalities_dict is None:
        modalities_dict = ['T1', 'T2', 'T1T2']

    tpl_trans_dir = f"{tpl_root}/tpl_xfm_build"
    tpl_dirs    = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
    tpl_months  = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)
    path_months = tpl_months[tpl_months.index(tpl_mov_month):tpl_months.index(tpl_fix_month) + 1]

    if path_months == [tpl_mov_month, tpl_fix_month]:
        print(f"[INFO] {tpl_mov_month}M and {tpl_fix_month}M are neighbors.")
    else:
        print(f"[INFO] Chaining through: {' -> '.join(path_months)}")

    commands = []
    for modalities in modalities_dict:
        # Build xfm chain
        xfm_chain = []
        for m_from, m_to in zip(path_months[:-1], path_months[1:]):
            if transf_type == 'tplonly':
                xfm_chain.append(f"-t {os.path.join(tpl_trans_dir, 'tplonly', f'{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_0GenericAffine.mat')}")
                xfm_chain.append(f"-t {os.path.join(tpl_trans_dir, 'tplonly', f'{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_1Warp.nii.gz')}")
            elif transf_type in ['avgsubj_all', 'avgsubj_30_all']:
                strategy = transf_type.replace('avgsubj_', '')
                xfm_chain.append(f"-t {os.path.join(tpl_trans_dir, 'avgsubj', f'Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_{strategy}_by_{modalities}_xfm.nii.gz')}")

        xfm_chain.reverse()
        xfm_chain_str = "\n    ".join(
            [f"{line} \\" for line in xfm_chain[:-1]] + [xfm_chain[-1]]
        )

        # Reslice
        if modalities == 'T1T2':
            tpl_mov_T1 = f'{tpl_root}/{tpl_mov_month}Month/BCP-{tpl_mov_month}M-T1.nii.gz'
            tpl_mov_T2 = f'{tpl_root}/{tpl_mov_month}Month/BCP-{tpl_mov_month}M-T2.nii.gz'
            tpl_fix_T1 = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T1.nii.gz'
            tpl_fix_T2 = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T2.nii.gz'
            out_T1 = os.path.join(tpl_trans_dir, 'tpl_trans',
                         f"T1_{tpl_mov_month}Mtpl_resliced_to_{tpl_fix_month}Mtpl_by_{transf_type}_by_{modalities}_xfm.nii.gz")
            out_T2 = os.path.join(tpl_trans_dir, 'tpl_trans',
                         f"T2_{tpl_mov_month}Mtpl_resliced_to_{tpl_fix_month}Mtpl_by_{transf_type}_by_{modalities}_xfm.nii.gz")
            # reslice_cmd = f"""
            #         antsApplyTransforms -d 3 \\
            #         -i {tpl_mov_T1} \\
            #         -r {tpl_fix_T1} \\
            #         -o {out_T1} \\
            #         {xfm_chain_str}
            #         """
            # commands.append(reslice_cmd.strip())
            # reslice_cmd = f"""
            #         antsApplyTransforms -d 3 \\
            #         -i {tpl_mov_T2} \\
            #         -r {tpl_fix_T2} \\
            #         -o {out_T2} \\
            #         {xfm_chain_str}
            #         """
            # commands.append(reslice_cmd.strip())
            commands.append(
                f"antsApplyTransforms -d 3 \\\n"
                f"    -i {tpl_mov_T1} \\\n"
                f"    -r {tpl_fix_T1} \\\n"
                f"    -o {out_T1} \\\n"
                f"    {xfm_chain_str}"
            )
            commands.append(
                f"antsApplyTransforms -d 3 \\\n"
                f"    -i {tpl_mov_T2} \\\n"
                f"    -r {tpl_fix_T2} \\\n"
                f"    -o {out_T2} \\\n"
                f"    {xfm_chain_str}"
            )
        else:
            tpl_mov_mod = f'{tpl_root}/{tpl_mov_month}Month/BCP-{tpl_mov_month}M-{modalities}.nii.gz'
            tpl_fix_mod = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-{modalities}.nii.gz'
            out_mod = os.path.join(tpl_trans_dir, 'tpl_trans',
                          f"{modalities}_{tpl_mov_month}Mtpl_resliced_to_{tpl_fix_month}Mtpl_by_{transf_type}_by_{modalities}_xfm.nii.gz")
            # reslice_cmd = f"""
            # antsApplyTransforms -d 3 \\
            #         -i {tpl_mov_mod} \\
            #         -r {tpl_fix_mod} \\
            #         -o {out_mod} \\
            #         {xfm_chain_str}
            #         """
            # commands.append(reslice_cmd.strip())
            commands.append(
                f"antsApplyTransforms -d 3 \\\n"
                f"    -i {tpl_mov_mod} \\\n"
                f"    -r {tpl_fix_mod} \\\n"
                f"    -o {out_mod} \\\n"
                f"    {xfm_chain_str}"
            )

    full_cmd   = "\n\n".join(commands)
    log_dir    = os.path.join(tpl_root, "tpl_xfm_build/log")
    job_prefix = f"resl_{tpl_mov_month}_{tpl_fix_month}_{transf_type}_tpl"

    if slurm:
        submit_slurm_job(
            full_cmd=full_cmd, log_dir=log_dir, job_prefix=job_prefix,
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
        submit_bash_job(full_cmd, job_script, job_prefix,
                        job_script.replace('.sh', '.out'),
                        job_script.replace('.sh', '.err'),
                        num_threads, verbose)
    return True



def subj_concate_JD_and_resli(transf_type, pipel_dir, dataset_subs, tpl_mov_month, tpl_fix_month, tpl_root, modalities_dict=None, multi_step=True, **kwargs):
    """
    Reslice the subject to tpl_fix space via tpl_mov using two types displacement field: 'directtpl', 'averagesubj'
    Args:
        transf_type (str): 'tplonly' | 'l1o' | '30_l1o' | 'all' | '30_all'
        pipel_dir (str): Pipeline root directory
        dataset_subs (pd.DataFrame): columns ['dataset', 'participant_id']
        tpl_mov_month (str): Moving template month (e.g., "00", "08")
        tpl_fix_month (str): Fixed template month (e.g., "216")
        tpl_root (str): Root directory of templates
        multi_step (bool): If True (default), chain transforms through all intermediate monthly
                          templates between tpl_mov and tpl_fix (e.g. 08→09→...→60→216).
                          If False, use a single direct transform from tpl_mov to tpl_fix (08→216),
                          which requires the corresponding single-step transform files to exist.
    """

    num_threads = kwargs.get('num_threads', 6)
    slurm = kwargs.get('slurm', True)
    verbose = kwargs.get('verbose', True)
    if modalities_dict is None:
        modalities_dict = ['T1', 'T2', 'T1T2']

    tpl_trans_dir = f"{tpl_root}/tpl_xfm_build"
    transf_type_str = f"avgsubj_{transf_type}" if transf_type in ["l1o", "30_l1o", "30_all", "all"] else transf_type
    if multi_step: # need to re-run for stepwise or clean old?
        transf_type_str = f"{transf_type_str}_multistep"
    else:
        transf_type_str = f"{transf_type_str}_1step"
    if multi_step:
        # Chain through all intermediate monthly templates between mov and fix
        tpl_dirs = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
        tpl_months = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)
        month_order = tpl_months
        idx_mov = month_order.index(tpl_mov_month)
        idx_fix = month_order.index(tpl_fix_month)
        path_months = month_order[idx_mov:idx_fix + 1]
    else:
        # Direct: single step from tpl_mov to tpl_fix, no intermediate templates
        path_months = [tpl_mov_month, tpl_fix_month]
    commands = []
    for modalities in modalities_dict:
        ref_mod = 'T1' if modalities == 'T1T2' else modalities
        tpl_fix_img = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-{ref_mod}.nii.gz'

        if path_months == [tpl_mov_month, tpl_fix_month]:  # Direct neighbors, no chaining needed
            print(f"[INFO] {tpl_mov_month}M and {tpl_fix_month}M are neighbors. Using neighbor transform.")
        else:
            print(f"[INFO] {tpl_mov_month}M and {tpl_fix_month}M are not neighbors. Chaining transforms through: {' → '.join(path_months)}")
            
        for dataset, subid in dataset_subs.values: 
            # Step 1: Combine transforms
            output_dir = os.path.join(pipel_dir, dataset, subid) 
            output_xfm = os.path.join(output_dir, f"Displacement_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type_str}_by_{modalities}_xfm.nii.gz")
            geometric_jd_file = os.path.join(output_dir, f"Displacement_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type_str}_by_{modalities}_xfm_log_geometric_JD.nii.gz")
            

            xfm_chain = []
            # First: subject ➝ mov template
            # might change to aff and warp 2 files to keep less intermediate files, but for now keep it simple with one combined aff+warp file
            subject_to_mov_aff = os.path.join(
                output_dir,f"{modalities}_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_0GenericAffine.mat") 
            xfm_chain.append(f"-t {subject_to_mov_aff}")
            subject_to_mov_warp = os.path.join(
                output_dir,f"{modalities}_Brain_pad_Norm_to_{tpl_mov_month}Mtpl_1Warp.nii.gz") 
            xfm_chain.append(f"-t {subject_to_mov_warp}")

            # Then: mov ➝ fix through intermediate template warps
            
            for k in range(len(path_months) - 1):
                m_from = path_months[k]
                m_to = path_months[k + 1]
                if transf_type == 'tplonly':
                    aff_path = os.path.join(
                        tpl_trans_dir, transf_type, 
                        f"{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_0GenericAffine.mat"
                    )
                    xfm_chain.append(f"-t {aff_path}")
                    warp_path = os.path.join(
                        tpl_trans_dir, transf_type, 
                        f"{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_1Warp.nii.gz"
                    )
                    xfm_chain.append(f"-t {warp_path}")
                elif transf_type in ['l1o','30_l1o']:

                    xfm_path = os.path.join(
                        output_dir,
                        f"Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_via_other_subjs_{transf_type}_by_{modalities}_xfm.nii.gz"
                    )

                    xfm_chain.append(f"-t {xfm_path}")
                elif transf_type in ['all','30_all']:
                    xfm_path = os.path.join(
                        tpl_trans_dir,  'avgsubj', 
                        f"Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_{transf_type}_by_{modalities}_xfm.nii.gz"
                    )
                    xfm_chain.append(f"-t {xfm_path}")

            # Reverse for antsApplyTransforms (last → first)
            xfm_chain.reverse()

            # Format lines: add "\" to all except last
            xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
            xfm_lines.append(xfm_chain[-1])
            xfm_chain_str = "\n    ".join(xfm_lines)  

            # Log for each subject
            commands.append(
                f"echo '[{modalities}] {subid} ({dataset})  "
                f"{tpl_mov_month}M→{tpl_fix_month}M  {transf_type_str}'"
            )

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
                {geometric_jd_file} 1 1
                """
            commands.append(generate_jd_cmd.strip())

            if modalities == "T1T2":
                tpl_fix_t1 = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T1.nii.gz'
                out_warped_t1 = os.path.join(output_dir, f"T1_resliced_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type_str}_by_{modalities}_xfm.nii.gz")
                reslice_t1_cmd = f"""
                    antsApplyTransforms -d 3 \\
                    -i {os.path.join(output_dir, f'T1_Brain_pad.nii.gz')} \\
                    -r {tpl_fix_t1} \\
                    -o {out_warped_t1} \\
                    {xfm_chain_str}
                    """
                commands.append(reslice_t1_cmd.strip())

                tpl_fix_t2 = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T2.nii.gz'
                out_warped_t2 = os.path.join(output_dir,f"T2_resliced_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type_str}_by_{modalities}_xfm.nii.gz")
                reslice_t2_cmd = f"""
                    antsApplyTransforms -d 3 \\
                    -i {os.path.join(output_dir, 'T2_Brain_pad_rigid2T1_N4.nii.gz')} \\
                    -r {tpl_fix_t2} \\
                    -o {out_warped_t2} \\
                    {xfm_chain_str}
                    """
                commands.append(reslice_t2_cmd.strip())
            else:
                tpl_fix_mod = f'{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-{modalities}.nii.gz'
                out_warped_mod = os.path.join(output_dir, f"{modalities}_resliced_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl_by_{transf_type_str}_by_{modalities}_xfm.nii.gz")
                input_mod = os.path.join(output_dir, f'{modalities}_Brain_pad.nii.gz') if modalities=='T1' else os.path.join(output_dir, f'{modalities}_Brain_pad_rigid2T1_N4.nii.gz')
                reslice_mod_cmd = f"""
                    antsApplyTransforms -d 3 \\
                    -i {input_mod} \\
                    -r {tpl_fix_mod} \\
                    -o {out_warped_mod} \\
                    {xfm_chain_str}
                    """
                commands.append(reslice_mod_cmd.strip())
            # delete out_xfm in the end to save space since we only care about the resliced image and JD, but keep it for now for debugging and checking the xfm quality
            delete_xfm_cmd = f"rm {output_xfm}"
            commands.append(delete_xfm_cmd.strip())
            commands.append(
                f"echo '[DONE] {subid} ({dataset})  {modalities}  $(date +%H:%M:%S)'"
            )

        

    # Combine all commands
    full_cmd = "\n\n".join(commands)
    print(full_cmd)
    # Submit
    log_dir = os.path.join(tpl_trans_dir, "log")
    job_prefix = f"resl_{tpl_mov_month}_{tpl_fix_month}_{transf_type_str}_subj"
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
            # email=kwargs.get("email", None),
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

    

def subj_cereb_isolate(pipel_dir, dataset_subs, **kwargs):
    """
    Run SUIT isolation for each subject: T1_Brain_pad.nii.gz → T1_Brain_pad_cerebellum_dseg.nii.gz
    Skipped per-subject if the mask already exists.
    Submit this first; use the returned job_id as dependency_jobid for subj_cereb_reslice.

    Args:
        pipel_dir    (str)          : pipeline root directory (…/Data)
        dataset_subs (pd.DataFrame) : columns ['dataset', 'participant_id']
        kwargs       : num_threads, slurm, verbose, time_limit, mem, job_prefix, …
    Returns:
        job_id (str or None)
    """
    num_threads = kwargs.get('num_threads', 4)
    slurm       = kwargs.get('slurm', True)
    verbose     = kwargs.get('verbose', True)
    job_prefix  = kwargs.get('job_prefix', 'cereb_isolate')

    commands = []

    for dataset, subid in dataset_subs.values:
        output_dir = os.path.join(pipel_dir, dataset, subid)
        t1_input   = os.path.join(output_dir, "T1_Brain_pad_N4.nii.gz")
        t2_input   = os.path.join(output_dir, "T2_Brain_pad_rigid2T1_N4.nii.gz")
        
        commands.append(f"echo '[SUIT] {subid} ({dataset})  $(date +%H:%M:%S)'")

        suit_cmd = (
            f'python - <<\'PYEOF\'\n'
            f'import nibabel as nib\n'
            f'import numpy as np\n'
            f'import SUITPy as suit\n'
            f'\n'
            f'# Generate brain mask from T1 (already brain-extracted, bg=0)\n'
            f't1      = nib.load("{t1_input}")\n'
            f'mask    = (t1.get_fdata() > 0).astype(np.uint8)\n'
            f'brain_mask_path = "{t1_input}".replace(".nii.gz", "_brain_mask.nii.gz")\n'
            f'nib.save(nib.Nifti1Image(mask, t1.affine, t1.header), brain_mask_path)\n'
            f'\n'
            f'suit.isolate(t1_file="{t1_input}", t2_file="{t2_input}", brain_mask_file=brain_mask_path)\n'
            f'print("SUIT done: {subid}")\n'
            f'PYEOF'
        )
        commands.append(suit_cmd)
        commands.append(f"echo '[DONE] {subid} ({dataset})  $(date +%H:%M:%S)'")
    log_dir    = '/project/4290000.01/yapwan/toolbox/BCP-atlas-for_release-Ver2.0.0/tpl_xfm_build/log/'
    full_cmd = "\n\n".join(commands)

    if slurm:
        job_id = submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=num_threads,
            time_limit=kwargs.get("time_limit", "12:00:00"),
            mem=kwargs.get("mem", "64G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
        return job_id
    else:
        job_script = os.path.join(log_dir, f"{job_prefix}.sh")
        output_log = os.path.join(log_dir, f"{job_prefix}.out")
        error_log  = os.path.join(log_dir, f"{job_prefix}.err")
        submit_bash_job(full_cmd, job_script, job_prefix,
                        output_log, error_log, num_threads, verbose)
        return None


def subj_cereb_reslice(transf_type, pipel_dir, dataset_subs, tpl_mov_month, tpl_root,
                       modalities_dict=None, multi_step=True, **kwargs):
    """
    Reslice each subject's cerebellar mask (T1_Brain_pad_cerebellum_dseg.nii.gz)
    from native space to 216M adult space, using the same transform chain as
    subj_concate_JD_and_resli (NearestNeighbor interpolation).

    Assumes the mask already exists (run subj_cereb_isolate first and pass its
    job_id via dependency_jobid).

    Args:
        transf_type   (str)          : 'tplonly' | 'l1o' | '30_l1o' | 'all' | '30_all'
        pipel_dir     (str)          : pipeline root directory (…/Data)
        dataset_subs  (pd.DataFrame) : columns ['dataset', 'participant_id']
        tpl_mov_month (str)          : age-matched template month, zero-padded (e.g. '00')
        tpl_root      (str)          : BCP atlas root directory
        modalities_dict (list)       : registration modalities for transform filenames,
                                       default ['T1', 'T1T2']
        multi_step    (bool)         : If True (default), chain through all intermediate
                                       monthly templates (tpl_mov → … → 216), and append
                                       '_stepwise' to the output filename. If False, use a
                                       single direct step tpl_mov → 216. Ignored for
                                       transf_type='direct' (always subject → 216 directly).
        kwargs        : num_threads, slurm, verbose, time_limit, mem, dependency_jobid, …
    Returns:
        job_id (str or None)
    """
    num_threads = kwargs.get('num_threads', 4)
    slurm       = kwargs.get('slurm', True)
    verbose     = kwargs.get('verbose', True)

    if modalities_dict is None:
        modalities_dict = ['T1', 'T1T2']

    tpl_fix_month   = '216'
    tpl_trans_dir   = f"{tpl_root}/tpl_xfm_build"
    transf_type_str = f"avgsubj_{transf_type}" if transf_type in ("l1o", "30_l1o", "30_all", "all") else transf_type
    if multi_step and transf_type != 'direct':
        transf_type_str = f"{transf_type_str}_multistep"
    else:
        transf_type_str = f"{transf_type_str}_1step"
    # Build step-by-step template path: tpl_mov_month → 216
    tpl_dirs    = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
    tpl_months  = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)
    if multi_step:

        idx_mov     = tpl_months.index(tpl_mov_month)
        idx_fix     = tpl_months.index(tpl_fix_month)
        path_months = tpl_months[idx_mov : idx_fix + 1]
    else:
        path_months = [tpl_mov_month, tpl_fix_month]
    tpl_fix_img = f"{tpl_root}/{tpl_fix_month}Month/BCP-{tpl_fix_month}M-T1.nii.gz"

    if path_months == [tpl_mov_month, tpl_fix_month]:
        print(f"[INFO] {tpl_mov_month}M and {tpl_fix_month}M are neighbors. Using neighbor transform.")
    else:
        print(f"[INFO] {tpl_mov_month}M and {tpl_fix_month}M are not neighbors. "
              f"Chaining through: {' → '.join(path_months)}")

    commands = []

    for modalities in modalities_dict:
        for dataset, subid in dataset_subs.values:
            output_dir = os.path.join(pipel_dir, dataset, subid)

            cereb_mask     = os.path.join(output_dir, "T1_Brain_pad_N4_cerebellum_dseg.nii.gz")
            cereb_resliced = os.path.join(
                output_dir,
                f"T1_cereb_mask_to_{tpl_mov_month}Mtpl_then_to_{tpl_fix_month}Mtpl"
                f"_by_{transf_type_str}_by_{modalities}_xfm.nii.gz",
            )

            commands.append(
                f"echo '[{transf_type_str}|{modalities}] {subid} ({dataset})  "
                f"{tpl_mov_month}M→{tpl_fix_month}M  $(date +%H:%M:%S)'"
            )

            # ── Build transform chain ──────────────────────────────────────────
            xfm_chain = []

            if transf_type == 'direct':
                # Subject → 216M directly (no age-matched template step)
                xfm_chain.append(f"-t {os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat')}")
                xfm_chain.append(f"-t {os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz')}")

            else:
                # Subject → age-matched template
                xfm_chain.append(f"-t {os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{path_months[0]}Mtpl_0GenericAffine.mat')}")
                xfm_chain.append(f"-t {os.path.join(output_dir, f'{modalities}_Brain_pad_Norm_to_{path_months[0]}Mtpl_1Warp.nii.gz')}")

                # Age-matched template → 216M
                for k in range(len(path_months) - 1):
                    m_from, m_to = path_months[k], path_months[k + 1]
                    if transf_type == 'tplonly':
                        xfm_chain.append(f"-t {os.path.join(tpl_trans_dir, transf_type, f'{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_0GenericAffine.mat')}")
                        xfm_chain.append(f"-t {os.path.join(tpl_trans_dir, transf_type, f'{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_1Warp.nii.gz')}")
                    elif transf_type in ('l1o', '30_l1o'):
                        xfm_chain.append(f"-t {os.path.join(output_dir, f'Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_via_other_subjs_{transf_type}_by_{modalities}_xfm.nii.gz')}")
                    elif transf_type in ('all', '30_all'):
                        xfm_chain.append(f"-t {os.path.join(tpl_trans_dir, 'avgsubj', f'Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_{transf_type}_by_{modalities}_xfm.nii.gz')}")

            # Reverse for antsApplyTransforms (last → first)
            xfm_chain.reverse()

            xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]]
            xfm_lines.append(xfm_chain[-1])
            xfm_chain_str = "\n    ".join(xfm_lines)

            # ── Reslice (NearestNeighbor preserves binary mask) ────────────────
            reslice_cmd = f"""antsApplyTransforms -d 3 \\
    -i {cereb_mask} \\
    -r {tpl_fix_img} \\
    -o {cereb_resliced} \\
    -n NearestNeighbor \\
    {xfm_chain_str}"""
            commands.append(reslice_cmd.strip())

            commands.append(f"echo '[DONE] {subid} ({dataset})  $(date +%H:%M:%S)'")

    # ── Submit ─────────────────────────────────────────────────────────────────
    full_cmd   = "\n\n".join(commands)
    log_dir    = os.path.join(tpl_trans_dir, "log")
    job_prefix = f"cereb_{tpl_mov_month}_to_{tpl_fix_month}_{transf_type_str}"

    if slurm:
        job_id = submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=num_threads,
            time_limit=kwargs.get("time_limit", "12:00:00"),
            mem=kwargs.get("mem", "16G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
        return job_id
    else:
        job_script = os.path.join(log_dir, f"{job_prefix}.sh")
        output_log = os.path.join(log_dir, f"{job_prefix}.out")
        error_log  = os.path.join(log_dir, f"{job_prefix}.err")
        submit_bash_job(full_cmd, job_script, job_prefix,
                        output_log, error_log, num_threads, verbose)
        return None


def tpl_cereb_isolate(tpl_root, tpl_months, **kwargs):
    """
    Run SUIT isolation on each BCP template T1 image to generate cerebellar masks.

    For each month in tpl_months:
      Input  : {tpl_root}/{month}Month/BCP-{month}M-T1.nii.gz
      Output : {tpl_root}/{month}Month/BCP-{month}M-cereb_mask.nii.gz

    Submit this first; use the returned job_id as dependency_jobid for tpl_cereb_reslice.

    Args:
        tpl_root   (str)       : BCP atlas root directory
        tpl_months (list[str]) : zero-padded month strings to process,
                                 e.g. ['00','01','02',...,'60']
                                 (typically excludes '216' which is the reference)
        kwargs     : num_threads, slurm, verbose, time_limit, mem, job_prefix, …
    Returns:
        job_id (str or None)
    """
    num_threads = kwargs.get('num_threads', 4)
    slurm       = kwargs.get('slurm', True)
    verbose     = kwargs.get('verbose', True)
    job_prefix  = kwargs.get('job_prefix', 'tpl_cereb_isolate')
    log_dir     = kwargs.get('log_dir', os.path.join(tpl_root, 'tpl_xfm_build', 'log'))

    commands = []

    for month in tpl_months:
        t1_input   = os.path.join(tpl_root, f"{month}Month", f"BCP-{month}M-T1.nii.gz")
        t2_input   = os.path.join(tpl_root, f"{month}Month", f"BCP-{month}M-T2.nii.gz")
        cereb_mask = os.path.join(tpl_root, f"{month}Month", f"BCP-{month}M-cereb_mask.nii.gz")

        commands.append(f"echo '[SUIT] {month}M template  $(date +%H:%M:%S)'")

        suit_cmd = (
            f'python - <<\'PYEOF\'\n'
            f'import nibabel as nib\n'
            f'import numpy as np\n'
            f'import SUITPy as suit\n'
            f'\n'
            f'# Build brain mask from template T1 (background = 0)\n'
            f't1 = nib.load("{t1_input}")\n'
            f'mask = (t1.get_fdata() > 0).astype(np.uint8)\n'
            f'brain_mask_path = "{t1_input}".replace(".nii.gz", "_brain_mask.nii.gz")\n'
            f'nib.save(nib.Nifti1Image(mask, t1.affine, t1.header), brain_mask_path)\n'
            f'\n'
            f'suit.isolate(t1_file="{t1_input}", t2_file="{t2_input}", brain_mask_file=brain_mask_path)\n'
            f'print("SUIT done: {month}M template")\n'
            f'PYEOF'
        )
        commands.append(suit_cmd)
        commands.append(f"echo '[DONE] {month}M  $(date +%H:%M:%S)'")

    full_cmd = "\n\n".join(commands)

    if slurm:
        job_id = submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=num_threads,
            time_limit=kwargs.get("time_limit", "12:00:00"),
            mem=kwargs.get("mem", "64G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
        return job_id
    else:
        job_script = os.path.join(log_dir, f"{job_prefix}.sh")
        output_log = os.path.join(log_dir, f"{job_prefix}.out")
        error_log  = os.path.join(log_dir, f"{job_prefix}.err")
        submit_bash_job(full_cmd, job_script, job_prefix,
                        output_log, error_log, num_threads, verbose)
        return None


def tpl_cereb_reslice(transf_type, tpl_root, tpl_mov_months, modalities_dict=None, **kwargs):
    """
    Reslice each template cerebellar mask from its native month space to 216M adult space,
    using the same template-to-template transform chain as tpl_concate_resli
    (NearestNeighbor interpolation preserves binary mask).

    For each month in tpl_mov_months:
      Input  : {tpl_root}/{mov}Month/BCP-{mov}M-cereb_mask.nii.gz
      Output : {tpl_root}/tpl_xfm_build/tpl_trans/
                 T1_{mov}Mtpl_cereb_mask_to_216Mtpl_by_{transf_type}_by_{modalities}_xfm.nii.gz

    Run tpl_cereb_isolate first and pass its job_id via dependency_jobid.

    Args:
        transf_type     (str)       : 'tplonly' | 'avgsubj_all' | 'avgsubj_30_all'
        tpl_root        (str)       : BCP atlas root directory
        tpl_mov_months  (list[str]) : months to reslice, e.g. ['00','01',...,'60']
        modalities_dict (list[str]) : transform modalities to use,
                                      default ['T1', 'T1T2']
        kwargs          : num_threads, slurm, verbose, time_limit, mem, dependency_jobid, …
    Returns:
        job_id (str or None)
    """
    num_threads = kwargs.get('num_threads', 4)
    slurm       = kwargs.get('slurm', True)
    verbose     = kwargs.get('verbose', True)

    if modalities_dict is None:
        modalities_dict = ['T1', 'T1T2']

    tpl_fix_month = '216'
    tpl_trans_dir = os.path.join(tpl_root, 'tpl_xfm_build')
    out_dir       = os.path.join(tpl_trans_dir, 'tpl_trans')
    os.makedirs(out_dir, exist_ok=True)

    # Build full month list for chaining
    tpl_dirs   = [d for d in os.listdir(tpl_root) if os.path.isdir(os.path.join(tpl_root, d))]
    all_months = sorted([d.replace("Month", "") for d in tpl_dirs if d.endswith("Month")], key=int)

    tpl_fix_img = os.path.join(tpl_root, f"{tpl_fix_month}Month",
                               f"BCP-{tpl_fix_month}M-T1.nii.gz")

    commands = []

    for modalities in modalities_dict:
        for mov in tpl_mov_months:
            cereb_mask    = os.path.join(tpl_root, f"{mov}Month",
                                         f"BCP-{mov}M-T1_cerebellum_dseg.nii.gz")
            cereb_resliced = os.path.join(
                out_dir,
                f"T1_{mov}Mtpl_cereb_mask_to_{tpl_fix_month}Mtpl"
                f"_by_{transf_type}_by_{modalities}_xfm.nii.gz",
            )

            # Step-by-step month chain: mov → 216
            idx_mov  = all_months.index(mov)
            idx_fix  = all_months.index(tpl_fix_month)
            chain_months = all_months[idx_mov : idx_fix + 1]

            if chain_months == [mov, tpl_fix_month]:
                chain_info = "direct neighbors"
            else:
                chain_info = " → ".join(chain_months)

            commands.append(
                f"echo '[{transf_type}|{modalities}] {mov}M→{tpl_fix_month}M  "
                f"({chain_info})  $(date +%H:%M:%S)'"
            )

            # Build transform chain
            xfm_chain = []
            if transf_type == 'direct':
                xfm_chain.append(
                    f"-t {os.path.join(tpl_trans_dir, 'tplonly', f'{modalities}_{mov}Mtpl_Norm_to_{tpl_fix_month}Mtpl_0GenericAffine.mat')}"
                )
                xfm_chain.append(
                    f"-t {os.path.join(tpl_trans_dir, 'tplonly', f'{modalities}_{mov}Mtpl_Norm_to_{tpl_fix_month}Mtpl_1Warp.nii.gz')}"
                )
            else:
                for m_from, m_to in zip(chain_months[:-1], chain_months[1:]):
                    if transf_type == 'tplonly':
                        xfm_chain.append(
                            f"-t {os.path.join(tpl_trans_dir, 'tplonly', f'{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_0GenericAffine.mat')}"
                        )
                        xfm_chain.append(
                            f"-t {os.path.join(tpl_trans_dir, 'tplonly', f'{modalities}_{m_from}Mtpl_Norm_to_{m_to}Mtpl_1Warp.nii.gz')}"
                        )
                    elif transf_type in ('avgsubj_all', 'avgsubj_30_all'):
                        strategy = transf_type.replace('avgsubj_', '')
                        xfm_chain.append(
                            f"-t {os.path.join(tpl_trans_dir, 'avgsubj', f'Avg_Displacement_{m_from}Mtpl_to_{m_to}Mtpl_{strategy}_by_{modalities}_xfm.nii.gz')}"
                        )
                    else:
                        raise ValueError(
                            f"transf_type '{transf_type}' not supported for template reslice. "
                            "Choose: 'tplonly' | 'avgsubj_all' | 'avgsubj_30_all'"
                        )

            # Reverse for ANTs (last → first)
            xfm_chain.reverse()
            xfm_lines = [f"{line} \\" for line in xfm_chain[:-1]] + [xfm_chain[-1]]
            xfm_chain_str = "\n    ".join(xfm_lines)

            reslice_cmd = (
                f"antsApplyTransforms -d 3 \\\n"
                f"    -i {cereb_mask} \\\n"
                f"    -r {tpl_fix_img} \\\n"
                f"    -o {cereb_resliced} \\\n"
                f"    -n NearestNeighbor \\\n"
                f"    {xfm_chain_str}"
            )
            commands.append(reslice_cmd)
            commands.append(f"echo '[DONE] {mov}M  $(date +%H:%M:%S)'")

    full_cmd   = "\n\n".join(commands)
    log_dir    = os.path.join(tpl_trans_dir, 'log')
    job_prefix = f"tpl_cereb_{tpl_fix_month}_{transf_type}"

    if slurm:
        job_id = submit_slurm_job(
            full_cmd=full_cmd,
            log_dir=log_dir,
            job_prefix=job_prefix,
            num_threads=num_threads,
            time_limit=kwargs.get("time_limit", "12:00:00"),
            mem=kwargs.get("mem", "16G"),
            ntasks=kwargs.get("ntasks", 1),
            use_gpu=kwargs.get("use_gpu", False),
            ants_path=kwargs.get("ants_path", DEFAULT_ANTSPATH),
            dependency_jobid=kwargs.get("dependency_jobid", None),
            verbose=verbose,
        )
        return job_id
    else:
        job_script = os.path.join(log_dir, f"{job_prefix}.sh")
        output_log = os.path.join(log_dir, f"{job_prefix}.out")
        error_log  = os.path.join(log_dir, f"{job_prefix}.err")
        submit_bash_job(full_cmd, job_script, job_prefix,
                        output_log, error_log, num_threads, verbose)
        return None


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
        f"#SBATCH --mem={mem}"
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
        "set -e", 
        f"echo 'Starting job: {log_name_full}'",
        f"export ANTSPATH={ants_path}",
        "export PATH=$ANTSPATH/bin:$PATH",
        "export LD_LIBRARY_PATH=$ANTSPATH/lib:$LD_LIBRARY_PATH",
        "echo 'Checking antsBrainExtraction.sh path: '",
        "which antsBrainExtraction.sh || echo '[WARNING] antsBrainExtraction.sh not found in PATH'",
        "echo 'ANTs Version:'",
        "antsRegistration --version || echo '[WARNING] antsRegistration not found'",
        f"{full_cmd}",
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
        # print("[INFO] Command:", " ".join(cmd_sbatch))

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
