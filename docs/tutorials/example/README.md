# Example visits of the tutorials

Two infant visits from the **Baby Open Brains (BOBs) repository** (Feczko, Stoyell et al.; participants of the Baby Connectome Project; OpenNeuro and `s3://bobsrepository`, DOI [10.17605/OSF.IO/WDR78](https://doi.org/10.17605/OSF.IO/WDR78)), licensed **CC-BY-4.0**. The images are unchanged, only renamed:

| here | BOBs file | age (months) | gestational age at birth (weeks) |
|---|---|---|---|
| `sub-ex01_T1w.nii.gz`, `sub-ex01_T2w.nii.gz` | `sub-584381/ses-1mo/anat/sub-584381_ses-1mo_space-INFANTMNIacpc_T1w.nii.gz`, `_T2w.nii.gz` | 1.0 | 42.0 |
| `sub-ex01_T1w_manual_mask.nii.gz` (made here) | `sub-584381/ses-1mo/anat/sub-584381_ses-1mo_space-INFANTMNIacpc_desc-aseg_dseg.nii.gz` | 1.0 | 42.0 |
| `sub-ex02_T1w.nii.gz` (T1w only) | `sub-372377/ses-8mo/anat/sub-372377_ses-8mo_space-INFANTMNIacpc_T1w.nii.gz` | 8.8 | 40.3 |

The images are the preprocessed T1w and T2w of BOBs (DCAN-Labs infant-abcd-bids-pipeline: ACPC-aligned, 1 mm, not skull-stripped, the T2w on the T1w grid), not the images as acquired. Ages: `phenotype/sessions.tsv` of BOBs.

`sub-ex01_T1w_manual_mask.nii.gz` (4.2 of `1.quickstart_direct_normalize.ipynb`) is the only file made here: the binary mask (all labels > 0, on the T1w grid) of the BOBs segmentation of that session, made with BIBSNet and then corrected by hand (BOBs: brain tissue and ventricles, no CSF around the brain).

Please cite: Feczko, E., Stoyell, S. M., et al. (2025). Baby Open Brains: An open-source dataset of infant brain segmentations. Scientific Data. https://doi.org/10.1038/s41597-025-05404-y

`out/` (the outputs of the tutorials) is not in the repository.
