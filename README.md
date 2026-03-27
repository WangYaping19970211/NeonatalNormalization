# **Cerebellar Normalization**
In large-scale neuroimaging studies, especially those involving longitudinal analysis, a critical first step is to transform individual brain images into a common template space. This standardization enables meaningful group-level comparisons and statistical analyses.

For adult brains, this process is typically performed using a **one-step** direct normalization to a standard template. However, this approach does not generalize well to neonatal brains. Due to substantial anatomical differences between neonatal and adult brains, direct normalization to an adult template often fails or yields suboptimal results.

A commonly adopted solution is to first normalize neonatal brains to age-specific templates (e.g., the UNC BCP 4D atlas, https://www.nitrc.org/projects/uncbcp_4d_atlas/), and then progressively map them to older templates in a stepwise manner until reaching the adult template space. This strategy has been shown to improve registration performance.

However, this **multi-step** approach may introduce systematic biases. If not properly addressed, these biases can propagate into downstream analyses and be mistakenly interpreted as biological signals.

**Objective:** Develop  a neonatal normalization pipeline that minimizes such biases and achieves consistency with adult normalization, thereby enabling more accurate and reliable cross-age comparisons.