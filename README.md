# Domain-Specific Normalization vs. Feature Alignment for Continual Wildfire Segmentation

Code for the paper **"On Domain-Specific Normalization and Feature Alignment for Replay-Based Continual Segmentation of Heterogeneous Wildfire Imagery."**

This repository contains the training, evaluation, and analysis code for a controlled study of two domain-adaptation mechanisms — **Domain-Specific Batch Normalization (DSBN)** and **Deep CORAL feature alignment** — added on top of a strong Learning-without-Forgetting (LwF) + experience-replay baseline, for continual semantic segmentation across two heterogeneous wildfire domains (smoke → flame).

---

## Summary of findings

- **DSBN** significantly improves new-domain (flame / T2) segmentation (**+1.03 mIoU points, paired t-test p = 0.0002**) and reduces feature drift on the old domain.
- **Deep CORAL** feature alignment **hurts** both tasks, because forcing together classes that are physically distinct (diffuse smoke vs. small bright flame) erodes discriminability.
- Because DSBN trades a small amount of old-task retention for its new-task gain, the two effects cancel in the pooled average (not significant) — so heterogeneous continual segmentation must be evaluated **per task**, not only in aggregate.

| Method | Avg mIoU (3 seeds) | T2 mIoU (flame) |
|---|---|---|
| LwF + Replay (baseline) | 0.8314 | 0.8015 |
| + DSBN | **0.8345** | **0.8118** |
| + CORAL | 0.8207 | 0.7957 |
| DSBN + CORAL | 0.8257 | 0.8083 |

---

## Tasks and datasets (publicly available)

Two public UAV wildfire datasets, learned sequentially:

- **T1 — Boreal Forest Fire** (foreground: smoke plume). Pesonen et al., *Scientific Data* **12**, 1419 (2025). DOI: 10.1038/s41597-025-05634-0
- **T2 — FLAME** (foreground: flame). Shamsoshoara et al., *Computer Networks* **193**, 108001 (2021).

No new data were created. Download the datasets from their original sources and set the paths in the notebook's configuration cell.

---

## Repository contents

```
.
├── DA_CLLwF_Wildfire_Stage1_Colab.ipynb   # training, variants, significance, figures
└── README.md
```
> Rename the notebook line above to match the actual file in your repo.

---

## Method

- **Backbone:** shared ResNet-50 encoder + FPN decoder (ImageNet-pretrained), with a lightweight task-specific head per task.
- **Continual objective (T2):** segmentation loss + LwF logit distillation + deepest-feature distillation + experience replay.
- **DSBN:** per-task BatchNorm statistics in the shared encoder, routed by task id (no added conv parameters, no inference cost).
- **Deep CORAL:** second-order feature alignment on a bottlenecked deepest-encoder feature, applied on replay-interleaved steps.
- **Diagnostics:** CKA feature-drift measurement; paired t-tests over 3 seeds.

---

## Requirements

```bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python numpy pandas matplotlib scipy
```

---

## How to run

1. Open the notebook in **Google Colab** (GPU runtime) or Jupyter.
2. Mount Drive and set dataset paths in the CONFIG cell.
3. Run top to bottom. The notebook is **resume-safe**: checkpoints and per-run results are cached, so finished runs are skipped and interrupted runs resume from the last epoch.
4. The variant loop trains `replay_only`, `cllwf`, `+DSBN`, `+CORAL`, and `DSBN+CORAL` across seeds, then runs the significance test and saves all figures.

---

## Reproducibility

- Fixed seeds (42, 43, 44) and deterministic cuDNN.
- Per-epoch checkpointing with automatic resume.
- All reported numbers come directly from the notebook's evaluation cells; the significance test and figures are generated from the saved per-seed results.

---

## Citation

```bibtex
@article{kumar_dsbn_coral_wildfire,
  title   = {On Domain-Specific Normalization and Feature Alignment for Replay-Based Continual Segmentation of Heterogeneous Wildfire Imagery},
  author  = {Kumar, Sujeet and Bhattacharya, Rajarshi},
  note    = {Manuscript},
  year    = {2026}
}
```

---

## Code

Repository: https://github.com/skelectronics1993/DSBN_CORAL_analysis_papercode

```bash
git clone https://github.com/skelectronics1993/DSBN_CORAL_analysis_papercode.git
```

## Contact

**Sujeet Kumar** — Department of Electronics and Communication Engineering, National Institute of Technology Patna, India. Email: sujeetk.phd24.ec@nitp.ac.in

## License

Add a `LICENSE` file (e.g., MIT). Datasets retain their original licenses.
