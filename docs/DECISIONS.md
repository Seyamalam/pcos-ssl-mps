# Decision Log

Research decisions that should be preserved for the manuscript.

| Date | Decision | Rationale | Consequence |
|---|---|---|---|
| 2026-05-21 | Use self-supervised learning as the main paper direction. | Plain CNN classification is unlikely to be novel; SSL is better matched to noisy and low-label medical imaging. | Main experiments compare ImageNet-supervised baselines with SimCLR pretraining and downstream fine-tuning. |
| 2026-05-21 | Treat duplicates as a central methodology issue. | Local audit found 1,956 exact duplicate groups and 7,788 duplicate files beyond the first copy. | Main split prevents exact duplicate groups crossing train/val/test. |
| 2026-05-21 | Add pHash near-duplicate split as stricter benchmark. | pHash threshold 4 found 1,788 near-duplicate groups, 9,448 files beyond first, and 38 cross-label near-duplicate groups. | Strict benchmark excludes cross-label near groups and prevents near groups crossing splits. |
| 2026-05-21 | Keep raw dataset and image derivatives out of GitHub. | Kaggle image files and Grad-CAM derivatives may carry dataset redistribution/licensing concerns. | `PCOS/`, checkpoints, runs, and Grad-CAM image folders are gitignored. |
| 2026-05-21 | Use PyTorch MPS with 18 CPU threads. | The Apple Silicon machine has a strong GPU plus many CPU cores; MPS handles model tensors while CPU workers handle image loading/augmentation. | Runtime logs always record `device=mps` and `cpu_threads=18`. |
| 2026-05-21 | Start SSL with SimCLR before BYOL/DINO. | SimCLR is simpler to debug, explain, and compare as a first SSL baseline. | BYOL/DINO remain planned after supervised and SimCLR baselines are stable. |
| 2026-05-21 | Treat very high baseline accuracy as a warning sign. | ResNet-18 reached 99.6% test accuracy after one epoch on the exact duplicate-aware split, but robustness collapsed under blur/downsample. | Next benchmark must use stricter pHash splits and robustness analysis. |
| 2026-05-22 | Report SSL results conservatively after the first 25-epoch SimCLR sweep. | SimCLR e25 improved over the one-epoch probe but did not beat supervised ResNet-18 at matched label fractions after one fine-tuning epoch. | Manuscript claims should emphasize leakage control, robustness, calibration, and label-efficiency nuance rather than claiming SSL superiority too early. |
