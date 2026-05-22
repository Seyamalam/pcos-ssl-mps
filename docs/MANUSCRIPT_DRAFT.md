# Manuscript Draft

Working title: **Beyond Clean Data: Self-Supervised Representation Learning for PCOS Ultrasound Classification Under Duplicate-Aware Evaluation**

## Abstract Draft

Public PCOS ultrasound classifiers often report high performance, but many datasets contain exact duplicates, near-duplicates, inconsistent preprocessing, and possible artifact shortcuts. We study whether self-supervised representation learning remains useful when evaluation is leakage-controlled and labels are limited. Using the PCOS-XAI ultrasound dataset, we audit exact and perceptual duplicates, construct pHash near-duplicate-aware train/validation/test splits, compare supervised CNN/ViT/ConvNeXt baselines with SimCLR- and BYOL-pretrained ResNet-18 encoders, and evaluate label efficiency, threshold-selected operating points, calibration, robustness, and Grad-CAM sanity checks. Current results show that supervised ImageNet-pretrained ResNet-18 remains a strong default, while SimCLR and BYOL fine-tuning become competitive after longer downstream training and can reach strong validation-selected sensitivity/specificity tradeoffs. Robustness and XAI sanity checks reveal substantial sensitivity to preprocessing and border/artifact perturbations, supporting the need for evaluation protocols beyond clean accuracy.

## Contributions

1. A leakage-controlled PCOS ultrasound evaluation protocol using exact duplicate and pHash near-duplicate grouping.
2. A label-efficiency comparison of supervised ImageNet transfer, SimCLR pretraining, and BYOL pretraining under matched low-label budgets where available.
3. Multi-seed evidence for 10% and 50% label budgets at 10 downstream epochs.
4. Calibration, threshold-selection, robustness severity, and Grad-CAM sanity-check analyses for reliability-oriented evaluation.
5. A reproducible Apple Silicon PyTorch/MPS workflow for local medical-imaging experimentation.

## Methods

Dataset audit records image readability, class balance, dimensions, exact MD5 duplicates, pHash near-duplicates, and cross-label near-duplicate groups. The main split is `metadata/splits_near_duplicate_aware_phash.csv`, which prevents near-duplicate leakage and excludes cross-label near-duplicate groups.

Model families include supervised ImageNet-pretrained ResNet-18, EfficientNet-B0, ViT-Tiny, ConvNeXt-Tiny, and SSL-pretrained ResNet-18 encoders using SimCLR and BYOL. Downstream evaluation uses default threshold metrics plus validation-selected thresholds optimized for balanced accuracy subject to minimum sensitivity. Calibration is measured with Brier score, ECE, and NLL; Platt scaling and temperature scaling are fit on validation predictions and evaluated on test predictions.

## Current Key Results

Multi-seed 10% label, 10 downstream epochs:

- Supervised ResNet-18: accuracy `0.9794 +/- 0.0075`, AUROC `0.9984 +/- 0.0009`, threshold-selected accuracy `0.9832 +/- 0.0149`.
- SimCLR e25 fine-tune: accuracy `0.9755 +/- 0.0019`, AUROC `0.9963 +/- 0.0011`, threshold-selected accuracy `0.9759 +/- 0.0080`.
- BYOL e25 fine-tune: accuracy `0.9507 +/- 0.0286`, AUROC `0.9942 +/- 0.0028`, threshold-selected accuracy `0.9675 +/- 0.0172`.

Multi-seed 50% label, 10 downstream epochs:

- Supervised ResNet-18: accuracy `0.9931 +/- 0.0044`, AUROC `0.9991 +/- 0.0000`, threshold-selected accuracy `0.9931 +/- 0.0044`.
- SimCLR e25 fine-tune: accuracy `0.9673 +/- 0.0029`, AUROC `0.9986 +/- 0.0004`, threshold-selected accuracy `0.9939 +/- 0.0025`.
- BYOL e25 fine-tune: accuracy `0.9734 +/- 0.0173`, AUROC `0.9981 +/- 0.0004`, threshold-selected accuracy `0.9899 +/- 0.0045`.

Calibration improvement:

- Platt scaling improves Brier score for several under-calibrated 10-epoch models.
- Temperature scaling is safer for some already-strong high-epoch SimCLR checkpoints.

XAI sanity checks:

- Trained-vs-random Grad-CAM similarity is low, supporting non-random attribution structure.
- Border masking can increase predicted PCOS probability in some cases, so artifact reliance and image framing remain important limitations to report.

## Discussion Notes

The evidence no longer supports a simple claim that SSL universally beats supervised transfer. The stronger claim is more nuanced and publishable: under low-cleanliness PCOS ultrasound conditions, SSL can become competitive with careful downstream tuning and threshold selection, but clean accuracy alone hides sensitivity to preprocessing, calibration, and possible artifact shortcuts.

## Limitations

- Single public dataset; external validation is still needed.
- pHash grouping is conservative but imperfect.
- XAI sanity checks are diagnostic, not clinical proof of anatomical reasoning.
- DINO/MAE remain optional extensions if we need ViT-style SSL; BYOL now covers a second SSL family.
- Some baselines are under-trained and should not be overinterpreted.

## Manuscript Assets

- Tables: `reports/manuscript_tables.md`
- Figure manifest: `reports/final_figures_manifest.csv`
- Multi-seed summary: `reports/seed_summary.csv`
- Threshold CIs: `reports/threshold_confidence_intervals.csv`
- Calibration improvement: `reports/calibration_improvement.csv`
- XAI sanity checks: `reports/xai_sanity_checks.csv`
