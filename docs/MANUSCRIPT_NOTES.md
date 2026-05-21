# Manuscript Notes

Working title:

**Beyond Clean Data: Self-Supervised Representation Learning for PCOS Ultrasound Classification Under Duplicate-Aware Evaluation**

## Central Thesis

Public PCOS ultrasound benchmarks can make models appear clinically reliable when duplicate leakage and preprocessing artifacts are not controlled. Self-supervised learning should be evaluated not only by clean accuracy, but also by label efficiency, robustness, calibration, and explanation stability under duplicate-aware and near-duplicate-aware protocols.

## Contributions

1. A leakage-aware audit of the PCOS-XAI ultrasound dataset.
2. Exact duplicate and pHash near-duplicate split protocols.
3. Supervised CNN/ViT baselines under exact and pHash leakage control.
4. SimCLR pretraining and fine-tuning under low-label regimes.
5. Robustness and XAI analysis showing whether models learn ovarian morphology or dataset artifacts.

## Early Evidence

The first ResNet-18 baseline reached 99.6% test accuracy and 0.9997 AUROC after one epoch on the exact duplicate-aware split. On the stricter pHash near-duplicate-aware split, it still reached 99.2% test accuracy and 0.9994 AUROC after one epoch. This suggests that exact/near-duplicate leakage is not the only reason performance is high.

However, robustness testing revealed severe accuracy drops under blur and downsampling. This supports a core manuscript argument: clean test performance alone is insufficient for trustworthy PCOS ultrasound AI.

Initial architecture comparison on the pHash split shows ResNet-18 outperforming EfficientNet-B0 and ViT-Tiny after one epoch. ViT-Tiny preserves high AUROC but has lower thresholded sensitivity, suggesting calibration/threshold selection should be included in the analysis.

The first SimCLR run was intentionally short: one pretraining epoch and a 10% linear probe. Its AUROC was only 0.7781, which is not a failure of SSL yet. It establishes the pipeline and motivates longer SSL pretraining before comparing label efficiency.

The 25-epoch SimCLR run reduced training loss from 3.3903 to 1.7324 and produced a useful downstream encoder, but the first full fine-tuning sweep still trails the supervised ImageNet-pretrained ResNet-18 baseline at matched label fractions. At 10% labels, SSL fine-tuning reached 0.9308 accuracy and 0.9711 AUROC versus supervised ResNet-18 at 0.9427 accuracy and 0.9930 AUROC. At 50% labels, SSL reached 0.9484 accuracy and 0.9893 AUROC versus supervised 0.9843 accuracy and 0.9992 AUROC. This is an important negative/neutral result for the manuscript: SSL is not automatically better without tuning, even though it may help robustness or calibration under some conditions.

The cross-model robustness results are more nuanced than the clean metrics. ResNet-18 is strongest on clean accuracy but collapses under blur/downsampling. ViT-Tiny has lower clean accuracy but is far more stable under blur and downsample transformations. This may become a major manuscript finding: architecture choice changes robustness even when clean AUROC is uniformly high.

The 25-epoch SimCLR 50% fine-tuned model improved Gaussian-noise accuracy compared with supervised ResNet-18 on the same pHash test split (0.8785 versus 0.7728), but it remained brittle to blur and downsampling (0.5639 and 0.5670 accuracy). This suggests the robustness claim should be transformation-specific rather than framed as SSL being broadly robust.

The first one-epoch label-efficiency pass suggests supervised ImageNet-pretrained ResNet-18 is already strong with small label fractions, but low-label runs can be poorly calibrated. Repeated seeds and longer fine-tuning are needed before making final low-label claims.

## Important Caution

The pHash audit found 38 cross-label near-duplicate groups. These must be inspected before making strong claims. They may indicate visually similar images, preprocessing collisions, annotation ambiguity, or actual label noise.

## Planned Tables

1. Dataset audit table: counts, duplicates, near-duplicates, split sizes.
2. Baseline comparison: ResNet, EfficientNet, ConvNeXt, ViT across exact and pHash splits.
3. Label-efficiency table: supervised vs SimCLR at 5%, 10%, 25%, 50%, 100%.
4. Robustness table: clean, crop, contrast, blur, downsample, noise.
5. Calibration table: AUROC, AUPRC, ECE, Brier score if added.

## Planned Figures

1. Dataset audit flowchart.
2. Duplicate and near-duplicate examples, only if licensing permits local/private use.
3. Train/validation curves for supervised vs SSL fine-tuning.
4. Label-efficiency curve.
5. Robustness degradation bar chart.
6. Grad-CAM comparison panel for supervised vs SSL.

## Candidate Journals

- Artificial Intelligence in Medicine
- Computer Methods and Programs in Biomedicine
- IEEE Journal of Biomedical and Health Informatics
- Computers in Biology and Medicine
- Expert Systems with Applications
