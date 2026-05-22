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

Validation-selected thresholding substantially changes the label-efficiency story. Using thresholds selected only on validation predictions, the 25% SimCLR fine-tuned model improves from 0.9138 to 0.9572 test accuracy and from 0.8466 to 0.9395 sensitivity. The 10% SimCLR model improves from 0.9308 to 0.9534 accuracy and from 0.8768 to 0.9171 sensitivity. This suggests one manuscript contribution should be an operating-point analysis: AUROC and default-threshold accuracy alone can understate clinically relevant low-label performance.

The first repeated-seed block uses seeds 42, 7, and 123 at 10% and 25% label fractions. At 10%, SimCLR e25 fine-tuning has a slight mean accuracy advantage over supervised ResNet-18 (0.9366 vs 0.9270 default accuracy; 0.9392 vs 0.9366 tuned accuracy), while AUROC is similar (0.9784 vs 0.9766). At 25%, supervised ResNet-18 remains stronger (0.9360 vs 0.9211 default accuracy; 0.9543 vs 0.9452 tuned accuracy). This supports a nuanced label-efficiency claim: SSL may help in the very low-label regime, but the effect is small and needs longer downstream tuning plus more seeds.

The first downstream epoch-sensitivity block changes the interpretation again. With 10% labels and seed 42, supervised ResNet-18 improves from 0.9427 accuracy / 0.9930 AUROC at one epoch to 0.9855 / 0.9989 at 10 epochs. SimCLR e25 fine-tuning improves from 0.9308 / 0.9711 to 0.9736 / 0.9975 over the same epoch range. This means SSL is not collapsed; it benefits from longer fine-tuning. But it also means the current best 10% result is supervised, so the manuscript should avoid claiming simple SSL superiority. The stronger and more defensible claim is about evaluation under low-cleanliness conditions: duplicate-aware splits, operating-point selection, repeated seeds, and robustness expose behavior that clean accuracy hides.

The 10-epoch SimCLR model has high tuned sensitivity (0.9922) but lower tuned specificity (0.9339), while the 10-epoch supervised model keeps both tuned sensitivity and specificity near 0.99. This creates a clinically meaningful operating-point comparison rather than a single-score leaderboard.

The 50% downstream epoch-sensitivity block adds another caution. Supervised ResNet-18 is essentially saturated by 5 epochs, reaching 0.9981 tuned accuracy with 0.9966 sensitivity and 1.0000 specificity. SimCLR e25 fine-tuning becomes competitive at 10 epochs, reaching 0.9924 tuned accuracy with 0.9955 sensitivity and 0.9885 specificity. At 25 epochs, SimCLR's default accuracy rises to 0.9943, but validation-selected tuned accuracy falls to 0.9843 because specificity drops to 0.9684. This supports a manuscript claim about operating-point reliability and threshold stability rather than simple SSL superiority.

Wilson 95% confidence intervals are now available for the locked validation-selected operating points. For example, supervised ResNet-18 at 50% labels and 5 epochs has tuned accuracy 0.9981 with 95% CI 0.9945-0.9994, while SimCLR at 50% labels and 10 epochs has tuned accuracy 0.9924 with 95% CI 0.9868-0.9957. These are binomial intervals for thresholded metrics; AUROC/AUPRC intervals are handled separately with raw prediction exports and paired bootstrap resampling.

Raw prediction export and bootstrap ranking intervals are now implemented for the main manuscript operating points. The 50% supervised ResNet-18 e5 model has AUROC 0.9995 with 95% bootstrap CI 0.9987-1.0000, while the 50% SimCLR e10 model has AUROC 0.9991 with 95% CI 0.9980-0.9999. This confirms that ranking performance is almost saturated for both once tuned, so the manuscript should not lean on AUROC differences alone.

The cross-model robustness results are more nuanced than the clean metrics. ResNet-18 is strongest on clean accuracy but collapses under blur/downsampling. ViT-Tiny has lower clean accuracy but is far more stable under blur and downsample transformations. This may become a major manuscript finding: architecture choice changes robustness even when clean AUROC is uniformly high.

The 25-epoch SimCLR 50% fine-tuned model improved Gaussian-noise accuracy compared with supervised ResNet-18 on the same pHash test split (0.8785 versus 0.7728), but it remained brittle to blur and downsampling (0.5639 and 0.5670 accuracy). This suggests the robustness claim should be transformation-specific rather than framed as SSL being broadly robust.

The robustness severity sweep strengthens the architecture-robustness argument. ViT-Tiny has the best mean corruption accuracy (0.9107) and lowest mean degradation (0.0314) despite lower clean accuracy than ResNet-18. ResNet-18 has the best clean accuracy among the first-epoch supervised baselines (0.9924) but mean degradation is 0.1942. The SimCLR e25 50% e10 checkpoint is clean-competitive (0.9704) but has the largest mean degradation (0.2307) and worst-case degradation (0.5324). This is a valuable negative result: SSL fine-tuning alone does not guarantee corruption robustness.

Calibration analysis adds another manuscript angle. The supervised ResNet-18 50% e5 checkpoint has the best Brier score (0.0043), while SimCLR 50% e25 has the best ECE (0.0049). However, SimCLR e25 has weaker validation-selected specificity than SimCLR e10, so calibration should be discussed alongside operating-point behavior rather than as an isolated score.

The cross-label pHash inspection is now concrete. All 38 cross-label pHash groups are excluded from the strict split, and several have minimum cross-label pHash distance 0. The largest conflicting group contains 219 images, with 189 infected and 30 noninfected. This supports a strong dataset-quality contribution: some public PCOS ultrasound labels are likely ambiguous, duplicated across labels, or visually indistinguishable under perceptual hashing.

Grad-CAM comparison panels have been generated for supervised ResNet-18 50% e5 and SimCLR ResNet-18 50% e10 on matched test images. These are useful for qualitative inspection, but they should remain a supporting analysis until randomized-weight sanity checks and artifact-mask checks are added.

The first one-epoch label-efficiency pass suggests supervised ImageNet-pretrained ResNet-18 is already strong with small label fractions, but low-label runs can be poorly calibrated. Repeated seeds and longer fine-tuning are needed before making final low-label claims.

## Important Caution

The pHash audit found 38 cross-label near-duplicate groups. These must be inspected before making strong claims. They may indicate visually similar images, preprocessing collisions, annotation ambiguity, or actual label noise.

## Planned Tables

1. Dataset audit table: counts, duplicates, near-duplicates, split sizes.
2. Baseline comparison: ResNet, EfficientNet, ConvNeXt, ViT across exact and pHash splits.
3. Label-efficiency table: supervised vs SimCLR at 5%, 10%, 25%, 50%, 100%.
4. Downstream epoch-sensitivity table: supervised vs SimCLR at 10% and 50% labels across 1, 5, 10, and 25 epochs where available.
5. Robustness table: clean, crop, contrast, blur, downsample, noise.
6. Threshold-selected operating-point table with Wilson confidence intervals.
7. Calibration table: AUROC, AUPRC, ECE, Brier score if added.
8. Bootstrap AUROC/AUPRC confidence interval table.
9. Robustness severity sweep table with mean and worst-case degradation.
10. Calibration table with Brier score and ECE.
11. Cross-label pHash near-duplicate inspection table.
12. Grad-CAM comparison panel.

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
