# Next Steps

Working queue for the PCOS SSL manuscript experiments.

## Immediate Priorities

1. Extend repeated seeds for the current supervised and SimCLR label-efficiency experiments.
   - Fractions completed with 3 seeds: 10% and 50% at 10 downstream epochs; 10% and 25% at one downstream epoch.
   - Remaining useful fractions: 5% and 25% with longer downstream tuning.
   - Target seeds: 5 if runtime allows.
   - Report mean, standard deviation, and confidence intervals.

2. Extend downstream fine-tuning beyond one epoch.
   - Start with ResNet-18 supervised and SimCLR e25 fine-tune at 10% and 50%.
   - Compared 1, 5, and 10 downstream epochs at 10% labels for supervised ResNet-18 and SimCLR e25.
   - Compared 1, 5, 10, and 25 downstream epochs at 50% labels for supervised ResNet-18 and SimCLR e25.
   - Watch calibration, not just accuracy/AUROC.

3. Extend validation-selected threshold analysis.
   - Current SSL fine-tuned models often have perfect specificity but weak sensitivity at the default 0.5 threshold.
   - Validation-selected thresholds have been added for balanced accuracy, Youden index, F1, and sensitivity-constrained settings.
   - Wilson confidence intervals have been added for balanced-accuracy-selected tuned accuracy, sensitivity, and specificity.
   - Raw validation/test predictions and AUROC/AUPRC bootstrap confidence intervals have been added for the main manuscript operating points.
   - Calibration plots and Brier score have been added.
   - Platt and temperature scaling experiments are complete for the main prediction exports.
   - Next: decide whether calibrated probabilities belong in the main paper or supplementary material.

4. Add more SSL methods.
   - BYOL as the next non-contrastive method.
   - DINO or MAE for ViT-style SSL if runtime permits.
   - Keep SimCLR as the reproducible baseline.

5. Strengthen robustness evaluation.
   - Severity sweeps are complete for blur, downsample, contrast, crop, and noise.
   - Summary tables now report mean and worst-case degradation across corruption severities.
   - Compared ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-Tiny, and SimCLR fine-tuned checkpoints.
   - Robustness plots have been generated.
   - Next: decide which severity table belongs in the main manuscript.

6. Expand explainability stability.
   - Grad-CAM grids and supervised-vs-SimCLR panels are generated.
   - Randomized-weight sanity checks and border-mask perturbation checks are complete for supervised ResNet-18 50% e5 and SimCLR 50% e10.
   - Next: add occlusion sensitivity or Eigen-CAM only if the paper needs a second XAI method.

## Current Interpretation

The first pass does not yet show SSL superiority. Supervised ImageNet-pretrained ResNet-18 remains very strong under the pHash near-duplicate-aware split, even at low label fractions. The useful manuscript angle is more subtle: strict leakage control, label-efficiency behavior, calibration failure modes, transformation-specific robustness, and explanation stability.

The 10% downstream epoch-sensitivity block shows SimCLR improves strongly with longer fine-tuning, but supervised ResNet-18 is currently best at 10 epochs. The 50% block shows supervised ResNet-18 saturates by 5 epochs, while SimCLR becomes competitive by 10 epochs but has less stable threshold-selected specificity at 25 epochs. Bootstrap AUROC/AUPRC intervals show ranking performance is nearly saturated for both strong supervised and SSL checkpoints. The robustness severity sweep shows ViT-Tiny is the strongest robustness model despite lower clean accuracy, while the SimCLR 50% e10 checkpoint is clean-competitive but not corruption-robust.

## Blockers To Resolve

- Decide whether to extend the multi-seed runner to 5%/25% longer downstream runs.
- Decide whether to include random image split only as a leakage demonstration.
- Decide which final figure set goes into main manuscript vs supplement.
