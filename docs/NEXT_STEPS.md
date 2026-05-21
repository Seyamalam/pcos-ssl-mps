# Next Steps

Working queue for the PCOS SSL manuscript experiments.

## Immediate Priorities

1. Run repeated seeds for the current supervised and SimCLR label-efficiency experiments.
   - Fractions: 5%, 10%, 25%, 50%.
   - Minimum seeds: 3.
   - Target seeds: 5 if runtime allows.
   - Report mean, standard deviation, and confidence intervals.

2. Extend downstream fine-tuning beyond one epoch.
   - Start with ResNet-18 supervised and SimCLR e25 fine-tune at 10% and 50%.
   - Compare 1, 5, 10, and 25 downstream epochs.
   - Watch calibration, not just accuracy/AUROC.

3. Tune decision thresholds on validation data.
   - Current SSL fine-tuned models often have perfect specificity but weak sensitivity at the default 0.5 threshold.
   - Add validation-selected thresholds for balanced accuracy, Youden index, and sensitivity-constrained settings.

4. Add more SSL methods.
   - BYOL as the next non-contrastive method.
   - DINO or MAE for ViT-style SSL if runtime permits.
   - Keep SimCLR as the reproducible baseline.

5. Strengthen robustness evaluation.
   - Add severity sweeps for blur, downsample, contrast, and noise.
   - Report robustness area-under-degradation-curve, not just one corruption severity.
   - Compare ResNet-18, EfficientNet-B0, ConvNeXt-Tiny, ViT-Tiny, and SSL fine-tuned checkpoints.

6. Add explainability stability.
   - Generate Grad-CAM grids for supervised ResNet-18, ViT-Tiny, and SimCLR fine-tuned ResNet-18.
   - Compare clean vs corrupted images.
   - Add sanity checks with randomized weights.

## Current Interpretation

The first pass does not yet show SSL superiority. Supervised ImageNet-pretrained ResNet-18 remains very strong under the pHash near-duplicate-aware split, even at low label fractions. The useful manuscript angle is more subtle: strict leakage control, label-efficiency behavior, calibration failure modes, transformation-specific robustness, and explanation stability.

## Blockers To Resolve

- Inspect the 38 cross-label pHash near-duplicate groups.
- Add validation-selected threshold evaluation.
- Add multi-seed experiment runner to avoid manual command loops.
- Decide whether to include random image split only as a leakage demonstration.
