# Experiment Ledger

Every run should be recorded here, including smoke tests and failed attempts. This is the source of truth for manuscript tables.

## Dataset Artifacts

| Artifact | Path | Description | Status |
|---|---|---|---|
| Image audit | `metadata/images.csv` | One row per readable image, including class, size, MD5, and pHash. | Done |
| Exact duplicate groups | `metadata/duplicate_groups.csv` | MD5 duplicate groups with group sizes. | Done |
| Exact duplicate split | `metadata/splits_duplicate_aware.csv` | Train/val/test split with no exact duplicate group crossing splits. | Done |
| pHash near-duplicate groups | `metadata/near_duplicate_groups_phash.csv` | Near-duplicate groups using pHash Hamming distance <= 4. | Done |
| pHash near-duplicate split | `metadata/splits_near_duplicate_aware_phash.csv` | Strict split with cross-label near groups excluded. | Done |

## Dataset Audit Results

| Metric | Value |
|---|---:|
| Total images | 11,784 |
| PCOS-positive / `infected` | 6,784 |
| Healthy / `noninfected` | 5,000 |
| Unreadable images | 0 |
| Exact duplicate groups | 1,956 |
| Exact duplicate files beyond first | 7,788 |
| Cross-class exact duplicate groups | 0 |
| pHash near-duplicate groups | 1,788 |
| pHash near-duplicate files beyond first | 9,448 |
| Cross-class pHash near-duplicate groups | 38 |

## Split Results

| Split file | Train | Validation | Test | Notes |
|---|---:|---:|---:|---|
| `metadata/splits_duplicate_aware.csv` | 8,228 | 1,803 | 1,753 | Exact duplicate groups cannot cross splits. |
| `metadata/splits_near_duplicate_aware_phash.csv` | 7,429 | 1,584 | 1,589 | pHash near groups cannot cross splits; cross-label near groups excluded. |

## Supervised Baselines

| Run ID | Backbone | Split | Epochs | Label fraction | Test accuracy | AUROC | F1 | ECE | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `resnet18_supervised_exact_e1` | ResNet-18 | Exact duplicate-aware | 1 | 1.0 | 0.9960 | 0.9997 | 0.9964 | 0.0304 | Very high; must compare against stricter pHash split. |
| `resnet18_supervised_phash_e1` | ResNet-18 | pHash near-duplicate-aware | 1 | 1.0 | 0.9924 | 0.9994 | 0.9933 | 0.0183 | Still very high after stricter near-duplicate control. |
| `efficientnet_b0_supervised_phash_e1` | EfficientNet-B0 | pHash near-duplicate-aware | 1 | 1.0 | 0.9635 | 0.9972 | 0.9681 | 0.0266 | Strong AUROC; weaker thresholded accuracy than ResNet-18. |
| `vit_tiny_patch16_224_supervised_phash_e1` | ViT-Tiny/16 | pHash near-duplicate-aware | 1 | 1.0 | 0.9421 | 0.9986 | 0.9457 | 0.0386 | High AUROC, but conservative threshold behavior with perfect specificity and lower sensitivity. |
| `convnext_tiny_supervised_phash_e1` | ConvNeXt-Tiny | pHash near-duplicate-aware | 1 | 1.0 | 0.8269 | 0.9565 | 0.8227 | 0.0446 | Under-trained after one epoch; included as modern CNN comparison point. |
| `resnet18_supervised_phash_5pct_e1` | ResNet-18 | pHash near-duplicate-aware | 1 | 0.05 | 0.8106 | 0.9734 | 0.7973 | 0.2356 | Low-label supervised baseline; high AUROC but poor calibration. |
| `resnet18_supervised_phash_10pct_e1` | ResNet-18 | pHash near-duplicate-aware | 1 | 0.10 | 0.9427 | 0.9930 | 0.9463 | 0.2901 | Strong ranking; poor calibration after one epoch. |
| `resnet18_supervised_phash_10pct_e5` | ResNet-18 | pHash near-duplicate-aware | 5 | 0.10 | 0.9648 | 0.9982 | 0.9676 | 0.0189 | Longer downstream tuning improves ranking and calibration. |
| `resnet18_supervised_phash_10pct_e10` | ResNet-18 | pHash near-duplicate-aware | 10 | 0.10 | 0.9855 | 0.9989 | 0.9873 | 0.0188 | Strongest 10% label result so far. |
| `resnet18_supervised_phash_25pct_e1` | ResNet-18 | pHash near-duplicate-aware | 1 | 0.25 | 0.9320 | 0.9937 | 0.9356 | 0.0739 | Slightly below 10% in thresholded accuracy; needs repeated seeds. |
| `resnet18_supervised_phash_50pct_e1` | ResNet-18 | pHash near-duplicate-aware | 1 | 0.50 | 0.9843 | 0.9992 | 0.9858 | 0.0278 | Approaches full-label performance after one epoch. |
| `resnet18_supervised_phash_50pct_e5` | ResNet-18 | pHash near-duplicate-aware | 5 | 0.50 | 0.9950 | 0.9995 | 0.9955 | 0.0073 | Best 50% supervised checkpoint by tuned operating point. |
| `resnet18_supervised_phash_50pct_e10` | ResNet-18 | pHash near-duplicate-aware | 10 | 0.50 | 0.9906 | 0.9991 | 0.9916 | 0.0068 | Still excellent, but below the 5-epoch checkpoint. |
| `resnet18_supervised_phash_50pct_e25` | ResNet-18 | pHash near-duplicate-aware | 25 | 0.50 | 0.9943 | 0.9991 | 0.9950 | 0.0012 | Saturated; extra epochs do not improve selected-threshold result. |

## Robustness Results

Checkpoint: `runs/resnet18_supervised_exact_e1/best_model.pt`

| Test condition | Accuracy | AUROC | F1 | ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| Clean resize | 0.9960 | 0.9997 | 0.9964 | 0.0304 | Strong clean result. |
| Center crop | 0.7895 | 0.9788 | 0.8391 | 0.1372 | Large accuracy drop despite high AUROC. |
| Low contrast | 0.9606 | 0.9978 | 0.9656 | 0.0400 | Moderate drop. |
| High contrast | 0.9663 | 0.9995 | 0.9686 | 0.0166 | Moderate drop. |
| Blur | 0.5562 | 0.8800 | 0.7135 | 0.4153 | Severe preprocessing sensitivity. |
| Downsample | 0.5562 | 0.8887 | 0.7135 | 0.4096 | Severe preprocessing sensitivity. |
| Gaussian noise | 0.7427 | 0.9885 | 0.8112 | 0.1194 | Accuracy drop with preserved ranking. |

Checkpoint: `runs/resnet18_supervised_phash_e1/best_model.pt`

| Test condition | Accuracy | AUROC | F1 | ECE | Interpretation |
|---|---:|---:|---:|---:|---|
| Clean resize | 0.9924 | 0.9994 | 0.9933 | 0.0183 | Strong pHash split result. |
| Center crop | 0.7590 | 0.9864 | 0.8229 | 0.1850 | Large accuracy drop. |
| Low contrast | 0.9717 | 0.9992 | 0.9753 | 0.0332 | Moderate drop. |
| High contrast | 0.9887 | 0.9987 | 0.9898 | 0.0142 | Stable. |
| Blur | 0.5645 | 0.9533 | 0.7207 | 0.3964 | Severe preprocessing sensitivity. |
| Downsample | 0.5670 | 0.9575 | 0.7219 | 0.3952 | Severe preprocessing sensitivity. |
| Gaussian noise | 0.7728 | 0.9816 | 0.8319 | 0.1256 | Accuracy drop with preserved ranking. |

### Cross-Model Robustness Snapshot

All models below use the pHash near-duplicate-aware test split.

| Model | Clean acc | Center crop acc | Blur acc | Downsample acc | Noise acc | Key observation |
|---|---:|---:|---:|---:|---:|---|
| ResNet-18 | 0.9924 | 0.7590 | 0.5645 | 0.5670 | 0.7728 | Excellent clean accuracy, severe blur/downsample sensitivity. |
| EfficientNet-B0 | 0.9635 | 0.8093 | 0.6589 | 0.7017 | 0.5746 | Better downsample than ResNet, but very weak under Gaussian noise. |
| ViT-Tiny/16 | 0.9421 | 0.9182 | 0.9836 | 0.9899 | 0.9239 | Lower clean accuracy, strongest robustness profile. |
| ConvNeXt-Tiny | 0.8269 | 0.8074 | 0.8609 | 0.8571 | 0.8125 | Under-trained but comparatively stable. |
| SimCLR ResNet-18 10% linear probe | 0.6388 | 0.6419 | 0.5437 | 0.5538 | 0.6973 | Too weak for final claims; pipeline validation only. |
| SimCLR ResNet-18 e25 50% fine-tune | 0.9484 | 0.9402 | 0.5639 | 0.5670 | 0.8785 | Better noise robustness than supervised ResNet, but still brittle to blur/downsample. |

## SSL Runs

| Run ID | Method | Backbone | Split | Epochs | Loss | Downstream result | Notes |
|---|---|---|---|---:|---:|---|---|
| `simclr_smoke` | SimCLR | ResNet-18 | Exact train split | 1 batch | 4.1280 | N/A | Runtime-only smoke test on MPS. |
| `simclr_resnet18_phash_e1` | SimCLR | ResNet-18 | pHash train split | 1 | 3.3903 | 10% linear probe AUROC 0.7781 | Pipeline-validating run only; needs 25-100 epochs. |
| `simclr_resnet18_phash_e25` | SimCLR | ResNet-18 | pHash train split | 25 | 1.7324 | Full fine-tune rows below | Longer pretraining reduced loss from 3.3903 to 1.7324. |

## SSL Downstream Results

| Run ID | Pretraining | Probe type | Split | Label fraction | Epochs | Test accuracy | AUROC | F1 | ECE | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `simclr_resnet18_phash_e1_linear_10pct_e1` | SimCLR ResNet-18, 1 epoch | Linear probe | pHash near-duplicate-aware | 0.1 | 1 | 0.6388 | 0.7781 | 0.5754 | 0.1055 | Not competitive; confirms longer SSL pretraining is needed. |
| `simclr_resnet18_phash_e25_finetune_05pct_e1` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.05 | 1 | 0.5595 | 0.9367 | 0.3554 | 0.2145 | AUROC improved vs short SSL, but thresholded recall is poor. |
| `simclr_resnet18_phash_e25_finetune_10pct_e1` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.10 | 1 | 0.9308 | 0.9711 | 0.9344 | 0.2250 | Accuracy approaches supervised 10%, but AUROC trails. |
| `simclr_resnet18_phash_e25_finetune_10pct_e5` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.10 | 5 | 0.9308 | 0.9893 | 0.9344 | 0.0567 | More downstream epochs improve ranking and calibration, but default thresholded accuracy remains flat. |
| `simclr_resnet18_phash_e25_finetune_10pct_e10` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.10 | 10 | 0.9736 | 0.9975 | 0.9759 | 0.0488 | Large downstream gain; still below supervised ResNet-18 10-epoch accuracy. |
| `simclr_resnet18_phash_e25_finetune_25pct_e1` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.25 | 1 | 0.9138 | 0.9878 | 0.9169 | 0.0847 | Below supervised 25% in this first pass. |
| `simclr_resnet18_phash_e25_finetune_50pct_e1` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.50 | 1 | 0.9484 | 0.9893 | 0.9519 | 0.0577 | Stronger than short SSL, but below supervised 50%. |
| `simclr_resnet18_phash_e25_finetune_50pct_e5` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.50 | 5 | 0.9622 | 0.9981 | 0.9652 | 0.0284 | Ranking improves sharply; default threshold still lags tuned operating point. |
| `simclr_resnet18_phash_e25_finetune_50pct_e10` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.50 | 10 | 0.9704 | 0.9991 | 0.9743 | 0.0197 | Best 50% SimCLR validation-selected operating point. |
| `simclr_resnet18_phash_e25_finetune_50pct_e25` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.50 | 25 | 0.9943 | 0.9992 | 0.9950 | 0.0088 | Default accuracy catches supervised, but tuned threshold sacrifices specificity. |

## Label-Efficiency Comparison

All rows below use the pHash near-duplicate-aware split and one downstream epoch.

| Label fraction | Supervised ResNet-18 acc | Supervised ResNet-18 AUROC | SimCLR e25 fine-tune acc | SimCLR e25 fine-tune AUROC | Current read |
|---:|---:|---:|---:|---:|---|
| 0.05 | 0.8106 | 0.9734 | 0.5595 | 0.9367 | SSL rank signal exists, but threshold is not useful yet. |
| 0.10 | 0.9427 | 0.9930 | 0.9308 | 0.9711 | SSL accuracy is close; supervised has stronger ranking. |
| 0.25 | 0.9320 | 0.9937 | 0.9138 | 0.9878 | SSL trails but is not collapsed. |
| 0.50 | 0.9843 | 0.9992 | 0.9484 | 0.9893 | Supervised is stronger in the first sweep. |

## Validation-Selected Threshold Results

Thresholds are selected on validation predictions and then applied unchanged to the test split. The table below uses the balanced-accuracy-selected threshold.

| Run ID | Label fraction | Selected threshold | Default acc | Tuned acc | Default sensitivity | Tuned sensitivity | Tuned specificity | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `resnet18_supervised_phash_5pct_e1` | 0.05 | 0.4349 | 0.8106 | 0.9062 | 0.6629 | 0.9406 | 0.8621 | 0.9734 |
| `resnet18_supervised_phash_10pct_e1` | 0.10 | 0.4580 | 0.9427 | 0.9654 | 0.8981 | 0.9474 | 0.9885 | 0.9930 |
| `resnet18_supervised_phash_10pct_e5` | 0.10 | 0.1783 | 0.9648 | 0.9818 | 0.9373 | 0.9832 | 0.9799 | 0.9982 |
| `resnet18_supervised_phash_10pct_e10` | 0.10 | 0.7567 | 0.9855 | 0.9912 | 0.9989 | 0.9933 | 0.9885 | 0.9989 |
| `resnet18_supervised_phash_25pct_e1` | 0.25 | 0.2482 | 0.9320 | 0.9547 | 0.8791 | 0.9351 | 0.9799 | 0.9937 |
| `resnet18_supervised_phash_50pct_e1` | 0.50 | 0.3242 | 0.9843 | 0.9899 | 0.9720 | 0.9821 | 1.0000 | 0.9992 |
| `resnet18_supervised_phash_50pct_e5` | 0.50 | 0.2517 | 0.9950 | 0.9981 | 0.9910 | 0.9966 | 1.0000 | 0.9995 |
| `resnet18_supervised_phash_50pct_e10` | 0.50 | 0.3283 | 0.9906 | 0.9906 | 0.9966 | 0.9966 | 0.9828 | 0.9991 |
| `resnet18_supervised_phash_50pct_e25` | 0.50 | 0.4344 | 0.9943 | 0.9906 | 0.9966 | 0.9966 | 0.9828 | 0.9991 |
| `simclr_resnet18_phash_e25_finetune_05pct_e1` | 0.05 | 0.3389 | 0.5595 | 0.8156 | 0.2161 | 0.7413 | 0.9109 | 0.9367 |
| `simclr_resnet18_phash_e25_finetune_10pct_e1` | 0.10 | 0.4211 | 0.9308 | 0.9534 | 0.8768 | 0.9171 | 1.0000 | 0.9711 |
| `simclr_resnet18_phash_e25_finetune_10pct_e5` | 0.10 | 0.3493 | 0.9308 | 0.9559 | 0.8768 | 0.9351 | 0.9828 | 0.9893 |
| `simclr_resnet18_phash_e25_finetune_10pct_e10` | 0.10 | 0.1947 | 0.9736 | 0.9666 | 0.9530 | 0.9922 | 0.9339 | 0.9975 |
| `simclr_resnet18_phash_e25_finetune_25pct_e1` | 0.25 | 0.1542 | 0.9138 | 0.9572 | 0.8466 | 0.9395 | 0.9799 | 0.9878 |
| `simclr_resnet18_phash_e25_finetune_50pct_e1` | 0.50 | 0.4668 | 0.9484 | 0.9597 | 0.9082 | 0.9283 | 1.0000 | 0.9893 |
| `simclr_resnet18_phash_e25_finetune_50pct_e5` | 0.50 | 0.1449 | 0.9622 | 0.9824 | 0.9328 | 0.9933 | 0.9684 | 0.9981 |
| `simclr_resnet18_phash_e25_finetune_50pct_e10` | 0.50 | 0.8938 | 0.9704 | 0.9924 | 0.9966 | 0.9955 | 0.9885 | 0.9991 |
| `simclr_resnet18_phash_e25_finetune_50pct_e25` | 0.50 | 0.1301 | 0.9943 | 0.9843 | 0.9966 | 0.9966 | 0.9684 | 0.9992 |

Interpretation: validation-selected thresholding materially changes the low-label comparison. It does not change AUROC, but it exposes that the default 0.5 threshold was suppressing sensitivity, especially for SSL fine-tuned models.

## Downstream Epoch Sensitivity

All rows use 10% labels, seed 42, the pHash near-duplicate-aware split, and ResNet-18. Tuned metrics use the balanced-accuracy-selected validation threshold.

| Method | Epochs | Default acc | AUROC | Tuned acc | Tuned sensitivity | Tuned specificity | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Supervised ImageNet ResNet-18 | 1 | 0.9427 | 0.9930 | 0.9654 | 0.9474 | 0.9885 | Strong low-label starting point. |
| Supervised ImageNet ResNet-18 | 5 | 0.9648 | 0.9982 | 0.9818 | 0.9832 | 0.9799 | Better sensitivity and calibration than one epoch. |
| Supervised ImageNet ResNet-18 | 10 | 0.9855 | 0.9989 | 0.9912 | 0.9933 | 0.9885 | Best 10% label model so far. |
| SimCLR e25 fine-tune | 1 | 0.9308 | 0.9711 | 0.9534 | 0.9171 | 1.0000 | Competitive accuracy but weaker ranking. |
| SimCLR e25 fine-tune | 5 | 0.9308 | 0.9893 | 0.9559 | 0.9351 | 0.9828 | AUROC improves; default threshold remains conservative. |
| SimCLR e25 fine-tune | 10 | 0.9736 | 0.9975 | 0.9666 | 0.9922 | 0.9339 | Strong sensitivity after tuning, but specificity tradeoff is larger. |

Interpretation: longer downstream fine-tuning is mandatory before making SSL claims. SimCLR catches up substantially by 10 epochs, but the supervised ResNet-18 baseline remains stronger in this 10% seed-42 comparison.

### 50% Downstream Epoch Sensitivity

All rows use 50% labels, seed 42, the pHash near-duplicate-aware split, and ResNet-18. Tuned metrics use the balanced-accuracy-selected validation threshold.

| Method | Epochs | Default acc | AUROC | Tuned acc | Tuned sensitivity | Tuned specificity | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Supervised ImageNet ResNet-18 | 1 | 0.9843 | 0.9992 | 0.9899 | 0.9821 | 1.0000 | Near saturated immediately. |
| Supervised ImageNet ResNet-18 | 5 | 0.9950 | 0.9995 | 0.9981 | 0.9966 | 1.0000 | Best 50% supervised operating point. |
| Supervised ImageNet ResNet-18 | 10 | 0.9906 | 0.9991 | 0.9906 | 0.9966 | 0.9828 | No gain over 5 epochs. |
| Supervised ImageNet ResNet-18 | 25 | 0.9943 | 0.9991 | 0.9906 | 0.9966 | 0.9828 | Saturated, with weaker selected specificity than e5. |
| SimCLR e25 fine-tune | 1 | 0.9484 | 0.9893 | 0.9597 | 0.9283 | 1.0000 | Under-tuned at one epoch. |
| SimCLR e25 fine-tune | 5 | 0.9622 | 0.9981 | 0.9824 | 0.9933 | 0.9684 | Sensitivity improves, specificity drops. |
| SimCLR e25 fine-tune | 10 | 0.9704 | 0.9991 | 0.9924 | 0.9955 | 0.9885 | Best 50% SimCLR operating point. |
| SimCLR e25 fine-tune | 25 | 0.9943 | 0.9992 | 0.9843 | 0.9966 | 0.9684 | Default threshold catches up, tuned specificity is weaker. |

Interpretation: with 50% labels, supervised ResNet-18 is already saturated and peaks by 5 epochs. SimCLR becomes highly competitive by 10 epochs, but the 25-epoch fine-tuned model shows threshold instability: default accuracy rises while the validation-selected operating point loses specificity.

## Threshold Confidence Intervals

Confidence intervals are Wilson 95% intervals for the locked validation-selected threshold operating points. Full output is in `reports/threshold_confidence_intervals.csv`.

| Run ID | Tuned acc (95% CI) | Tuned sensitivity (95% CI) | Tuned specificity (95% CI) |
|---|---:|---:|---:|
| `resnet18_supervised_phash_50pct_e5` | 0.9981 (0.9945-0.9994) | 0.9966 (0.9902-0.9989) | 1.0000 (0.9945-1.0000) |
| `simclr_resnet18_phash_e25_finetune_50pct_e10` | 0.9924 (0.9868-0.9957) | 0.9955 (0.9885-0.9983) | 0.9885 (0.9775-0.9942) |
| `simclr_resnet18_phash_e25_finetune_50pct_e25` | 0.9843 (0.9769-0.9893) | 0.9966 (0.9902-0.9989) | 0.9684 (0.9526-0.9790) |

These intervals describe thresholded accuracy/sensitivity/specificity only. AUROC/AUPRC confidence intervals still need raw prediction export or bootstrap support.

## Repeated-Seed Summary

Three-seed summary for the first manuscript-grade repeat block. Seeds are 42, 7, and 123. All runs use one downstream epoch, the pHash near-duplicate-aware split, and ResNet-18.

| Method | Label fraction | Accuracy mean | Accuracy std | AUROC mean | AUROC std | Tuned accuracy mean | Tuned accuracy std | Tuned sensitivity mean | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SimCLR e25 fine-tune | 0.10 | 0.9366 | 0.0214 | 0.9784 | 0.0072 | 0.9392 | 0.0176 | 0.9201 | Slight mean accuracy edge over supervised, but needs longer tuning and more seeds. |
| Supervised ResNet-18 | 0.10 | 0.9270 | 0.0137 | 0.9766 | 0.0153 | 0.9366 | 0.0249 | 0.9044 | Competitive and variable in low-label setting. |
| SimCLR e25 fine-tune | 0.25 | 0.9211 | 0.0064 | 0.9859 | 0.0020 | 0.9452 | 0.0104 | 0.9332 | Stable but below supervised at 25%. |
| Supervised ResNet-18 | 0.25 | 0.9360 | 0.0058 | 0.9925 | 0.0044 | 0.9543 | 0.0245 | 0.9537 | Stronger mean performance, threshold-selected results have higher variance. |

## Failed or Non-Paper Runs

| Date | Run | What happened | Decision |
|---|---|---|---|
| 2026-05-21 | Tiny supervised smoke tests | Validation subsets contained one class due to ordered first batches. | Keep as runtime tests only; never report as model performance. |
| 2026-05-21 | Robustness with persistent workers | DataLoader cleanup hung after repeated transformed test passes. | Use `persistent_workers=False`; use `num_workers=0` for robustness scripts when reliability matters. |
