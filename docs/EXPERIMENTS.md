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

### Robustness Severity Sweep

All rows use the pHash near-duplicate-aware test split. Each model was evaluated on clean images and three severity levels for center crop, low contrast, high contrast, blur, downsample, and Gaussian noise. Full condition-level results are in `reports/robustness_severity_long.csv`.

| Model | Clean acc | Mean corruption acc | Worst corruption acc | Mean degradation | Worst degradation | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| ResNet-18 | 0.9924 | 0.7983 | 0.5620 | 0.1942 | 0.4305 | Excellent clean accuracy, large corruption penalty. |
| EfficientNet-B0 | 0.9635 | 0.8021 | 0.4795 | 0.1614 | 0.4840 | Strong clean result, but noise/downsample are severe failures. |
| ViT-Tiny/16 | 0.9421 | 0.9107 | 0.7778 | 0.0314 | 0.1643 | Best robustness profile despite lower clean accuracy. |
| ConvNeXt-Tiny | 0.8269 | 0.8154 | 0.6023 | 0.0116 | 0.2247 | Under-trained clean model, but degradation is small. |
| SimCLR ResNet-18 e25 50% fine-tune | 0.9704 | 0.7397 | 0.4380 | 0.2307 | 0.5324 | Competitive clean score, but weakest severity robustness among the main models. |

Condition-level interpretation: ViT-Tiny is strongest under blur and downsample severity sweeps. ResNet-18 and SimCLR ResNet-18 are particularly brittle under blur/downsample, and the SimCLR e25 50% fine-tune also has the weakest worst-case Gaussian-noise accuracy in this sweep. This shifts the robustness argument away from broad SSL robustness and toward architecture-sensitive robustness.

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

These intervals describe thresholded accuracy/sensitivity/specificity only. AUROC/AUPRC bootstrap intervals are reported separately below from raw prediction exports.

## Bootstrap AUROC/AUPRC Confidence Intervals

Raw validation/test prediction CSVs are exported under `reports/predictions/`. The intervals below use 2,000 paired bootstrap resamples on the test split.

| Run ID | AUROC (95% CI) | AUPRC (95% CI) |
|---|---:|---:|
| `resnet18_supervised_phash_10pct_e10` | 0.9989 (0.9967-1.0000) | 0.9994 (0.9984-1.0000) |
| `simclr_resnet18_phash_e25_finetune_10pct_e10` | 0.9975 (0.9947-0.9994) | 0.9985 (0.9971-0.9996) |
| `resnet18_supervised_phash_50pct_e5` | 0.9995 (0.9987-1.0000) | 0.9996 (0.9992-1.0000) |
| `simclr_resnet18_phash_e25_finetune_50pct_e10` | 0.9991 (0.9980-0.9999) | 0.9994 (0.9987-0.9999) |
| `simclr_resnet18_phash_e25_finetune_50pct_e25` | 0.9992 (0.9981-1.0000) | 0.9995 (0.9988-1.0000) |

Interpretation: ranking performance is very high for both supervised and SSL models after sufficient fine-tuning. The more meaningful separation for the manuscript is now threshold stability, calibration, and robustness under severity sweeps.

## Calibration and Brier Score

Calibration is computed from exported test predictions. The reliability plot is saved locally at `reports/figures/calibration_curves.png`.

| Run ID | Brier score | ECE | Interpretation |
|---|---:|---:|---|
| `resnet18_supervised_phash_50pct_e5` | 0.0043 | 0.0073 | Best Brier score among main operating points. |
| `simclr_resnet18_phash_e25_finetune_50pct_e25` | 0.0049 | 0.0049 | Best ECE, but weaker tuned specificity than SimCLR e10. |
| `resnet18_supervised_phash_10pct_e10` | 0.0128 | 0.0178 | Strong low-label supervised calibration. |
| `simclr_resnet18_phash_e25_finetune_50pct_e10` | 0.0215 | 0.0123 | Strong ranking and threshold metrics, weaker Brier score. |
| `simclr_resnet18_phash_e25_finetune_10pct_e10` | 0.0237 | 0.0486 | Needs calibration attention at 10% labels. |

## Cross-Label pHash Inspection

The 38 cross-label pHash near-duplicate groups were inspected and summarized in `reports/cross_label_phash_groups.csv` and `reports/cross_label_phash_examples.csv`. Example montage is saved locally at `reports/figures/cross_label_phash_examples.png`.

Key finding: all 38 groups are excluded from the strict pHash split, and several groups have minimum cross-label pHash distance `0`. This indicates visually near-identical, and in some cases perceptually identical, examples under conflicting labels. The largest group contains 219 images: 189 infected and 30 noninfected.

## XAI Comparison

Grad-CAM comparison panels were generated for supervised ResNet-18 50% e5 and SimCLR ResNet-18 50% e10 on matched pHash-aware test samples. Panels are saved locally under `reports/gradcam_xai_comparison/`; the tracked manifest is `reports/xai_comparison_manifest.csv`.

These panels are for manuscript inspection and qualitative auditing. They should be interpreted cautiously until sanity checks with randomized weights and artifact masks are added.

## Manuscript Tables

Current manuscript-ready tables are generated in `reports/manuscript_tables/` with a combined Markdown version at `reports/manuscript_tables.md`.

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

## 2026-05-22: XAI Sanity, Calibration Improvement, Multi-Seed Sweep, and Manuscript Assets

**Goal**

Move from promising experiments toward manuscript-grade evidence: sanity-check explanations, improve calibration analysis, repeat key 10-epoch low-label runs across seeds, and regenerate final tables/figures.

**What was added**

- `scripts/generate_xai_sanity_checks.py`
- `scripts/calibrate_predictions.py`
- `scripts/run_multiseed_experiments.py`
- `scripts/build_final_figures.py`
- `docs/MANUSCRIPT_DRAFT.md`

**Key outputs**

- `reports/xai_sanity_checks.csv`
- `reports/calibration_improvement.csv`
- `reports/multiseed_plan.csv`
- `reports/seed_summary.csv`
- `reports/final_figures_manifest.csv`
- `reports/manuscript_tables.md`

**Main results**

- 10% labels, 10 epochs, 3 seeds:
  - Supervised ResNet-18: default accuracy `0.9794 +/- 0.0075`, AUROC `0.9984 +/- 0.0009`.
  - SimCLR e25 fine-tune: default accuracy `0.9755 +/- 0.0019`, AUROC `0.9963 +/- 0.0011`.
- 50% labels, 10 epochs, 3 seeds:
  - Supervised ResNet-18: default accuracy `0.9931 +/- 0.0044`, AUROC `0.9991 +/- 0.0000`.
  - SimCLR e25 fine-tune: default accuracy `0.9673 +/- 0.0029`, AUROC `0.9986 +/- 0.0004`, validation-selected accuracy `0.9939 +/- 0.0025`.
- Platt scaling improved Brier score for several under-calibrated supervised/SimCLR 10-epoch models, but overcorrected some already strong checkpoints.
- Grad-CAM sanity checks showed low trained-vs-random similarity, but border masking can raise PCOS probability in some cases.

**Interpretation**

The strongest paper angle is not "SSL beats supervised." The stronger contribution is a reliability-centered protocol for low-cleanliness PCOS ultrasound classification: leakage-aware splits, low-label multi-seed comparison, threshold-selected clinical operating points, calibration, robustness severity, and XAI sanity checks.

## 2026-05-22: BYOL Second SSL Family

Goal: add a non-contrastive SSL family and compare it fairly against SimCLR under the same pHash split, ResNet-18 encoder, 10%/50% label fractions, three seeds, and 10 downstream epochs.

BYOL pretraining:

| Run | Method | Backbone | Split | Epochs | Final train loss | Notes |
|---|---|---|---|---:|---:|---|
| `byol_smoke` | BYOL | ResNet-18 | pHash train split | 1 batch | 2.0523 | MPS smoke test. |
| `byol_resnet18_phash_e25` | BYOL | ResNet-18 | pHash train split | 25 | 0.0606 | Stable loss, no collapse observed. |

Matched 10-epoch multi-seed downstream comparison:

| Method | Label budget | Seeds | Default acc mean +/- std | AUROC mean +/- std | Tuned acc mean +/- std | Tuned sensitivity mean | Tuned specificity mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supervised ResNet-18 | 10% | 3 | 0.9794 +/- 0.0075 | 0.9984 +/- 0.0009 | 0.9832 +/- 0.0149 | 0.9828 | 0.9837 |
| SimCLR e25 fine-tune | 10% | 3 | 0.9755 +/- 0.0019 | 0.9963 +/- 0.0011 | 0.9759 +/- 0.0080 | 0.9854 | 0.9636 |
| BYOL e25 fine-tune | 10% | 3 | 0.9507 +/- 0.0286 | 0.9942 +/- 0.0028 | 0.9675 +/- 0.0172 | 0.9690 | 0.9655 |
| Supervised ResNet-18 | 50% | 3 | 0.9931 +/- 0.0044 | 0.9991 +/- 0.0000 | 0.9931 +/- 0.0044 | 0.9966 | 0.9885 |
| SimCLR e25 fine-tune | 50% | 3 | 0.9673 +/- 0.0029 | 0.9986 +/- 0.0004 | 0.9939 +/- 0.0025 | 0.9951 | 0.9923 |
| BYOL e25 fine-tune | 50% | 3 | 0.9734 +/- 0.0173 | 0.9981 +/- 0.0004 | 0.9899 +/- 0.0045 | 0.9948 | 0.9837 |

Main seed-42 BYOL operating points:

| Run | Default acc | AUROC | Tuned acc | Tuned sensitivity | Tuned specificity | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| `byol_resnet18_phash_e25_finetune_10pct_e10` | 0.9836 | 0.9972 | 0.9818 | 0.9765 | 0.9885 | 0.0258 |
| `byol_resnet18_phash_e25_finetune_50pct_e10` | 0.9818 | 0.9985 | 0.9937 | 0.9955 | 0.9914 | 0.0120 |

Interpretation: BYOL gives a real second SSL family but does not reverse the main conclusion. Supervised ResNet-18 remains the strongest and most stable default baseline. SimCLR is more stable than BYOL at 10% labels in this three-seed block, while BYOL is competitive at 50% labels and has a strong calibrated operating point after Platt scaling. The manuscript should present SSL as useful but reliability-sensitive, not as a universal replacement for supervised transfer.
