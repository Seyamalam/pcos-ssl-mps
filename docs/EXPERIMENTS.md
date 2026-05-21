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
| `resnet18_supervised_phash_25pct_e1` | ResNet-18 | pHash near-duplicate-aware | 1 | 0.25 | 0.9320 | 0.9937 | 0.9356 | 0.0739 | Slightly below 10% in thresholded accuracy; needs repeated seeds. |
| `resnet18_supervised_phash_50pct_e1` | ResNet-18 | pHash near-duplicate-aware | 1 | 0.50 | 0.9843 | 0.9992 | 0.9858 | 0.0278 | Approaches full-label performance after one epoch. |

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
| `simclr_resnet18_phash_e25_finetune_25pct_e1` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.25 | 1 | 0.9138 | 0.9878 | 0.9169 | 0.0847 | Below supervised 25% in this first pass. |
| `simclr_resnet18_phash_e25_finetune_50pct_e1` | SimCLR ResNet-18, 25 epochs | Full fine-tune | pHash near-duplicate-aware | 0.50 | 1 | 0.9484 | 0.9893 | 0.9519 | 0.0577 | Stronger than short SSL, but below supervised 50%. |

## Label-Efficiency Comparison

All rows below use the pHash near-duplicate-aware split and one downstream epoch.

| Label fraction | Supervised ResNet-18 acc | Supervised ResNet-18 AUROC | SimCLR e25 fine-tune acc | SimCLR e25 fine-tune AUROC | Current read |
|---:|---:|---:|---:|---:|---|
| 0.05 | 0.8106 | 0.9734 | 0.5595 | 0.9367 | SSL rank signal exists, but threshold is not useful yet. |
| 0.10 | 0.9427 | 0.9930 | 0.9308 | 0.9711 | SSL accuracy is close; supervised has stronger ranking. |
| 0.25 | 0.9320 | 0.9937 | 0.9138 | 0.9878 | SSL trails but is not collapsed. |
| 0.50 | 0.9843 | 0.9992 | 0.9484 | 0.9893 | Supervised is stronger in the first sweep. |

## Failed or Non-Paper Runs

| Date | Run | What happened | Decision |
|---|---|---|---|
| 2026-05-21 | Tiny supervised smoke tests | Validation subsets contained one class due to ordered first batches. | Keep as runtime tests only; never report as model performance. |
| 2026-05-21 | Robustness with persistent workers | DataLoader cleanup hung after repeated transformed test passes. | Use `persistent_workers=False`; use `num_workers=0` for robustness scripts when reliability matters. |
