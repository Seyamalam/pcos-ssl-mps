# Manuscript Tables

## Dataset Audit

| metric | value |
| --- | --- |
| Total readable images | 11784 |
| PCOS-positive images | 6784 |
| Healthy/non-PCOS images | 5000 |
| Exact duplicate groups | 1956 |
| Exact duplicate files beyond first | 7788 |
| pHash near-duplicate groups | 1788 |
| pHash near-duplicate files beyond first | 9448 |
| Cross-label pHash near-duplicate groups | 38 |
| pHash-aware train images | 7429 |
| pHash-aware validation images | 1584 |
| pHash-aware test images | 1589 |

## Baseline Architectures

| run_id | accuracy | auroc | auprc | f1 | ece |
| --- | --- | --- | --- | --- | --- |
| resnet18_supervised_phash_e1 | 0.9924480805538074 | 0.9993918214464996 | 0.9996081596666616 | 0.9932960893854748 | 0.0182895815760447 |
| efficientnet_b0_supervised_phash_e1 | 0.9634990560100692 | 0.997232626687776 | 0.9982072282595414 | 0.9680616740088106 | 0.026609779160801 |
| vit_tiny_patch16_224_supervised_phash_e1 | 0.9421019509125236 | 0.9986130954679436 | 0.9991158057814168 | 0.9456906729634004 | 0.0385569733127118 |
| convnext_tiny_supervised_phash_e1 | 0.8269351793580868 | 0.9565458032461932 | 0.9683515667055912 | 0.8226950354609929 | 0.0445585407673036 |

## Downstream Epoch Sensitivity

| method | label_fraction | downstream_epochs | default_accuracy | auroc | selected_accuracy | selected_sensitivity | selected_specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Supervised ResNet-18 | 0.1 | 1 | 0.942731 | 0.992991 | 0.965387 | 0.947368 | 0.988506 |
| Supervised ResNet-18 | 0.1 | 5 | 0.964758 | 0.998208 | 0.98175 | 0.983203 | 0.979885 |
| Supervised ResNet-18 | 0.1 | 10 | 0.985525 | 0.998874 | 0.991189 | 0.993281 | 0.988506 |
| Supervised ResNet-18 | 0.5 | 1 | 0.984267 | 0.999154 | 0.989931 | 0.982083 | 1.0 |
| Supervised ResNet-18 | 0.5 | 5 | 0.994965 | 0.999482 | 0.998112 | 0.996641 | 1.0 |
| Supervised ResNet-18 | 0.5 | 10 | 0.99056 | 0.99908 | 0.99056 | 0.996641 | 0.982759 |
| Supervised ResNet-18 | 0.5 | 25 | 0.994336 | 0.99907 | 0.99056 | 0.996641 | 0.982759 |
| SimCLR e25 fine-tune | 0.1 | 1 | 0.930774 | 0.97111 | 0.95343 | 0.917133 | 1.0 |
| SimCLR e25 fine-tune | 0.1 | 5 | 0.930774 | 0.989336 | 0.955947 | 0.93505 | 0.982759 |
| SimCLR e25 fine-tune | 0.1 | 10 | 0.973568 | 0.99748 | 0.966646 | 0.992161 | 0.933908 |
| SimCLR e25 fine-tune | 0.5 | 1 | 0.948395 | 0.989291 | 0.959723 | 0.928331 | 1.0 |
| SimCLR e25 fine-tune | 0.5 | 5 | 0.96224 | 0.998127 | 0.982379 | 0.993281 | 0.968391 |
| SimCLR e25 fine-tune | 0.5 | 10 | 0.970422 | 0.999083 | 0.992448 | 0.995521 | 0.988506 |
| SimCLR e25 fine-tune | 0.5 | 25 | 0.994336 | 0.999173 | 0.984267 | 0.996641 | 0.968391 |

## Threshold Operating Points With Wilson CIs

| run_id | selected_accuracy | selected_accuracy_ci_low | selected_accuracy_ci_high | selected_sensitivity | selected_sensitivity_ci_low | selected_sensitivity_ci_high | selected_specificity | selected_specificity_ci_low | selected_specificity_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18_supervised_phash_50pct_e5 | 0.998112020138452 | 0.994463631809454 | 0.999357724940646 | 0.9966405375139976 | 0.9901695220913032 | 0.9988568567029004 | 1.0 | 0.994510757862922 | 0.9999999999999998 |
| simclr_resnet18_phash_e25_finetune_50pct_e10 | 0.9924480805538074 | 0.9868460263781876 | 0.99567477167229 | 0.9955207166853304 | 0.988539545527058 | 0.998256785062424 | 0.9885057471264368 | 0.97748395138645 | 0.9941644902037612 |
| simclr_resnet18_phash_e25_finetune_50pct_e25 | 0.9842668344870988 | 0.9768769709043876 | 0.9893207979133029 | 0.9966405375139976 | 0.9901695220913032 | 0.9988568567029004 | 0.9683908045977012 | 0.9526046795436603 | 0.9790347085693069 |

## Bootstrap AUROC/AUPRC CIs

| run_id | auroc | auroc_ci_low | auroc_ci_high | auprc | auprc_ci_low | auprc_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18_supervised_phash_10pct_e10 | 0.9988737434194436 | 0.9967278328134248 | 0.9999566278603466 | 0.9994484161515635 | 0.99844242531996 | 0.9999678473215184 |
| resnet18_supervised_phash_50pct_e5 | 0.999481921972944 | 0.9987332250118386 | 1.0 | 0.9996483219465856 | 0.9991509280259266 | 1.0 |
| simclr_resnet18_phash_e25_finetune_10pct_e10 | 0.9974804031354984 | 0.9947222431285638 | 0.9993889092341676 | 0.9985376133176171 | 0.9970993870830348 | 0.9995820482690462 |
| simclr_resnet18_phash_e25_finetune_50pct_e10 | 0.9990829053558328 | 0.9979930795347656 | 0.9999144768807384 | 0.9993938917470722 | 0.9986941579619112 | 0.9999393774611464 |
| simclr_resnet18_phash_e25_finetune_50pct_e25 | 0.9991730058822772 | 0.9980846705970056 | 1.0 | 0.9994671328277482 | 0.998755930269215 | 1.0 |

## Calibration

| run_id | brier_score | ece |
| --- | --- | --- |
| resnet18_supervised_phash_50pct_e5 | 0.0042811421277036 | 0.0073174248024613 |
| simclr_resnet18_phash_e25_finetune_50pct_e25 | 0.0049134543553548 | 0.0048938289369891 |
| resnet18_supervised_phash_10pct_e10 | 0.0127593994833468 | 0.017786435360629 |
| simclr_resnet18_phash_e25_finetune_50pct_e10 | 0.0214944070983563 | 0.0123159457003971 |
| simclr_resnet18_phash_e25_finetune_10pct_e10 | 0.0236721530054331 | 0.0486331291079446 |

## Robustness Severity

| run_id | clean_accuracy | mean_corruption_accuracy | worst_corruption_accuracy | mean_accuracy_degradation | worst_accuracy_degradation |
| --- | --- | --- | --- | --- | --- |
| vit_tiny_patch16_224_phash_e1 | 0.9421019509125236 | 0.91067058247675 | 0.7778477029578351 | 0.0314313684357736 | 0.1642542479546884 |
| convnext_tiny_phash_e1 | 0.8269351793580868 | 0.8153625620585973 | 0.6022655758338578 | 0.0115726172994895 | 0.224669603524229 |
| efficientnet_b0_phash_e1 | 0.9634990560100692 | 0.8021467030277604 | 0.4795468848332284 | 0.1613523529823088 | 0.4839521711768408 |
| resnet18_phash_e1 | 0.9924480805538074 | 0.798265855534578 | 0.5619886721208307 | 0.1941822250192294 | 0.4304594084329767 |
| simclr_resnet18_phash_e25_finetune_50pct_e10 | 0.9704216488357458 | 0.7397035172365568 | 0.4380113278791693 | 0.2307181315991889 | 0.5324103209565765 |

## Cross-Label pHash Examples

| near_group_id | group_size | infected_count | noninfected_count | min_cross_label_phash_distance | representative_infected | representative_noninfected |
| --- | --- | --- | --- | --- | --- | --- |
| phash_001185 | 219 | 189 | 30 | 0 | PCOS/infected/image10444.jpg | PCOS/noninfected/Image_159.jpg |
| phash_001184 | 53 | 29 | 24 | 0 | PCOS/infected/image10510.jpg | PCOS/noninfected/Image_599.jpg |
| phash_001197 | 28 | 2 | 26 | 0 | PCOS/infected/image10648.jpg | PCOS/noninfected/Image_019.jpg |
| phash_001269 | 23 | 15 | 8 | 0 | PCOS/infected/image10459.jpg | PCOS/noninfected/Image_899.jpg |
| phash_001755 | 15 | 1 | 14 | 0 | PCOS/infected/image10445.jpg | PCOS/noninfected/Image_341.jpg |
| phash_001274 | 10 | 4 | 6 | 0 | PCOS/infected/image10920.jpg | PCOS/noninfected/Image_175.jpg |
| phash_001801 | 9 | 1 | 8 | 0 | PCOS/infected/image10493.jpg | PCOS/noninfected/Image_443.jpg |
| phash_001811 | 9 | 1 | 8 | 0 | PCOS/infected/image10503.jpg | PCOS/noninfected/Image_463.jpg |
| phash_001814 | 9 | 1 | 8 | 0 | PCOS/infected/image10507.jpg | PCOS/noninfected/Image_471.jpg |
| phash_001819 | 9 | 1 | 8 | 0 | PCOS/infected/image12278.jpg | PCOS/noninfected/Image_481.jpg |

