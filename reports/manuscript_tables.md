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

## Multi-Seed Summary

| method | label_fraction | epochs | n_seeds | seeds | accuracy_mean | accuracy_std | auroc_mean | auroc_std | selected_accuracy_mean | selected_accuracy_std | selected_sensitivity_mean | selected_specificity_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simclr_e25_finetune | 10pct | 1 | 3 | 7 42 123 | 0.9366477868680512 | 0.0213816748618202 | 0.9783844546558376 | 0.0072278116455368 | 0.9391650933501152 | 0.0176361229762815 | 0.920119447555058 | 0.9636015325670496 |
| supervised_resnet18 | 10pct | 1 | 3 | 7 42 123 | 0.9269981120201384 | 0.0137158557065471 | 0.9765749357497436 | 0.015309013044013 | 0.9366477868680512 | 0.0249385975936356 | 0.904441955953714 | 0.977969348659004 |
| simclr_e25_finetune | 10pct | 5 | 1 | 42 | 0.9307740717432348 | 0.0 | 0.989335959120104 | 0.0 | 0.9559471365638766 | 0.0 | 0.93505039193729 | 0.9827586206896552 |
| supervised_resnet18 | 10pct | 5 | 1 | 42 | 0.9647577092511012 | 0.0 | 0.9982076430989432 | 0.0 | 0.9817495280050346 | 0.0 | 0.9832026875699889 | 0.9798850574712644 |
| simclr_e25_finetune | 10pct | 10 | 3 | 7 42 123 | 0.975456261799874 | 0.0018879798615482 | 0.9963048057904604 | 0.0011199240795052 | 0.9758758128802182 | 0.0080182567038456 | 0.9854423292273236 | 0.9636015325670498 |
| supervised_resnet18 | 10pct | 10 | 3 | 7 42 123 | 0.9794419970631424 | 0.0075080890153862 | 0.9983814083999434 | 0.0009232949116964 | 0.9832179567862388 | 0.0149103047148452 | 0.9828294139604332 | 0.9837164750957856 |
| simclr_e25_finetune | 25pct | 1 | 3 | 7 42 123 | 0.921124396895322 | 0.0064281743008367 | 0.9859389547480832 | 0.0019848382957967 | 0.9452485840151038 | 0.0103600236835458 | 0.9331840238895108 | 0.960727969348659 |
| supervised_resnet18 | 25pct | 1 | 3 | 7 42 123 | 0.9360184602475352 | 0.0058474343884062 | 0.9925162931785322 | 0.0044285003745776 | 0.9542689322425004 | 0.0245464274827943 | 0.9537140724150802 | 0.9549808429118776 |
| simclr_e25_finetune | 50pct | 1 | 1 | 42 | 0.948395217117684 | 0.0 | 0.9892909088568818 | 0.0 | 0.9597230962869728 | 0.0 | 0.9283314669652856 | 1.0 |
| supervised_resnet18 | 50pct | 1 | 1 | 42 | 0.9842668344870988 | 0.0 | 0.9991536986266104 | 0.0 | 0.9899307740717432 | 0.0 | 0.9820828667413214 | 1.0 |
| simclr_e25_finetune | 50pct | 5 | 1 | 42 | 0.9622404027690372 | 0.0 | 0.998127196200332 | 0.0 | 0.9823788546255506 | 0.0 | 0.9932810750279956 | 0.9683908045977012 |
| supervised_resnet18 | 50pct | 5 | 1 | 42 | 0.9949653870358716 | 0.0 | 0.999481921972944 | 0.0 | 0.998112020138452 | 0.0 | 0.9966405375139976 | 1.0 |
| simclr_e25_finetune | 50pct | 10 | 3 | 7 42 123 | 0.9672750157331654 | 0.0028839368753656 | 0.9986227490957768 | 0.0004173201765587 | 0.9939165093350116 | 0.0025433932563419 | 0.9951474430757744 | 0.9923371647509578 |
| supervised_resnet18 | 50pct | 10 | 3 | 7 42 123 | 0.9930774071743236 | 0.0043601027251577 | 0.9990786148545736 | 1.772267877638518e-05 | 0.9930774071743236 | 0.0043601027251577 | 0.9966405375139976 | 0.9885057471264368 |
| simclr_e25_finetune | 50pct | 25 | 1 | 42 | 0.9943360604153556 | 0.0 | 0.9991730058822772 | 0.0 | 0.9842668344870988 | 0.0 | 0.9966405375139976 | 0.9683908045977012 |
| supervised_resnet18 | 50pct | 25 | 1 | 42 | 0.9943360604153556 | 0.0 | 0.999070033852055 | 0.0 | 0.9905601006922592 | 0.0 | 0.9966405375139976 | 0.9827586206896552 |

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

