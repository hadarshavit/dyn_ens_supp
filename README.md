
# DyMBO: Dynamic Mixture of Surrogate Models for Hyperparameter Optimisation

This directory contains the source code for the paper "DyMBO: Dynamic Mixture of Surrogate Models for Hyperparameter Optimisation" submitted to ICML 2025. This is an anonymised repository.

## DyMBO
In the DyMBO directory there is the source code of DyMBO, based on the HEBO package.

To use DyMBO implemented within the HEBO framework, simply set `model_name='ens'` in the HEBO constructor. To set the $\alpha$ value, also set the `model_config=dict(alpha=0.9)`.

The code to run our experiments is available at `yahpo_hebo.py` and `jahs_hebo.py` and is based on the submitit package.
We used python 3.11 for the experiments with YAHPO and 3.10 for JAHS.

The source code for the ensemble is in `DyMBO/hebo/models/ensemble/ensemble.py`


## Additional figures
In the figs directory there are additional figures.

The `jahs` subdirectory contains figures for JAHS, and `yahpo` for YAHPO.

The `co` subdirectory contains convergence plots aggregated over 51 seeds.

## Selected figures and captions

* Weight evolution (which surrogates are used and when):

In the following figures, we show raw weight evolution for different surrogate models (RF and GP) over the course of optimisation. 
On x-axis: number of function evaluations.
On y-axis: weight value in [0, 1], smoothed over 51 seeds.
Please note that in the folder `figs/weights` there are such plots for all DyMBO variants for different alpha values on different benchmarks. For all plots, we use a default weight initialisation scheme with the initial value of 1 assigned to RF and 0 to GP.

Paths to figures (glmnet benchmark [Binder et al., 2020], dataset ID 1067): 
alpha = 0.1: [`figs/weights/w_2d_init_siaml_glmneti1067_smens_ranking_loss_rf_gp_init_orf01_eval_all.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/weights/w_2d_init_siaml_glmneti1067_smens_ranking_loss_rf_gp_init_orf01_eval_all.png)
alpha = 0.5: [`figs/weights/w_2d_init_siaml_glmneti1067_smens_ranking_loss_rf_gp_init_orf05_eval_all.png`]((https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/weights/w_2d_init_siaml_glmneti1067_smens_ranking_loss_rf_gp_init_orf05_eval_all.png)
alpha = 0.9: [`figs/weights/w_2d_init_siaml_glmneti1067_smens_ranking_loss_rf_gp_init_orf09_eval_all.png`]((https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/weights/w_2d_init_siaml_glmneti1067_smens_ranking_loss_rf_gp_init_orf09_eval_all.png)

Caption: In all three plots, we see that the weight assignment shifts immediately to GP in the very beginning of the optimisation, suggesting that here GP might better approximate the target function. However, this trend changes over the course of optimisation, and this changes varies according to different value of alpha used. 
For alpha = 0.1, we heavily rely on the historical accuracy and put less emphasis on the newly assigned weights, resulting in smoother curves with a single drastic change at around 50% of the budget, where we shift from predominantly using GP to RF (in a symetric fashion).
For alpha = 0.5, we see a similar overall trend, with a larger number of sharp spikes at certain timesteps, suggesting that even within regions where GP better captures the local landscape of the target function, we resort to "activating" RF to further boost predictive accuracy, and vice versa.
For alpha = 0.9, we heavily rely on the newly assigned weights while putting less emphasis on the historical accuracy, and we see more frequent changes (sharp spikes), reflecting the fact that at each iteration ranking loss of surrogates (which itself determines the newly assigned weights) can drastically change.

Paths to figures (NAS Bench 301 [Zela et al., 2022]):
alpha = 0.1: [`figs/weights/w_2d_init_snb301iCIFAR10_smens_ranking_loss_rf_gp_init_orf01_eval_all.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/weights/w_2d_init_snb301iCIFAR10_smens_ranking_loss_rf_gp_init_orf01_eval_all.png)
alpha = 0.5: [`figs/weights/w_2d_init_snb301iCIFAR10_smens_ranking_loss_rf_gp_init_orf05_eval_all.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/weights/w_2d_init_snb301iCIFAR10_smens_ranking_loss_rf_gp_init_orf05_eval_all.png)
alpha = 0.9: [`figs/weights/w_2d_init_snb301iCIFAR10_smens_ranking_loss_rf_gp_init_orf09_eval_all.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/weights/w_2d_init_snb301iCIFAR10_smens_ranking_loss_rf_gp_init_orf09_eval_all.png)

Caption: Contrary to the figures in the example above, for a NAS scenario we see a consistent trend in all three plots, regardless of the alpha value used. After the beginning of the optimisation where RF influences the ensemble the most, the weight shifts towards GP starting from at around 90 function evaluations, and the ensemble remains only impacted by GP.

* Performance of DyMBO with other alpha values (dynamic ensembling vs selection-only):

In the following figures, we show convergence curves of DyMBO with different alpha values, ranging from 0.1 (highly relying on history of weights) to 0.9 (highly relying on new weights), with alpha = 1.0 as a selection-only mode.
On x-axis: fraction of the total optimisation budget (plot starts at 0.2 as the first 20% of the budget goes towards the initial design, i.e., random sampling).
On y-axis: regret aggregated over 51 seeds (lower is better).
Please note that those are some selected examples out of 857 figures in total.

Path to figure: [`figs/yahpo_alpha_ablation/co/co_8_init_sfcnetifcnet_parkinsons_telemonitoring_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_sfcnetifcnet_parkinsons_telemonitoring_time.png)
Caption: All DyMBO variants are competitive in the first half of the optimisation on the FCNet benchmark [Falkner et al., 2018] on parkinsons_telemonitoring dataset, with very slight performance differences among them. In the second half of optimisation (i.e., after 50% of the budget), DyMBO with alpha = 0.9 outperforms other variants, and is overtaken by DyMBO with alpha = 0.1 towards the very end of the optimisation (i.e., around 90% of the budget).

Path to figure: [`figs/yahpo_alpha_ablation/co/co_8_init_siaml_rangeri1067_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_siaml_rangeri1067_time.png)
Caption: DyMBO with alpha = 0.9 consistently outperforms other DyMBO variants throughout the entire optimisation on the iaml_ranger benchmark [Pfisterer et al., 2022] on the dataset ID 1067.

Paths to figures: [`figs/yahpo_alpha_ablation/co/co_8_init_slcbenchi34539_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_slcbenchi34539_time.png) and [`figs/yahpo_alpha_ablation/co/co_8_init_slcbenchi189909_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_slcbenchi189909_time.png)
Caption: DyMBO with alpha = 0.9 consistently outperforms other DyMBO variants throughout the entire optimisation on the LCBench benchmark [Zimmer et al., 2021] on the datasets ID 34539 and ID 189909.

Path to figure: [`figs/yahpo_alpha_ablation/co/co_8_init_slcbenchi167181_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_slcbenchi167181_time.png)
Caption: DyMBO with alpha = 0.5 (equally relying on the historical accuracy and the newly assigned weights) consistently outperforms other DyMBO variants throughout the entire optimisation on the LCBench benchmark [Zimmer et al., 2021] on the dataset ID 34539.

Path to figure: [`figs/yahpo_alpha_ablation/co/co_8_init_srbv2_aknni38_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_srbv2_aknni38_time.png)
Caption: DyMBO with alpha = 0.9 consistently outperforms other DyMBO variants throughout the entire optimisation on the rbv2_aknn benchmark [Binder et al., 2020] on the dataset ID 38.

Path to figure: [`figs/yahpo_alpha_ablation/co/co_8_init_srbv2_xgboosti54_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_srbv2_xgboosti54_time.png)
Caption: DyMBO with alpha = 0.9 (heavily relying on the newly assigned weights) outperforms other DyMBO variants in the first half of the optimisation, when it is taken over by DyMBO with alpha = 0.1 (heavily relying on historical accuracy) on the rbv2_xgboost benchmark [Binder et al., 2020] on the dataset ID 54.

Path to figure: [`figs/yahpo_alpha_ablation/co/co_8_init_srbv2_xgboosti41212_time.png`](https://anonymous.4open.science/r/dyn_ens_supp-D5C2/figs/yahpo_alpha_ablation/co/co_8_init_srbv2_xgboosti41212_time.png)
Caption: DyMBO with alpha = 0.1 strongly outperforms other DyMBO variants throughout the optimisation, with the exception of the phase around 30% of the budget where it is taken over by DyMBO with alpha = 0.5.
