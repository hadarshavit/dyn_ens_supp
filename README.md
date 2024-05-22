# Dynamic Ensembling of Surrogate Models for Hyperparameter Optimisation

This directory contains the source code for the paper "Dynamic Ensembling of Surrogate Models for Hyperparameter Optimisation" submitted to NeurIPS 2024. This is an anonymised repository.

## HEBO
To use our method with HEBO, simply set `model_name='ens'` in the HEBO constructor. To set the $\alpha$ value, also set the `model_config=dict(alpha=0.9)`.

The code to run our experiments is available at `yahpo_hebo.py` and `jahs_hebo.py` and is based on the submitit package.
We used python 3.11 for the experiments with YAHPO and 3.10 for JAHS.

The source code for the ensemble is in `HEBO/hebo/models/ensemble/ensemble.py`

##  SMAC
To use our method with SMAC import `from smac.model.ensemble import Ensemble` in and set `model=Ensemble(scenario, alpha)`.

The code to run our experiments is available at `yahpo_smac.py` and is based on the submitit package.
We used python 3.11 for the experiments.

The source code for the ensemble is in `SMAC3/smac/model/ensemble.py`

## Additional figures
In the figs directory there are additional figures.

The jahs subdirectory contains figures for JAHS, and yahpo for YAHPO.

The co subdirectory contains cnvergence plots aggregated over 51 seeds, weights contains the weights of the ensembles aggregated over 51 seeds.

Convergence plots that end with _smac are for the SMAC results, the others are for HEBO.# dyn_ens_supp
