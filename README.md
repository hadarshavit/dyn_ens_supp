# DyMBO: Dynamic Mixture of Surrogate Models for Hyperparameter Optimisation

This directory contains the source code for the paper "DyMBO: Dynamic Mixture of Surrogate Models for Hyperparameter Optimisation" submitted to ICML 2025. This is an anonymised repository.

## DyMBO
To use DyMBO implemented within the HEBO framework, simply set `model_name='ens'` in the HEBO constructor. To set the $\alpha$ value, also set the `model_config=dict(alpha=0.9)`.

The code to run our experiments is available at `yahpo_hebo.py` and `jahs_hebo.py` and is based on the submitit package.
We used python 3.11 for the experiments with YAHPO and 3.10 for JAHS.

The source code for the ensemble is in `DyMBO/hebo/models/ensemble/ensemble.py`


## Additional figures
In the figs directory there are additional figures.

The jahs subdirectory contains figures for JAHS, and yahpo for YAHPO.

The co subdirectory contains cnvergence plots aggregated over 51 seeds.


