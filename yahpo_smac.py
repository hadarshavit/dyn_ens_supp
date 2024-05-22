import os

os.environ['OMP_NUM_THREADS'] = '1'
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt
import submitit
from sklearn.metrics import mean_squared_error
from smac.model.random_forest.skl_et import SKL_ET
from smac.model.random_forest.skl_rf import SKL_RF
from smac.model.random_forest.skl_gb import SKL_GB
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.model.ensemble import Ensemble
from smac.runhistory.dataclasses import TrialValue
import ConfigSpace as CS
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import BlackBoxFacade as BBFacade

from smac import RunHistory, Scenario
from copy import copy

from yahpo_gym import *


class TargetFunction:
    def __init__(self, scenario, instance, seed, bench) -> None:
        print(scenario, instance)
        self.bench = bench
        self.cs = self.bench.get_opt_space(drop_fidelity_params=True, seed=seed)
        self.max_fidelity = {h.name: h.upper for h in self.bench.get_fidelity_space().get_hyperparameters()}
        if 'rbv2' in scenario:
            self.obj_key = 'acc'
            self.minus1 = True
        elif 'iaml' in scenario:
            self.obj_key = 'mmce'
            self.minus1 = False
        elif scenario == 'nb301':
            self.obj_key = 'val_accuracy'
            self.minus1 = True
        elif scenario == 'fcnet':
            self.obj_key = 'valid_loss'
            self.minus1 = False
        elif scenario == 'lcbench':
            self.obj_key = 'val_accuracy'
            self.minus1 = True
        self.const_hps = {}
        self.seed = seed
        self.configs = []
        self.evals = []
        self.infos = []

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.cs

    def train(self, config: Configuration, seed) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        config = dict(config)
        config.update(self.max_fidelity)
        config.update(self.const_hps)
        self.configs.append(config)
        res = self.bench.objective_function(config, logging=False, multithread=False)[0]
        v = res[self.obj_key]

        if self.minus1:
            v = -1.0 * v

        self.evals.append(v)
        self.infos.append(res)
        return v

def run(scenario, instance, seed, surrogate_model, bench=None):
    if bench is None:
        bench = BenchmarkSet(scenario=scenario)
        bench.set_instance(instance)
    os.system(f'rm -rf /dev/shm/ms{scenario}i{instance}sm{surrogate_model}s{seed}')
    os.makedirs(f'/dev/shm/ms{scenario}i{instance}sm{surrogate_model}s{seed}')
    model = TargetFunction(scenario, instance, seed, bench)

    smac_scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=1024, 
                        seed=seed,
                        output_directory=f'/dev/shm/ms{scenario}i{instance}sm{surrogate_model}s{seed}')
    if surrogate_model == 'orig':
        smodel = None
    elif surrogate_model == 'gb':
        smodel = SKL_GB(model.configspace, seed=seed)
    elif surrogate_model == 'et':
        smodel = SKL_ET(model.configspace, seed=seed)
    elif surrogate_model == 'rf':
        smodel = SKL_RF(model.configspace, seed=seed)
    elif surrogate_model == 'gp':
        smodel = BBFacade.get_model(smac_scenario)
        smodel._n_restarts = 0
    elif surrogate_model == 'ens':
        smodel = Ensemble(smac_scenario,
        alpha=1.0)
    elif surrogate_model == 'ens05':
        smodel = Ensemble(smac_scenario,
        alpha=0.5)
    elif surrogate_model == 'ens01':
        smodel = Ensemble(smac_scenario,
        alpha=0.1)
    elif surrogate_model == 'ens09':
        smodel = Ensemble(smac_scenario,
        alpha=0.9)

    smac = HPOFacade(
        smac_scenario,
        model.train,
        overwrite=True,
        model=smodel,
        initial_design=HPOFacade.get_initial_design(smac_scenario, n_configs=2 * len(model.configspace))
    )
    smac._config_selector._retries = 1024 * 1024
    smac_times = []
    total_smac_time = 0
    start = time.process_time()
    smac_start = time.process_time()
    for t in range(smac_scenario.n_trials):
        trial = smac.ask()
        smac_end = time.process_time()
        total_smac_time += smac_end - smac_start
        smac_times.append(smac_end - smac_start)
        v = model.train(trial.config, trial.seed)
        smac_start = time.process_time()
        smac.tell(trial, TrialValue(v))
    smac_end = time.process_time()
    total_smac_time += smac_end - smac_start
    smac_times.append(smac_end - smac_start)
    end = time.process_time()

    np.save(f'smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_evals.npy', model.evals)
    np.save(f'smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_configs.npy', model.configs)
    np.save(f'smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_infos.npy', model.infos)
    np.save(f'smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_time.npy', np.array([start, end, total_smac_time, smac_times], dtype=object))
    if surrogate_model.startswith('ens'):
        np.save(f'smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_weights.npy', smodel.weights_history)
    os.system(f'rm -rf /dev/shm/ms{scenario}i{instance}sm{surrogate_model}s{seed}')


if __name__ == "__main__":
    executor = submitit.AutoExecutor('logs', 'slurm')
    executor.update_parameters(timeout_min=60 * 48, slurm_partition="PARTITION", slurm_array_parallelism=1024, cpus_per_task=1, mem_gb=14, job_name='YAHPO')
    
    i = 0
    for scenario in list_scenarios()[:]: 
            print(scenario)
            b = BenchmarkSet(scenario=scenario)

            for instance in b.instances:
                with executor.batch():
                
                    for surrogate_model in ['ens09', 'ens05', 'ens01', 'ens', 'gp', 'rf', 'et', 'gb']:
                        for seed in range(0, 51):
                            i += 1
                            j = executor.submit(run, scenario, instance, seed, surrogate_model, None)

