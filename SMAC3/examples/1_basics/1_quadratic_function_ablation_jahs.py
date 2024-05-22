"""
Quadratic Function
^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize a quadratic function.

We use the black-box facade because it is designed for black-box function optimization.
The black-box facade uses a :term:`Gaussian Process<GP>` as its surrogate model.
The facade works best on a numerical hyperparameter configuration space and should not
be applied to problems with large evaluation budgets (up to 1000 evaluations).
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
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
from jahs_bench.api import Benchmark
from jahs_bench.lib.core.configspace import joint_config_space
# from hpobench.container.benchmarks.nas.nasbench_101 import NASCifar10ABenchmark, NASCifar10BBenchmark, NASCifar10CBenchmark
# from hpobench.container.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class QuadraticFunction:
    def __init__(self, dataset, seed) -> None:
        self.bench = Benchmark(
            task=dataset,
            save_dir='/home/shavit/smac_ensemble_sur/jahs_bench_201/jahs_bench_data',
            kind='surrogate',
            download=False
        )
        self.cs = joint_config_space
        self.nepochs = 200
        self.const_hps = {}
        self.seed = seed
        self.configs = []
        self.evals = []
        self.infos = []

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.cs

    def train(self, config: Configuration) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""

        res = self.bench(config, nepochs=self.nepochs)
        v = res[self.nepochs]['valid-acc']
        v = -1.0 * v

        self.evals.append(v)
        self.infos.append(res)
        return v

def run(sdataset , seed, surrogate_model):
    os.system(f'rm -rf /dev/shm/d{dataset}sm{surrogate_model}s{seed}')
    os.makedirs(f'/dev/shm/d{dataset}sm{surrogate_model}s{seed}')
    model = QuadraticFunction(dataset, seed)

    smac_scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=1024, 
                        seed=seed,
                        output_directory=f'/dev/shm/d{dataset}sm{surrogate_model}s{seed}')
    if surrogate_model == 'orig':
        smodel = None
    elif surrogate_model == 'gb':
        smodel = SKL_GB(model.configspace, seed=seed)
    elif surrogate_model == 'et':
        smodel = SKL_ET(model.configspace, seed=seed)
    elif surrogate_model == 'rf':
        smodel = SKL_RF(model.configspace, seed=seed)
    elif surrogate_model == 'gp':
        smac_scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=512, 
                        seed=seed,
                        output_directory=f'/dev/shm/d{dataset}sm{surrogate_model}s{seed}')
        smodel = BBFacade.get_model(smac_scenario)
    elif surrogate_model == 'gp_nrestarts':
        smodel = BBFacade.get_model(smac_scenario)
        smodel._n_restarts = 0
    elif surrogate_model == 'ens':
        smodel = Ensemble(smac_scenario,
        alpha=1.0, 
        n_restarts=10)
        smac_scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=512, 
                        seed=seed,
                        output_directory=f'd{dataset}sm{surrogate_model}s{seed}')
    elif surrogate_model == 'ens05':
        smodel = Ensemble(smac_scenario,
        alpha=0.5, 
        n_restarts=10)
        smac_scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=512, 
                        seed=seed,
                        output_directory=f'd{dataset}sm{surrogate_model}s{seed}')
    elif surrogate_model == 'ens01':
        smodel = Ensemble(smac_scenario,
        alpha=0.1, 
        n_restarts=10)
        smac_scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=512, 
                        seed=seed,
                        output_directory=f'd{dataset}sm{surrogate_model}s{seed}')
    elif surrogate_model == 'ens09':
        smodel = Ensemble(smac_scenario,
        alpha=0.9, 
        n_restarts=10)
        smac_scenario = Scenario(model.configspace, 
                        deterministic=True, 
                        n_trials=512, 
                        seed=seed,
                        output_directory=f'd{dataset}sm{surrogate_model}s{seed}')
    elif surrogate_model == 'ens_nr':
        smodel = Ensemble(smac_scenario,
        alpha=1.0, 
        n_restarts=0)
    elif surrogate_model == 'ens05_nr':
        smodel = Ensemble(smac_scenario,
        alpha=0.5, 
        n_restarts=0)
    elif surrogate_model == 'ens01_nr':
        smodel = Ensemble(smac_scenario,
        alpha=0.1, 
        n_restarts=0)
    elif surrogate_model == 'ens09_nr':
        smodel = Ensemble(smac_scenario,
        alpha=0.9, 
        n_restarts=0)
    # vals = np.load(f'/home/shavit/smac_ensemble_sur/hpobench_test_data/{model_name}_{task_id}.npz')
    # save_callback = ValidationCallback(
    #     Xtest=vals['x'],
    #     Ytest=vals['y'],
    #     metric=mean_squared_error,
    #     save_dir=f'/dev/shm/m{model_name}t{task_id}sm{surrogate_model}s{seed}sp{split}/vals')
    smac = HPOFacade(
        smac_scenario,
        model.train,
        overwrite=True,
        model=smodel,
        initial_design=HPOFacade.get_initial_design(smac_scenario, n_configs=2 * len(model.configspace))
        # callbacks=[save_callback]
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
    # save_callback.save_validation_ti
    np.save(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_d{dataset}sm{surrogate_model}s{seed}_evals.npy', model.evals)
    np.save(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_d{dataset}sm{surrogate_model}s{seed}_configs.npy', model.configs)
    np.save(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_d{dataset}sm{surrogate_model}s{seed}_infos.npy', model.infos)
    np.save(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_d{dataset}sm{surrogate_model}s{seed}_time.npy', np.array([start, end, total_smac_time, smac_times], dtype=object))
    if surrogate_model.startswith('ens'):
        np.save(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_d{dataset}sm{surrogate_model}s{seed}_weights.npy', smodel.weights_history)
    # os.system(f'tar -cf /mnt/nfs-extra/shavit/smac_output_hpobench_new/m{model_name}t{task_id}sm{surrogate_model}s{seed}sp{split}.tar /dev/shm/m{model_name}t{task_id}sm{surrogate_model}s{seed}sp{split}')
    # os.system(f'rm -rf /dev/shm/ms{scenario}i{instance}sm{surrogate_model}s{seed}')


def submit(scenario, instance, surrogate_model):
    bench = BenchmarkSet(scenario=scenario)
    bench.set_instance(instance)
    # for surrogate_model in ['gp', 'rf', 'gb', 'et', 'orig', 'ens', 'ens05', 'ens01']:#['gp', 'gp_nrestarts', 'rf', 'gb', 'et', 'orig']: # #
    for seed in range(0, 51):
        if os.path.exists(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_evals.npy'):
            continue
        if os.path.exists(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_err.npy'):
            continue
        try:
            run(scenario, instance, seed, surrogate_model, bench)
        except Exception as e:
            import traceback
            np.save(f'/home/shavit/smac_ensemble_sur/final_exps/yahpo/smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_err.npy', [e, traceback.format_exc()])
            traceback.print_exc()

if __name__ == "__main__":
    # b = BenchmarkSet(scenario='lcbench', instance='3945')
    # run('lcbench', '3945', 10, 'ens_nr', b)
    # exit()
    executor = submitit.AutoExecutor('/storage/work/shavit/logs/smac_ensemble_sur/smac_jahs', 'slurm')
    executor.update_parameters(timeout_min=60 * 48, slurm_partition="Kathleen", slurm_array_parallelism=1024, cpus_per_task=1, mem_gb=14, slurm_additional_parameters={'exclude': 'kathleencpu[16]'}, job_name='YAHPO')
    
    #     j = executor.submit(run, 'lcbench', '7593', 4, 'ens01')
    # j.result()
    i = 0
    for dataset in ['cifar10', 'colorectal_histology', 'fashion_mnist']:
                import subprocess
                out = subprocess.run('squeue -u shavit -h -t pending -r'.split(' '), stdout=subprocess.PIPE).stdout
                print(out.decode('utf-8').count('\n'), flush=True)
                while out.decode('utf-8').count('\n') > 5000:
                    time.sleep(100)
                    out = subprocess.run('squeue -u shavit -h -t pending -r'.split(' '), stdout=subprocess.PIPE).stdout
                    print(out.decode('utf-8').count('\n'), flush=True)
                with executor.batch():
                
                    for surrogate_model in ['ens09', 'ens05', 'ens01', 'ens', 'gp', 'rf', 'et', 'gb', 'gp_nrestarts', 'ens_nr', 'ens01_nr', 'ens05_nr', 'ens09_nr']: # #
                        # executor.submit(submit, scenario, instance, surrogate_model)
                        for seed in range(0, 51):
                            # if os.path.exists(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_evals.npy'):
                            #     continue
                            # if os.path.exists(f'/home/shavit/smac_ensemble_sur/final_exps/jahs/smac_2d_s{scenario}i{instance}sm{surrogate_model}s{seed}_err.npy'):
                            #     continue
                            i += 1
                            j = executor.submit(run,  dataset, seed, surrogate_model)
                            # j.result()
        # run('xgb', 3917, 'rf', False)
