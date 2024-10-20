
import os

os.environ['OMP_NUM_THREADS'] = '1'
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from matplotlib import pyplot as plt
import submitit
from sklearn.metrics import mean_squared_error
import ConfigSpace as CS
from jahs_bench.api import Benchmark
from jahs_bench.lib.core.configspace import joint_config_space
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.models.rf_skopt.rf_skopt import RFSKOPT
from hebo.models.rf.rf import RF
from copy import copy
from hebo.models.et.et import ET
from hebo.models.xgb.xgb import GB
from hebo.models.gp.gp import GP
from functools import partial
from hebo.models.ensemble.ensemble import weights_history
from hebo.models.ensemble.ensemble import AdaptiveWeightModel


class TargetFunction:
    def __init__(self, dataset, seed) -> None:
        self.bench = Benchmark(
            task=dataset,
            save_dir='jahs_bench_data',
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

    def _transform_space(self, cs: CS.ConfigurationSpace):
            # Create hebo configspace
            hebo_dc = []
            for hp in cs.get_hyperparameters():
                param = None
                if isinstance(hp, CS.OrdinalHyperparameter):
                    # HEBO does not handle ordinal hyperparameter, make them int
                    param = {"name": hp.name, 'type': 'int', 'lb':0, 'ub': len(hp.sequence) - 1}
                elif isinstance(hp, CS.UniformIntegerHyperparameter):
                    if hp.log:
                        param = {"name": hp.name, 'type': 'pow_int', 'lb': hp.lower, 'ub': hp.upper}
                    else:
                        param = {"name": hp.name, 'type': 'int', 'lb': hp.lower, 'ub': hp.upper}
                elif isinstance(hp, CS.UniformFloatHyperparameter):
                    if hp.log:
                        param = {"name": hp.name, 'type': 'pow', 'lb': hp.lower, 'ub': hp.upper}
                    else:
                        param = {"name": hp.name, 'type': 'num', 'lb': hp.lower, 'ub': hp.upper}
                elif isinstance(hp, CS.CategoricalHyperparameter):
                    param = {"name": hp.name, 'type': 'cat', 'categories': hp.choices}
                elif isinstance(hp, CS.Constant):
                    self.const_hps.update({hp.name: hp.value})
                    continue
                else:
                    raise NotImplementedError("Unknown Parameter Type", hp)
                hebo_dc.append(param)
            return hebo_dc

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._transform_space(self.cs)

    def train(self, config: Configuration) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        config = copy(config.to_dict())
        for h in self.cs.get_hyperparameters():
            if isinstance(h, CS.OrdinalHyperparameter):
                config[h.name] = h.sequence[int(config[h.name])]
            elif isinstance(h, CS.UniformIntegerHyperparameter):
                config[h.name] = int(np.rint(config[h.name]))

        config.update(self.const_hps)
        self.configs.append(config)

        change = True
        conds = self.cs.get_conditions()
        while change:
            change = False
            to_remove = []
            for cond in conds:
                if isinstance(cond, (CS.InCondition, CS.EqualsCondition)):
                    parent = cond.parent
                    try:
                        values = cond.values
                    except:
                        values = [cond.value]
                    child = cond.child

                    if (parent.name not in config or config[parent.name] not in values) and child.name in config:
                        del config[child.name]
                        change = True

                elif isinstance(cond, CS.AndConjunction):
                    componenets = cond.components
                    for comp in componenets:
                        conds.append(comp)
                    to_remove.append(cond)
                conds = [c for c in conds if c not in to_remove]

        res = self.bench(config, nepochs=self.nepochs)
        v = res[self.nepochs]['valid-acc']
        v = -1.0 * v

        self.evals.append(v)
        self.infos.append(res)
        return np.array([v], dtype=float)

def run(dataset , seed, surrogate_model):
    model = TargetFunction(dataset, seed)
    if surrogate_model in ['rf_skopt', 'et', 'gb', 'rf']:
        model_config = dict(n_estimators=10)
        surrogate_model_in = surrogate_model
    elif surrogate_model == 'ens':
        model_config = dict(alpha=1.0)
        surrogate_model_in = 'ens'
    elif surrogate_model == 'ens05':
        model_config = dict(alpha=0.5)
        surrogate_model_in = 'ens'
    elif surrogate_model == 'ens01':
        model_config = dict(alpha=0.1)
        surrogate_model_in = 'ens'
    elif surrogate_model == 'ens09':
        model_config = dict(alpha=0.9)
        surrogate_model_in = 'ens'
    else:
        model_config = None
        surrogate_model_in = surrogate_model
    opt = HEBO(space=DesignSpace().parse(model.configspace), scramble_seed=seed, model_name=surrogate_model_in, model_config=model_config, rand_sample=2 * len(model.configspace))
    total_hebo_time = 0
    hebo_times = []
    start = time.process_time()
    for i in range(128):
        print(i)
        hebo_start = time.process_time()
        try:
            rec = opt.suggest(n_suggestions=8)
        except Exception as e:
            # raise
            np.save(f'hebo8_2d_d{dataset}sm{surrogate_model}s{seed}_err.npy', e)
            # return
        hebo_end = time.process_time()
        total_hebo_time += hebo_end - hebo_start
        hebo_times.append(hebo_end - hebo_start)
        obss = []
        for index, row in rec.iterrows():
            obss.append(model.train(row))
        opt.observe(rec, np.array(obss, dtype=float))
    end = time.process_time()
    # save_callback.save_validation_ti
    # import pdb; pdb.set_trace()
    np.save(f'hebo8_2d_d{dataset}sm{surrogate_model}s{seed}_evals.npy', model.evals)
    np.save(f'hebo8_2d_d{dataset}sm{surrogate_model}s{seed}_configs.npy', model.configs)
    np.save(f'hebo8_2d_d{dataset}sm{surrogate_model}s{seed}_infos.npy', model.infos)
    np.save(f'hebo8_2d_d{dataset}sm{surrogate_model}s{seed}_time.npy',
             np.array([start, end, total_hebo_time, hebo_times], dtype=object))

    global weights_history
    np.save(f'hebo8_2d_d{dataset}sm{surrogate_model}s{seed}_weights.npy', np.array(AdaptiveWeightModel.get_weights_history(), dtype=object))


if __name__ == "__main__":
    executor = submitit.AutoExecutor('log', 'slurm')
    executor.update_parameters(timeout_min=60 * 48, slurm_partition="CLUSTER", slurm_array_parallelism=1024, cpus_per_task=1, mem_gb=14, job_name='JAHS')

    with executor.batch():
        for dataset in ['cifar10', 'colorectal_histology', 'fashion_mnist']:
            for surrogate_model in ['gp','ens', 'ens01', 'ens05', 'rf', 'et', 'gb', 'ens09']:
                for seed in range(51):
                    j = executor.submit(run, dataset, seed, surrogate_model)
