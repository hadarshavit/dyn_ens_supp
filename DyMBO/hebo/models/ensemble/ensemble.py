import torch
from skopt.learning import GradientBoostingQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from torch import FloatTensor, LongTensor
import numpy as np
from sklearn.metrics import mean_squared_error

from ..base_model import BaseModel
from ..layers import OneHotTransform
from ..util import filter_nan
from functools import partial
from hebo.models.rf.rf import RF
from copy import copy
from hebo.models.et.et import ET
from hebo.models.xgb.xgb import GB
from hebo.models.gp.gp import GP

trained_models = {}
cnt = 0
weights = None
_fitted = False
weights_history = []
model_keys = []


def roll_col(X, shift):
    """
    Rotate columns to right by shift.
    """
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


def compute_ranking_loss(
    f_samps: np.ndarray,
    target_y: np.ndarray,
    target_model: bool,
) -> np.ndarray:
    """
    Compute ranking loss for each sample from the posterior over target points.
    """
    f_samps = f_samps.flatten()
    target_y = target_y.flatten()
    loss = 0
    for i in range(len(f_samps)):
        for j in range(len(f_samps)):
            if f_samps[i] < f_samps[j]:
                if not (target_y[i] < target_y[j]):
                    loss += 1
            else:
                if target_y[i] < target_y[j]:
                    loss += 1
    return loss
            


class AdaptiveWeightModel(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf) -> None:
        super().__init__(num_cont, num_enum, num_out, **conf)

        global weights, weights_history

        initialising = False
        if weights is None:
            weights = []
            initialising = True

        self.models_classes = []
        if self.conf.get('use_rf', True):
            self.models_classes.append(partial(RF, conf=dict(n_estimators=10)))
            if initialising:
                if self.conf.get('init_rf', True):
                    weights.append(1)
                else:
                    weights.append(0)
        if self.conf.get('use_et', False):
            self.models_classes.append(partial(ET, conf=dict(n_estimators=10)))
            if initialising:
                if self.conf.get('init_et', True):
                    weights.append(1)
                else:
                    weights.append(0)
        if self.conf.get('use_gb', False):
            self.models_classes.append(partial(GB, conf=dict(n_estimators=10)))
            if initialising:
                if self.conf.get('init_gb', True):
                    weights.append(1)
                else:
                    weights.append(0)
        if self.conf.get('use_gp', True):
            self.models_classes.append(partial(GP, conf={
                                                                'lr'           : 0.01,
                                                                'num_epochs'   : 100,
                                                                'verbose'      : False,
                                                                'noise_lb'     : 8e-4, 
                                                                'pred_likeli'  : False
                                                                }))
            if initialising:
                if self.conf.get('init_gp', False):
                    weights.append(1)
                else:
                    weights.append(0)

        if initialising:
            weights = np.array(weights)
            weights_history = [weights.copy()]
            
        self.alpha = self.conf.get('alpha')
        self.prevX = None
        self.prevY = None
        self.mse_weights = False
        self.evaluate_all = self.conf.get('evaluate_all', True)
        self.use_total_variance = self.conf.get('use_total_variance', False)
        self.use_ranking_loss = self.conf.get('use_ranking_loss', True)
        self.est_noise = torch.zeros(self.num_out)
        self.conf = conf
        
    def fit(self, Xc : torch.Tensor, Xe : torch.Tensor, y : torch.Tensor):
        print('FIT ENS8')
        global _fitted, cnt, weights_history, weights, model_keys, trained_models
        if _fitted:
            mses = []
            if self.evaluate_all:
                newXc = Xc
                newXe = Xe
                newY = y
            else:
                newXc = Xc[-8:]
                newXe = Xe[-8:]
                newY = y[-8:]

            for i, (model_name, w) in enumerate(zip(model_keys, weights)):
                m = trained_models[model_name]
                if m is None:
                    mses.append(100000)
                    continue
                model, mc = m
                try:
                    means, vars = model.predict(newXc, newXe)
                except:
                    mses.append(100000)
                    continue
                if not self.use_ranking_loss:
                    mses.append(mean_squared_error(newY, means.detach().numpy()))
                else:
                    mses.append(compute_ranking_loss(newY, means.detach(), False))

            mses = np.array(mses)
            if not self.mse_weights:
                new_weights = np.zeros_like(weights)
                new_weights[np.argmin(mses)] = 1

            else:
                mses = 1 / np.power(mses, 2)
                normalised_mses = mses / mses.sum()
                new_weights = normalised_mses

            if cnt == 1:
                weights = new_weights
            else:
                weights = (1 - self.alpha) * weights + self.alpha * new_weights
            weights_history.append(weights.copy())
        
        trained_models = {}
        model_keys = []
        int_c = 0
        self.est_noise = torch.zeros(self.num_out)
        for mc in self.models_classes:
            try:
                model = mc(self.num_cont, self.num_enum, self.num_out, **self.conf)
                model.fit(Xc, Xe, y)
                trained_models[f"{model.__class__.__name__}{cnt}_{int_c}"] = (model, mc)
                model_keys.append(f"{model.__class__.__name__}{cnt}_{int_c}")
                self.est_noise += model.noise * weights[int_c]
                int_c += 1
            except:
                trained_models[f"{model.__class__.__name__}{cnt}_{int_c}"] = None
                model_keys.append(f"{model.__class__.__name__}{cnt}_{int_c}")
                weights[int_c] = 0
                if np.count_nonzero(weights) == 0:
                    weights[0] = 1

        cnt += 1
        _fitted = True
        trained_models2 = trained_models
        self.est_noise /= weights.sum()

    @property
    def noise(self):
        return self.est_noise
    
    def predict(self, Xc : torch.Tensor, Xe : torch.Tensor):
        global weights, trained_models, model_keys
        current_means = torch.zeros((Xc.shape[0], 1), dtype=torch.float32)
        current_vars = torch.zeros((Xc.shape[0], 1), dtype=torch.float32)
        all_weights = 0
        for model_name, w in zip(model_keys, weights):
            m = trained_models[model_name]
            if m is None:
                continue
            model, mc = m
            if w == 0:
                continue
            try:
                means, vars = model.predict(Xc, Xe)
            except:
                continue
            
            if not self.use_total_variance:
                current_means += means * w
                current_vars += vars * w
                all_weights += w
            else:
                current_means += means * w
                current_vars += (vars + means ** 2) * w
        
        if all_weights == 0:
            model, mc = trained_models[model_keys[0]]
            means, vars = model.predict(Xc, Xe)
            return means, vars

        if not self.use_total_variance:
            current_means = current_means / all_weights
            current_vars = current_vars / all_weights
        else:
            current_means = current_means / all_weights
            current_vars = current_vars / all_weights - current_means ** 2

        return current_means, current_vars
    
    @staticmethod
    def get_weights_history():
        global weights_history
        return weights_history

    @staticmethod
    def reset():
        global _fitted, cnt, weights_history, weights, model_keys, trained_models

        trained_models = {}
        cnt = 0
        weights = None
        _fitted = False
        weights_history = []
        model_keys = []
