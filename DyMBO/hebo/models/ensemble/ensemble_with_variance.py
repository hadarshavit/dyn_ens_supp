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
weights_history_var = []
weights_var = None


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


class AdaptiveWeightModelVar(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf) -> None:
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.models_classes = [partial(RF, conf=dict(n_estimators=10)),
                                               partial(GP, conf={
                                                                'lr'           : 0.01,
                                                                'num_epochs'   : 100,
                                                                'verbose'      : False,
                                                                'noise_lb'     : 8e-4, 
                                                                'pred_likeli'  : False
                                                                })
                                                                ]
        self.alpha = self.conf.get('alpha')
        self.prevX = None
        self.prevY = None
        self.mse_weights = False
        self.est_noise = torch.zeros(self.num_out)
        self.conf = conf
        global weights, weights_history, weights_var, weights_history_var
        if weights is None:
            weights[0] = 1

    def fit(self, Xc : torch.Tensor, Xe : torch.Tensor, y : torch.Tensor):
        print('FIT ENS8')
        global _fitted, cnt, weights_history, weights, model_keys, trained_models, weights_var, weights_history_var
        if _fitted:
            mses = []
            vars_err = []

            newXc = Xc
            newXe = Xe
            newY = y
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
                mses.append(mean_squared_error(newY, means.detach().numpy()))
                vars_edge_high = means.detach().numpy() + vars.detach().numpy()
                vars_edge_low = means.detach().numpy() - vars.detach().numpy()

                dist_high = np.abs(newY - vars_edge_high)
                dist_low = np.abs(newY - vars_edge_low)
                vars_err.append(np.minimum(dist_high, dist_low).sum())

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

            vars_err = np.array(vars_err)
            new_weights = np.zeros_like(weights_var)
            new_weights[np.argmin(vars_err)] = 1

            if cnt == 1:
                weights = new_weights
            else:
                weights_var = (1 - self.alpha) * weights_var + self.alpha * new_weights
            
            weights_history_var.append(weights_var.copy())

        
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
        global weights, trained_models, model_keys, weights_var
        current_means = torch.zeros((Xc.shape[0], 1), dtype=torch.float32)
        current_vars = torch.zeros((Xc.shape[0], 1), dtype=torch.float32)
        all_weights = 0
        all_weights_var = 0
        for model_name, w, w_var in zip(model_keys, weights, weights_var):
            m = trained_models[model_name]
            if m is None:
                continue
            model, mc = m
            if w == 0 and w_var == 0:
                continue
            try:
                means, vars = model.predict(Xc, Xe)
            except:
                continue
            
            current_means += means * w
            current_vars += vars * w_var
            all_weights += w
            all_weights_var += w_var
        
        if all_weights == 0:
            model, mc = trained_models[model_keys[0]]
            means, vars = model.predict(Xc, Xe)
            return means, vars
        current_means = current_means / all_weights
        current_vars = current_vars / all_weights_var

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
