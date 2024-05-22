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


class AdaptiveWeightModel(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf) -> None:
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.models_classes = [partial(RF, conf=dict(n_estimators=10)),
                                               partial(ET, conf=dict(n_estimators=10)), 
                                               partial(GB, conf=dict(n_estimators=10)),
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
        # self.enums
        global weights, weights_history
        if weights is None:
            if num_enum > 0:
                weights = np.zeros((len(self.models_classes)))
                weights[0] = 1
                weights_history = [weights.copy()]
            else:
                weights = np.ones((len(self.models_classes)))
                weights[3] = 1
                weights_history = [weights.copy()]
        

    def fit(self, Xc : torch.Tensor, Xe : torch.Tensor, y : torch.Tensor):
        global _fitted, cnt, weights_history, weights, model_keys, trained_models
        if _fitted:
            mses = []
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
                mses.append(mean_squared_error(newY, means.detach().numpy()))

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
            
            current_means += means * w
            current_vars += vars * w
            all_weights += w
        
        if all_weights == 0:
            model, mc = trained_models[model_keys[0]]
            means, vars = model.predict(Xc, Xe)
            return means, vars
        current_means = current_means / all_weights
        current_vars = current_vars / all_weights

        return current_means, current_vars
    
    @staticmethod
    def get_weights_history():
        global weights_history
        return weights_history
    