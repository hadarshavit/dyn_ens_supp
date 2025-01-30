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
weights = np.array([1, 1, 1, 1])
_fitted = False
active_model = 0
weights_history = []
model_keys = []


class EnsSel(BaseModel):
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
     
        

    def fit(self, Xc : torch.Tensor, Xe : torch.Tensor, y : torch.Tensor):
        global _fitted, cnt, weights_history, weights, model_keys, trained_models, active_model
        
        indexes = torch.arange(y.shape[0])
        if not _fitted:
            print('FIT SELEC')

            mses = np.zeros(len(self.models_classes))

            for i in range(y.shape[0]):
                cur_indexes = indexes[indexes != i]
                for mid, mc in enumerate(self.models_classes):
                    try:
                        model = mc(self.num_cont, self.num_enum, self.num_out, **self.conf)
                        model.fit(Xc[cur_indexes], Xe[cur_indexes], y[cur_indexes])
                        means, vars = model.predict(Xc[[i]], Xe[[i]])
                        mses[mid] += mean_squared_error(y[[i]], means.detach().numpy())
                    except:
                        mses[mid] += 10000

            weights = np.zeros(len(self.models_classes))
            weights[np.argmin(mses)] = 1
            active_model = np.argmin(mses)


        trained_models = {}
        model_keys = []
        int_c = 0
        self.est_noise = torch.zeros(self.num_out)
        
        model = self.models_classes[active_model](self.num_cont, self.num_enum, self.num_out, **self.conf)
        model.fit(Xc, Xe, y)
        trained_models = (model, self.models_classes[active_model])
        model_keys.append(f"{model.__class__.__name__}{cnt}_{int_c}")
        self.est_noise += model.noise * weights[int_c]
        
        cnt += 1
        _fitted = True
       
        self.est_noise /= weights.sum()

    @property
    def noise(self):
        return self.est_noise
    
    def predict(self, Xc : torch.Tensor, Xe : torch.Tensor):
        global weights, trained_models, model_keys
        model, mc = trained_models
        means, vars = model.predict(Xc, Xe)

        return means, vars
    
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
