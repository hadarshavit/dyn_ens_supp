from __future__ import annotations

from typing import Any

from copy import deepcopy
import pickle
import lzma
import os

import numpy as np
from ConfigSpace import ConfigurationSpace
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest
from pyrfr.regression import default_data_container as DataContainer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from smac.constants import N_TREES, VERY_SMALL_NUMBER
from smac.model import AbstractModel
from smac.model.random_forest import SKL_ET, SKL_GB, SKL_RF
from smac.facade import BlackBoxFacade
from ConfigSpace.hyperparameters import NumericalHyperparameter
from functools import partial

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Ensemble(AbstractModel):
    def __init__(
        self,
        scenario,
        alpha, 
        n_restarts=0,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ) -> None:
        super().__init__(
            configspace=scenario.configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        self.models_classes = [SKL_RF(scenario.configspace),
                                SKL_ET(scenario.configspace), 
                                SKL_GB(scenario.configspace),
                                BlackBoxFacade.get_model(scenario)
                            ]
        self.models_classes[-1]._n_restarts = n_restarts
        self.alpha = alpha
        self.prevX = None
        self.prevY = None
        self.mse_weights = False
       
        self.trained_models = {}
        self.cnt = 0
        self.weights = None
        self._fitted = False
        self.weights_history = []
        self.model_keys = []

        has_cats = False
        for hp in scenario.configspace:
            if not isinstance(hp, NumericalHyperparameter):
                has_cats = True
                break

        if has_cats:
            self.weights = np.zeros((len(self.models_classes)))
            self.weights[0] = 1
            self.weights_history = [self.weights.copy()]
        else:
            self.weights = np.ones((len(self.models_classes)))
            self.weights[3] = 1
            self.weights_history = [self.weights.copy()]


    def _train(self, X: np.ndarray, y: np.ndarray):
        if self._fitted:
            mses = []
            newX = X[-8:]
            newY = y[-8:]

            for i, (model_name, w) in enumerate(zip(self.model_keys, self.weights)):
                model = self.trained_models[model_name]
                if model is None:
                    mses.append(100000)
                    continue
                try:
                    means, vars = model.predict(newX)
                except:
                    mses.append(100000)
                    continue
                mses.append(mean_squared_error(newY, means))

            mses = np.array(mses)
            if not self.mse_weights:
                new_weights = np.zeros_like(self.weights)
                new_weights[np.argmin(mses)] = 1

            else:
                mses = 1 / np.power(mses, 2)
                normalised_mses = mses / mses.sum()
                new_weights = normalised_mses

            if self.cnt == 1:
                self.weights = new_weights
            else:
                self.weights = (1 - self.alpha) * self.weights + self.alpha * new_weights
            self.weights_history.append(self.weights.copy())
        
        self.trained_models = {}
        self.model_keys = []
        int_c = 0
        for model in self.models_classes:
            try:
                model.train(X, y)
                self.trained_models[f"{model.__class__.__name__}{self.cnt}_{int_c}"] = model
                self.model_keys.append(f"{model.__class__.__name__}{self.cnt}_{int_c}")
                int_c += 1
            except:
                self.trained_models[f"{model.__class__.__name__}{self.cnt}_{int_c}"] = None
                self.model_keys.append(f"{model.__class__.__name__}{self.cnt}_{int_c}")
                self.weights[int_c] = 0
                if np.count_nonzero(self.weights) == 0:
                    self.weights[0] = 1

        self.cnt += 1
        self._fitted = True

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._types), X.shape[1]))

        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        current_means = np.zeros((X.shape[0], 1), dtype=np.float64)
        current_vars = np.zeros((X.shape[0], 1), dtype=np.float64)
        all_weights = 0
        # print(X.shape)
        for model_name, w in zip(self.model_keys, self.weights):
            model = self.trained_models[model_name]
            if model is None:
                continue
            if w == 0:
                continue
            # print(model_name)
            try:
                means, vars = model.predict(X)
            except:
                continue
            # print((any(means < 0)), any(vars < 0))
            
            current_means += means * w
            current_vars += vars * w
            all_weights += w
        
        if all_weights == 0:
            model = self.trained_models[self.model_keys[0]]
            means, vars = model.predict(X)
            return means, vars
        current_means = current_means / all_weights
        current_vars = current_vars / all_weights

        return current_means, current_vars

    def copy_model(self):
        return deepcopy(self._et)

    def set_model(self, model):
        self._et = model

    def reset_model(self):
        self._et = ExtraTreesRegressor(criterion="squared_error", max_features=1, random_state=self.seed, n_estimators=10)
    
    def save_model(self, path) -> None:
        with lzma.open(path, 'wb') as f:
            pickle.dump(f, self._gb)

    def load_model(self, path):
         with lzma.open(path, 'rb') as f:
            self._gb = pickle.load(f)
        
    def save_class(self, path):
        et = self._et
        self._et = None

        with lzma.open(path, 'wb') as f:
            pickle.dump(f, self)

        self._et = et

    def save(self, path):
        os.makedirs(path)
        with lzma.open(os.path.join(path, 'main_class.pkl'), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with lzma.open(os.path.join(path, 'main_class.pkl'), 'rb') as f:
            cls = pickle.load(f)
            return cls._et, cls

    