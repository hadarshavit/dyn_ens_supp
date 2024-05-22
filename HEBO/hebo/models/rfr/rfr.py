from __future__ import annotations
import torch
# from skopt.learning import GradientBoostingQuantileRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from torch import FloatTensor, LongTensor
import numpy as np

from ..base_model import BaseModel
from ..layers import OneHotTransform
from ..util import filter_nan
from typing import Any
import lzma
import pickle
import os

import numpy as np
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest
from pyrfr.regression import default_data_container as DataContainer

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"



from typing import Any

import numpy as np


__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


from abc import abstractmethod
from typing import Any, TypeVar

import copy
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler


class SMACRFR:
    """Random forest that takes instance features into account.

    Parameters
    ----------
    n_trees : int, defaults to `N_TREES`
        The number of trees in the random forest.
    n_points_per_tree : int, defaults to -1
        Number of points per tree. If the value is smaller than 0, the number of samples will be used.
    ratio_features : float, defaults to 5.0 / 6.0
        The ratio of features that are considered for splitting.
    min_samples_split : int, defaults to 3
        The minimum number of data points to perform a split.
    min_samples_leaf : int, defaults to 3
        The minimum number of data points in a leaf.
    max_depth : int, defaults to 2**20
        The maximum depth of a single tree.
    eps_purity : float, defaults to 1e-8
        The minimum difference between two target values to be considered.
    max_nodes : int, defaults to 2**20
        The maximum total number of nodes in a tree.
    bootstrapping : bool, defaults to True
        Enables bootstrapping.
    log_y: bool, defaults to False
        The y values (passed to this random forest) are expected to be log(y) transformed.
        This will be considered during predicting.
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    """

    def __init__(
        self,
        n_trees: int = 10,
        n_points_per_tree: int = -1,
        ratio_features: float = 1.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        eps_purity: float = 1e-8,
        max_nodes: int = 2**20,
        bootstrapping: bool = True,
        log_y: bool = False,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ) -> None:

        self._rf_opts = regression.forest_opts()
        self._rf_opts.num_trees = n_trees
        self._rf_opts.do_bootstrapping = bootstrapping
        self._rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self._rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self._rf_opts.tree_opts.max_depth = max_depth
        self._rf_opts.tree_opts.epsilon_purity = eps_purity
        self._rf_opts.tree_opts.max_num_nodes = max_nodes
        self._rf_opts.compute_law_of_total_variance = False
        self._rf: BinaryForest | None = None
        self._log_y = log_y
        self._rng = regression.default_random_engine(seed)

        self._n_trees = n_trees
        self._n_points_per_tree = n_points_per_tree
        self._ratio_features = ratio_features
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_depth = max_depth
        self._eps_purity = eps_purity
        self._max_nodes = max_nodes
        self._bootstrapping = bootstrapping
        self._seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.flatten()

        # self.X = X
        # self.y = y.flatten()

        if self._n_points_per_tree <= 0:
            self._rf_opts.num_data_points_per_tree = X.shape[0]
        else:
            self._rf_opts.num_data_points_per_tree = self._n_points_per_tree
        max_features = 0 if self._ratio_features > 1.0 else max(1, int(X.shape[0] * self._ratio_features))
        self._rf_opts.tree_opts.max_features = max_features
        print(X.shape)

        self._rf = regression.binary_rss_forest()
        self._rf.options = self._rf_opts

        data = self._init_data_container(X, y)
        self._rf.fit(data, rng=self._rng)

    def _init_data_container(self, X: np.ndarray, y: np.ndarray) -> DataContainer:
        """Fills a pyrfr default data container s.t. the forest knows categoricals and bounds for continous data.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameter + #features]
            Input data points.
        Y : np.ndarray [#samples, #objectives]
            The corresponding target values.

        Returns
        -------
        data : DataContainer
            The filled data container that pyrfr can interpret.
        """
        # Retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self._bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)

        return data

    def predict(
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

        assert self._rf is not None
        
        means, vars_ = [], []
        for row_X in X:
            mean_, var = self._rf.predict_mean_var(row_X)
            means.append(mean_)
            vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))


class RFR(BaseModel):
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        self.n_estimators = self.conf.get('n_estimators', 100)
        self.rfr = SMACRFR(n_trees=self.n_estimators)
        self.est_noise = torch.zeros(self.num_out)
        if self.num_enum > 0:
            self.one_hot = OneHotTransform(self.conf['num_uniqs'])

    def xtrans(self, Xc : FloatTensor, Xe: LongTensor) -> np.ndarray:
        if self.num_enum == 0:
            return Xc.detach().numpy()
        else:
            Xe_one_hot = self.one_hot(Xe)
            if Xc is None:
                Xc = torch.zeros(Xe.shape[0], 0)
            return torch.cat([Xc, Xe_one_hot], dim = 1).numpy()

    def fit(self, Xc : torch.Tensor, Xe : torch.Tensor, y : torch.Tensor):
        Xc, Xe, y = filter_nan(Xc, Xe, y, 'all')
        Xtr = self.xtrans(Xc, Xe)
        ytr = y.numpy().reshape(-1)
        self.rfr.fit(Xtr, ytr)
        mse = np.mean((self.rfr.predict(Xtr)[0].reshape(-1) - ytr)**2).reshape(self.num_out)
        # self.est_noise = torch.FloatTensor(mse)

    @property
    def noise(self):
        return self.est_noise
    
    def predict(self, Xc : torch.Tensor, Xe : torch.Tensor):
        X     = self.xtrans(Xc, Xe)
        mean, var = self.rfr.predict(X, return_std=True)
        return torch.FloatTensor(mean.reshape([-1,1])), torch.FloatTensor(var.reshape([-1,1])) + self.noise
