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

from smac.constants import N_TREES, VERY_SMALL_NUMBER
from smac.model.random_forest import AbstractRandomForest

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class SKL_ET(AbstractRandomForest):
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
        configspace: ConfigurationSpace,
        n_trees: int = N_TREES,
        n_points_per_tree: int = -1,
        ratio_features: float = 5.0 / 6.0,
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
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        max_features = 0 if ratio_features > 1.0 else max(1, int(len(self._types) * ratio_features))
        self.seed = seed
        self._et = None
        self.reset_model()
        self._log_y = log_y

        self._n_trees = n_trees
        self._n_points_per_tree = n_points_per_tree
        self._ratio_features = ratio_features
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_depth = max_depth
        self._eps_purity = eps_purity
        self._max_nodes = max_nodes
        self._bootstrapping = bootstrapping

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "n_trees": self._n_trees,
                "n_points_per_tree": self._n_points_per_tree,
                "ratio_features": self._ratio_features,
                "min_samples_split": self._min_samples_split,
                "min_samples_leaf": self._min_samples_leaf,
                "max_depth": self._max_depth,
                "eps_purity": self._eps_purity,
                "max_nodes": self._max_nodes,
                "bootstrapping": self._bootstrapping,
                "pca_components": self._pca_components,
            }
        )

        return meta

    def _train(self, X: np.ndarray, y: np.ndarray):
        X = self._impute_inactive(X)
        y = y.flatten()

        self._et.fit(X, y)

        return self

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

        assert not self._log_y
        X = self._impute_inactive(X)

        means_  = self._et.predict(X)
        preds = []
        for estimator in self._et.estimators_:
            preds.append(estimator.predict(X).reshape([-1,1]))
        vars_ = np.var(np.concatenate(preds, axis=1), axis=1)

        return means_.reshape((-1, 1)), vars_.reshape((-1, 1))

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

