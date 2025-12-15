import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.tree import DecisionTreeRegressor

from ensembles.utils import ConvergenceHistory, rmsle, whether_to_stop


class RandomForestMSE:
    def __init__(
        self, n_estimators: int, tree_params: dict[str, Any] | None = None
    ) -> None:
        """
        Handmade random forest regressor.

        Classic ML algorithm that trains a set of independent tall decision trees and averages its predictions. 
        Employs scikit-learn `DecisionTreeRegressor` under the hood.

        Args:
            n_estimators (int): Number of trees in the forest.
            tree_params (dict[str, Any] | None, optional): Parameters for sklearn trees. Defaults to None.
        """
        self.n_estimators = n_estimators
        if tree_params is None:
            tree_params = {}
        self.forest = [
            DecisionTreeRegressor(**tree_params) for _ in range(n_estimators)
        ]

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64] | None = None,
        y_val: npt.NDArray[np.float64] | None = None,
        trace: bool | None = None,
        patience: int | None = None,
    ) -> ConvergenceHistory | None:
        """
        Train an ensemble of trees on the provided data.

        Args:
            X (npt.NDArray[np.float64]): Objects features matrix, array of shape (n_objects, n_features).
            y (npt.NDArray[np.float64]): Regression labels, array of shape (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional): Validation set of objects, array of shape
            (n_val_objects, n_features). Defaults to None.
            y_val (npt.NDArray[np.float64] | None, optional): Validation set of labels, array of shape
            (n_val_objects,). Defaults to None.
            trace (bool | None, optional): Whether to calculate rmsle while training. 
            True by default if validation data is provided. Defaults to None.
            patience (int | None, optional): Number of training steps without decreasing the train loss
            (or validation if provided), after which to stop training. Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory` if `trace=True` or if validation data 
            is provided.
        """

        if trace is None:
            trace = X_val is not None and y_val is not None

        y_log = np.log1p(y)
        y_val_log = np.log1p(y_val) if y_val is not None else None

        history: ConvergenceHistory | None = None
        if trace:
            history = {'train': [], 'val': [] if X_val is not None else None}

        n_objects = X.shape[0]
        n_trained = 0

        train_sum = np.zeros_like(y_log, dtype=np.float64)
        val_sum = np.zeros_like(y_val_log, dtype=np.float64) if y_val is not None else None

        for tree in self.forest:
            idx = np.random.choice(n_objects, size=n_objects, replace=True)
            tree.fit(X[idx], y_log[idx])

            y_pred_new_train = tree.predict(X)
            train_sum += y_pred_new_train

            if X_val is not None:
                y_pred_new_val = tree.predict(X_val)
                val_sum += y_pred_new_val

            n_trained += 1

            if trace:
                avg_train_pred_log = train_sum / n_trained
                loss_train = rmsle(y, np.expm1(avg_train_pred_log))
                history['train'].append(loss_train)

                if X_val is not None:
                    avg_val_pred_log = val_sum / n_trained
                    loss_val = rmsle(y_val, np.expm1(avg_val_pred_log))
                    history['val'].append(loss_val)

                if patience is not None and whether_to_stop(history, patience):
                    break

        return history

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make prediction with ensemble of trees.

        All the trees make their own predictions which then are averaged.

        Args:
            X (npt.NDArray[np.float64]): Objects' features matrix, array of shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape (n_objects,).
        """
        preds_log = np.zeros(X.shape[0], dtype=np.float64)
        n_trained_trees = 0
        for tree in self.forest:
            if hasattr(tree, 'tree_'):
                preds_log += tree.predict(X)
                n_trained_trees += 1

        if n_trained_trees == 0:
            raise RuntimeError('Cannot predict: no trained trees in the forest.')

        return np.expm1(preds_log / n_trained_trees)

    def dump(self, dirpath: str) -> None:
        """
        Save the trained model to the specified directory.

        Args:
            dirpath (str): Path to the directory where the model will be saved.
        """
        path = Path(dirpath)
        path.mkdir(parents=True)

        params = {"n_estimators": self.n_estimators}
        with (path / "params.json").open("w") as file:
            json.dump(params, file, indent=4)

        trees_path = path / "trees"
        trees_path.mkdir()
        for i, tree in enumerate(self.forest):
            joblib.dump(tree, trees_path / f"tree_{i:04d}.joblib")

    @classmethod
    def load(cls, dirpath: str) -> "RandomForestMSE":
        """
        Load a trained model from the specified directory.

        Args:
            dirpath (str): Path to the directory where the model is saved.

        Returns:
            RandomForestMSE: An instance of the loaded model.
        """
        with (Path(dirpath) / "params.json").open() as file:
            params = json.load(file)
        instance = cls(params["n_estimators"])

        trees_path = Path(dirpath) / "trees"

        instance.forest = [
            joblib.load(trees_path / f"tree_{i:04d}.joblib")
            for i in range(params["n_estimators"])
        ]

        return instance
