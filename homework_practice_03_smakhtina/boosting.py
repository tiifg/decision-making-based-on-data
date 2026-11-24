from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tqdm.auto import tqdm


class MeanTargetEncoder:
    """Счетчики: категория -> доля объектов с y = +1 среди объектов этой категории.

    ordered=True: для объектов обучающей выборки счетчик считается только по объектам,
    стоящим выше в (случайно перемешанной) выборке, как в Ordered Boosting из CatBoost.
    """

    def __init__(self, cat_features: Iterable[int], ordered: bool = False,
                 smoothing: float = 0.0, random_state: int | None = None):
        self.cat_features = list(cat_features)
        self.ordered = ordered
        self.smoothing = smoothing
        self.random_state = random_state
        self.indexes_ = {}
        self.means_ = {}
        self.prior_ = 0.5

    def _smooth(self, sums, counts):
        total = counts + self.smoothing
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (sums + self.smoothing * self.prior_) / total
        return np.where(total > 0, values, self.prior_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanTargetEncoder":
        positive = (np.asarray(y) == 1).astype(float)
        self.prior_ = float(positive.mean())
        for j in self.cat_features:
            codes, uniques = pd.factorize(X[:, j], use_na_sentinel=False)
            sums = np.bincount(codes, weights=positive, minlength=len(uniques))
            counts = np.bincount(codes, minlength=len(uniques))
            self.indexes_[j] = pd.Index(uniques)
            self.means_[j] = self._smooth(sums, counts.astype(float))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = np.empty(X.shape, dtype=float)
        for j in range(X.shape[1]):
            if j in self.indexes_:
                codes = self.indexes_[j].get_indexer(X[:, j])
                result[:, j] = np.where(codes >= 0, self.means_[j][codes], self.prior_)
            else:
                result[:, j] = X[:, j]
        return result

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        if not self.ordered:
            return self.transform(X)

        positive = (np.asarray(y) == 1).astype(float)
        permutation = np.random.RandomState(self.random_state).permutation(len(positive))
        result = np.empty(X.shape, dtype=float)
        for j in range(X.shape[1]):
            if j not in self.indexes_:
                result[:, j] = X[:, j]
                continue
            codes = self.indexes_[j].get_indexer(X[:, j])[permutation]
            shuffled_positive = positive[permutation]
            sums_before = pd.Series(shuffled_positive).groupby(codes).cumsum().to_numpy() - shuffled_positive
            counts_before = pd.Series(codes).groupby(codes).cumcount().to_numpy().astype(float)
            encoded = np.empty(len(codes))
            encoded[permutation] = self._smooth(sums_before, counts_before)
            result[:, j] = encoded
        return result


class Quantizer:
    """Квантизация признаков: uniform, quantile или piecewise (границы по InfoGain относительно таргета)."""

    def __init__(self, quantization_type: str, nbins: int = 255):
        if quantization_type not in ("uniform", "quantile", "piecewise"):
            raise ValueError(f"неизвестный тип квантизации: {quantization_type}")
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.edges_ = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Quantizer":
        self.edges_ = []
        for j in range(X.shape[1]):
            column = X[:, j]
            if self.quantization_type == "uniform":
                edges = np.linspace(np.nanmin(column), np.nanmax(column), self.nbins + 1)[1:-1]
            elif self.quantization_type == "quantile":
                edges = np.unique(np.nanquantile(column, np.linspace(0, 1, self.nbins + 1)[1:-1]))
            else:
                tree = DecisionTreeClassifier(
                    criterion="entropy", max_leaf_nodes=self.nbins,
                    min_samples_leaf=max(20, len(column) // 1000),
                ).fit(column[:, None], np.asarray(y) == 1)
                edges = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
            self.edges_.append(edges)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = np.empty(X.shape, dtype=float)
        for j, edges in enumerate(self.edges_):
            bins = np.digitize(X[:, j], edges).astype(float)
            result[:, j] = np.where(np.isnan(X[:, j]), len(edges) + 1, bins)
        return result

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def _best_split_sorted(values, grads, hessians, l2, min_samples_leaf):
    """Лучший порог для признака, отсортированного вместе с градиентами и гессианами."""
    n = len(values)
    if n < 2:
        return None
    grad_left = np.cumsum(grads)[:-1]
    hess_left = np.cumsum(hessians)[:-1]
    grad_total, hess_total = grads.sum(), hessians.sum()

    criterion = (
        grad_left ** 2 / (hess_left + l2)
        + (grad_total - grad_left) ** 2 / (hess_total - hess_left + l2)
        - grad_total ** 2 / (hess_total + l2)
    )
    allowed = values[1:] != values[:-1]
    if min_samples_leaf > 1:
        left_size = np.arange(1, n)
        allowed &= (left_size >= min_samples_leaf) & (n - left_size >= min_samples_leaf)
    if not allowed.any():
        return None
    criterion = np.where(allowed, criterion, -np.inf)
    best = int(np.argmax(criterion))
    return (values[best] + values[best + 1]) / 2, float(criterion[best]), best


def find_best_split(feature_vector, grad_vector, hess_vector, l2: float = 1.0, min_samples_leaf: int = 1):
    """Критерий H(R) = G_l^2 / (H_l + l2) + G_r^2 / (H_r + l2) - G^2 / (H + l2) для всех порогов признака.

    Возвращает (thresholds, criteria, threshold_best, criterion_best); пороги - середины между
    соседними различными значениями признака.
    """
    feature_vector = np.asarray(feature_vector, dtype=float)
    order = np.argsort(feature_vector, kind="stable")
    values, grads, hessians = feature_vector[order], np.asarray(grad_vector)[order], np.asarray(hess_vector)[order]

    grad_left, hess_left = np.cumsum(grads)[:-1], np.cumsum(hessians)[:-1]
    grad_total, hess_total = grads.sum(), hessians.sum()
    criteria = (
        grad_left ** 2 / (hess_left + l2)
        + (grad_total - grad_left) ** 2 / (hess_total - hess_left + l2)
        - grad_total ** 2 / (hess_total + l2)
    )
    thresholds = (values[1:] + values[:-1]) / 2
    allowed = values[1:] != values[:-1]
    if min_samples_leaf > 1:
        left_size = np.arange(1, len(values))
        allowed &= (left_size >= min_samples_leaf) & (len(values) - left_size >= min_samples_leaf)
    thresholds, criteria = thresholds[allowed], criteria[allowed]
    if len(criteria) == 0:
        return thresholds, criteria, None, None
    best = int(np.argmax(criteria))
    return thresholds, criteria, thresholds[best], float(criteria[best])


class GHTree:
    """Дерево, которое строится по градиентам и гессианам (как в XGBoost).

    fit получает антиградиенты и гессианы. Значение в листе: сумма антиградиентов / (сумма гессианов + l2).
    Категориальные признаки (cat_features или нечисловые столбцы) нумеруются внутри дерева, категории
    упорядочиваются по G / (H + l2) и делятся порогом по этому порядку, так что для сплита неважно,
    числовой признак или категориальный.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 l2: float | None = 1.0, cat_features: Iterable[int] | None = None,
                 random_state: int | None = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.l2 = 0.0 if l2 is None else l2
        self.cat_features = None if cat_features is None else list(cat_features)
        self.random_state = random_state

    def _to_matrix(self, X, fit: bool) -> np.ndarray:
        X = np.asarray(X)
        matrix = np.empty(X.shape, dtype=float)
        if fit:
            self.is_cat_ = np.zeros(X.shape[1], dtype=bool)
            self.cat_indexes_ = {}
            if self.cat_features is not None:
                self.is_cat_[self.cat_features] = True
            for j in range(X.shape[1]):
                if not self.is_cat_[j] and X.dtype == object:
                    try:
                        X[:, j].astype(float)
                    except (TypeError, ValueError):
                        self.is_cat_[j] = True
        for j in range(X.shape[1]):
            if self.is_cat_[j]:
                if fit:
                    codes, uniques = pd.factorize(X[:, j], use_na_sentinel=False)
                    self.cat_indexes_[j] = pd.Index(uniques)
                    matrix[:, j] = codes
                else:
                    matrix[:, j] = self.cat_indexes_[j].get_indexer(X[:, j])
            else:
                matrix[:, j] = X[:, j]
        return matrix

    def fit(self, X, antigradient, hessian=None, sample_weight=None) -> "GHTree":
        Xm = self._to_matrix(X, fit=True)
        n, d = Xm.shape
        weights = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        hessian = np.ones(n) if hessian is None else np.asarray(hessian, dtype=float)
        self._G = np.asarray(antigradient, dtype=float) * weights
        self._H = hessian * weights
        self._X = Xm
        self.n_categories_ = {j: len(self.cat_indexes_[j]) for j in self.cat_indexes_}

        self.nodes_ = []
        self.gain_ = np.zeros(d)
        orders = [None if self.is_cat_[j] else np.argsort(Xm[:, j], kind="stable") for j in range(d)]
        self._build(np.arange(n), orders, depth=0)

        total = self.gain_.sum()
        self.feature_importances_ = self.gain_ / total if total > 0 else self.gain_
        del self._G, self._H, self._X
        return self

    def _build(self, idx, orders, depth) -> int:
        G, H, Xm = self._G, self._H, self._X
        node_id = len(self.nodes_)
        node = {"value": G[idx].sum() / (H[idx].sum() + self.l2)}
        self.nodes_.append(node)

        n_node = len(idx)
        if depth >= self.max_depth or n_node < self.min_samples_split or n_node < 2 * self.min_samples_leaf:
            return node_id

        best = None
        for j in range(Xm.shape[1]):
            if self.is_cat_[j]:
                codes = Xm[idx, j].astype(int)
                n_cat = self.n_categories_[j]
                valid = codes >= 0
                sum_g = np.bincount(codes[valid], weights=G[idx][valid], minlength=n_cat)
                sum_h = np.bincount(codes[valid], weights=H[idx][valid], minlength=n_cat)
                present = np.bincount(codes[valid], minlength=n_cat) > 0
                if present.sum() < 2:
                    continue
                ratio = np.where(present, sum_g / (sum_h + self.l2), np.inf)
                rank = np.empty(n_cat)
                rank[np.argsort(ratio, kind="stable")] = np.arange(n_cat)
                ranks = rank[codes]
                local = np.argsort(ranks, kind="stable")
                result = _best_split_sorted(ranks[local], G[idx][local], H[idx][local], self.l2, self.min_samples_leaf)
                if result is not None and (best is None or result[1] > best[0]):
                    threshold = result[0]
                    best = (result[1], j, True, np.flatnonzero(present & (rank < threshold)))
            else:
                o = orders[j]
                result = _best_split_sorted(Xm[o, j], G[o], H[o], self.l2, self.min_samples_leaf)
                if result is not None and (best is None or result[1] > best[0]):
                    best = (result[1], j, False, result[0])

        if best is None or best[0] <= 1e-12:
            return node_id

        gain, j, is_cat, split = best
        if is_cat:
            left_mask = np.isin(Xm[idx, j].astype(int), split)
        else:
            left_mask = Xm[idx, j] <= split
        left_idx, right_idx = idx[left_mask], idx[~left_mask]
        if len(left_idx) == 0 or len(right_idx) == 0:
            return node_id

        mask = np.zeros(len(G), dtype=bool)
        mask[left_idx] = True
        left_orders, right_orders = [], []
        for k, o in enumerate(orders):
            if o is None:
                left_orders.append(None)
                right_orders.append(None)
            else:
                to_left = mask[o]
                left_orders.append(o[to_left])
                right_orders.append(o[~to_left])

        self.gain_[j] += gain
        node.update(feature=j, is_cat=is_cat, split=split)
        node["left"] = self._build(left_idx, left_orders, depth + 1)
        node["right"] = self._build(right_idx, right_orders, depth + 1)
        return node_id

    def predict(self, X) -> np.ndarray:
        Xm = self._to_matrix(X, fit=False)
        result = np.empty(Xm.shape[0])
        self._predict_node(0, np.arange(Xm.shape[0]), Xm, result)
        return result

    def _predict_node(self, node_id, idx, Xm, result) -> None:
        node = self.nodes_[node_id]
        if "feature" not in node:
            result[idx] = node["value"]
            return
        column = Xm[idx, node["feature"]]
        if node["is_cat"]:
            go_left = np.isin(column.astype(int), node["split"])
        else:
            go_left = column <= node["split"]
        self._predict_node(node["left"], idx[go_left], Xm, result)
        self._predict_node(node["right"], idx[~go_left], Xm, result)


class Boosting(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        early_stopping_rounds: int | None = 0,
        eval_metric: str | None = None,
        cat_features: Iterable[int] | None = None,
        ordered_cat: bool = False,
        cat_smoothing: float = 0.0,
        subsample: float | None = None,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = "Bernoulli",
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
        quantization_type: str | None = None,
        nbins: int = 255,
        dart: bool = False,
        dropout_rate: float = 0.05,
        l2: float | None = 0.0,
        loss: str = "BCE",
        focal_gamma: float = 2.0,
        random_state: int | None = None,
        verbose: bool = True,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.cat_features = cat_features
        self.ordered_cat = ordered_cat
        self.cat_smoothing = cat_smoothing

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.goss = goss
        self.goss_k = goss_k

        self.quantization_type = quantization_type
        self.nbins = nbins

        self.dart = dart
        self.dropout_rate = dropout_rate
        self.l2 = l2

        self.loss = loss
        self.focal_gamma = focal_gamma

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)
        self.weights = [1.0] * (n_estimators)
        self.feature_subsets = [None] * (n_estimators)

        self.random_state = random_state
        self.verbose = verbose

        self.history = defaultdict(list)

        self.sigmoid = expit
        self._setup_loss()

        self.cat_encoder = None
        self.quantizer = None
        self.best_iteration_ = None
        self._train_pred = None
        self._n_fitted = 0

    def _setup_loss(self) -> None:
        if self.loss == "BCE":
            self.loss_per_object = lambda y, z: np.logaddexp(0, -y * z)
            self.grad_fn = lambda y, z: -y * self.sigmoid(-y * z)
        elif self.loss == "focal":
            gamma = self.focal_gamma

            def loss_per_object(y, z):
                log_p = log_expit(y * z)
                return -((1 - np.exp(log_p)) ** gamma) * log_p

            def grad_fn(y, z):
                log_p = log_expit(y * z)
                p = np.exp(log_p)
                return y * (1 - p) ** gamma * (gamma * p * log_p - (1 - p))

            self.loss_per_object = loss_per_object
            self.grad_fn = grad_fn
        else:
            raise ValueError(f"неизвестный лосс: {self.loss}")
        self.loss_fn = lambda y, z: self.loss_per_object(y, z).mean()
        self.hess_fn = lambda y, z: np.clip(self.sigmoid(z) * self.sigmoid(-z), 1e-6, None)

    def _cat_fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Обучает счетчики и возвращает закодированную обучающую выборку."""
        if self.cat_features is not None and len(self.cat_features) > 0:
            self.cat_encoder = MeanTargetEncoder(
                self.cat_features, ordered=self.ordered_cat,
                smoothing=self.cat_smoothing, random_state=self.random_state,
            )
            return self.cat_encoder.fit_transform(X, y)
        self.cat_encoder = None
        return self._cat_transform(X)

    def _cat_transform(self, X: np.ndarray) -> np.ndarray:
        if self.cat_encoder is not None:
            return self.cat_encoder.transform(X)
        X = np.asarray(X)
        return X if X.dtype == object else X.astype(float)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = self._cat_transform(X)
        if self.quantizer is not None:
            X = self.quantizer.transform(X)
        return X

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)

    def _build_base_model(self):
        params = dict(self.base_model_params)
        accepted = inspect.signature(self.base_model_class.__init__).parameters
        if "random_state" in accepted:
            params.setdefault("random_state", self.random_state)
        if self._newton and "l2" in accepted:
            params["l2"] = self.l2
        return self.base_model_class(**params)

    def _subset_predict(self, step: int, X: np.ndarray) -> np.ndarray:
        features = self.feature_subsets[step]
        return self.models[step].predict(X if features is None else X[:, features])

    def _contribution(self, step: int, X: np.ndarray) -> np.ndarray:
        return self.learning_rate * self.gammas[step] * self.weights[step] * self._subset_predict(step, X)

    def _sample_objects(self, antigradient: np.ndarray):
        n = len(antigradient)
        if self.goss:
            top = max(1, int(self.goss_k * n))
            rest_rate = 0.3 if self.subsample is None else self.subsample
            by_magnitude = np.argsort(-np.abs(antigradient))
            large, small = by_magnitude[:top], by_magnitude[top:]
            chosen_small = small[self._rng.random_sample(len(small)) < rest_rate]
            idx = np.concatenate([large, chosen_small])
            weights = np.concatenate([
                np.ones(len(large)),
                np.full(len(chosen_small), (1 - self.goss_k) / rest_rate),
            ])
            return idx, weights
        if self.bootstrap_type == "Bayesian":
            return None, (-np.log(self._rng.random_sample(n))) ** self.bagging_temperature
        rate = 1.0 if self.subsample is None else self.subsample
        if self.bootstrap_type == "Bernoulli" and rate < 1.0:
            return np.flatnonzero(self._rng.random_sample(n) < rate), None
        return None, None

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Один шаг бустинга. X и y уже подготовлены (после счетчиков и квантизации, метки из {-1, +1})."""
        if self._n_fitted >= self.n_estimators:
            raise RuntimeError("все базовые модели уже обучены")
        if self._train_pred is None:
            self._train_pred = np.zeros(X.shape[0])
        if not hasattr(self, "_rng"):
            self._rng = np.random.RandomState(self.random_state)
        self._newton = bool(self.l2) and "l2" in inspect.signature(self.base_model_class.__init__).parameters

        step = self._n_fitted

        dropped = np.array([], dtype=int)
        if self.dart and step > 0:
            dropped = np.flatnonzero(self._rng.random_sample(step) < self.dropout_rate)
        dropped_pred = np.zeros(X.shape[0])
        for j in dropped:
            dropped_pred += self._contribution(j, X)
        current = self._train_pred - dropped_pred

        gradient = self.grad_fn(y, current)
        antigradient = -gradient

        features = None
        if self.rsm < 1.0:
            mask = self._rng.random_sample(X.shape[1]) < self.rsm
            if not mask.any():
                mask[self._rng.randint(X.shape[1])] = True
            features = np.flatnonzero(mask)

        idx, weights = self._sample_objects(antigradient)

        def take(array):
            array = array if idx is None else array[idx]
            return array if features is None or array.ndim == 1 else array[:, features]

        model = self._build_base_model()
        fit_kwargs = {} if weights is None else {"sample_weight": weights}
        if self._newton:
            hessian = self.hess_fn(y, current)
            hessian = hessian if idx is None else hessian[idx]
            model.fit(take(X), take(antigradient), hessian, **fit_kwargs)
        else:
            model.fit(take(X), take(antigradient), **fit_kwargs)

        new_predictions = model.predict(X if features is None else X[:, features])

        k = len(dropped)
        new_weight = 1.0 / (k + 1)
        kept = current + dropped_pred * (k / (k + 1) if k else 1.0)
        scaled_new = new_weight * new_predictions

        gamma = 1.0 if self._newton else self._find_optimal_gamma(y, kept, scaled_new)

        self.models[step] = model
        self.gammas[step] = gamma
        self.weights[step] = new_weight
        self.feature_subsets[step] = features
        self._changed = [(j, self.weights[j], self.weights[j] * k / (k + 1)) for j in dropped]
        for j, _, new in self._changed:
            self.weights[j] = new

        self._train_pred = kept + self.learning_rate * gamma * scaled_new
        self._n_fitted += 1

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        use_best_model: bool = False,
    ) -> None:
        self.classes_ = np.unique(y_train)
        y_train = self._encode_labels(y_train)
        self._setup_loss()

        X_train = self._cat_fit(X_train, y_train)
        self.quantizer = None
        if self.quantization_type is not None and np.asarray(X_train).dtype != object:
            self.quantizer = Quantizer(self.quantization_type, self.nbins)
            X_train = self.quantizer.fit_transform(X_train, y_train)
        self.n_features_ = X_train.shape[1]

        if eval_set is not None:
            X_valid, y_valid = eval_set
            X_valid = self._prepare(X_valid)
            y_valid = self._encode_labels(y_valid)
            valid_predictions = np.zeros(X_valid.shape[0])

        use_early_stopping = bool(self.early_stopping_rounds)
        if use_early_stopping and eval_set is None:
            raise ValueError("для early stopping нужно передать eval_set")
        if self.dart and (use_early_stopping or use_best_model):
            raise ValueError("DART меняет веса старых деревьев, поэтому early stopping и use_best_model с ним несовместимы")
        metric_key = self.eval_metric or "valid_roc_auc"
        maximize = "loss" not in metric_key

        self.models = [0] * self.n_estimators
        self.gammas = [0] * self.n_estimators
        self.weights = [1.0] * self.n_estimators
        self.feature_subsets = [None] * self.n_estimators
        self.history = defaultdict(list)
        self._train_pred = np.zeros(X_train.shape[0])
        self._n_fitted = 0
        self._rng = np.random.RandomState(self.random_state)
        self.best_iteration_ = None

        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for step in estimator_range:
            self.partial_fit(X_train, y_train)

            self.history["train_loss"].append(self.loss_fn(y_train, self._train_pred))
            self.history["train_roc_auc"].append(roc_auc_score(y_train == 1, self._train_pred))

            if eval_set is not None:
                for j, old, new in self._changed:
                    valid_predictions += (new - old) * self.learning_rate * self.gammas[j] * self._subset_predict(j, X_valid)
                valid_predictions += self._contribution(step, X_valid)
                self.history["valid_loss"].append(self.loss_fn(y_valid, valid_predictions))
                self.history["valid_roc_auc"].append(roc_auc_score(y_valid == 1, valid_predictions))

                current = self.history[metric_key][-1]
                best = self.history[metric_key][self.best_iteration_] if self.best_iteration_ is not None else None
                if best is None or (current > best if maximize else current < best):
                    self.best_iteration_ = step

                if use_early_stopping and step - self.best_iteration_ >= self.early_stopping_rounds:
                    break

        n_models = self._n_fitted
        if use_best_model and eval_set is not None:
            n_models = self.best_iteration_ + 1
        self.models = self.models[:n_models]
        self.gammas = self.gammas[:n_models]
        self.weights = self.weights[:n_models]
        self.feature_subsets = self.feature_subsets[:n_models]

        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros(X.shape[0])
        for step in range(len(self.models)):
            predictions += self._contribution(step, X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare(X)
        positive_proba = self.sigmoid(self._raw_predict(X))
        return np.column_stack([1 - positive_proba, positive_proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]

    def _find_optimal_gamma(
        self,
        y: np.ndarray,
        old_predictions: np.ndarray,
        new_predictions: np.ndarray
    ) -> float:
        # лосс выпуклый по gamma, поэтому вместо сетки из 100 точек хватает поиска на отрезке [0, 1]
        loss = lambda gamma: self.loss_fn(y, old_predictions + gamma * new_predictions)
        gamma = minimize_scalar(loss, bounds=(0, 1), method="bounded", options={"xatol": 1e-4}).x
        return float(gamma) if loss(gamma) < loss(0.0) else 0.0

    def get_feature_importance(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        type: str = "split",
        sample_size: int = 30000,
    ) -> np.ndarray:
        importance = np.zeros(self.n_features_)

        if type == "split":
            for step, model in enumerate(self.models):
                features = self.feature_subsets[step]
                features = np.arange(self.n_features_) if features is None else features
                importance[features] += self.gammas[step] * self.weights[step] * model.feature_importances_

        elif type == "gain":
            if X is None or y is None:
                raise ValueError("для type='gain' нужны X и y")
            X = self._prepare(X)
            y = self._encode_labels(y)
            if len(y) > sample_size:
                chosen = np.random.RandomState(self.random_state).choice(len(y), sample_size, replace=False)
                X, y = X[chosen], y[chosen]
            predictions = np.zeros(len(y))
            for step, model in enumerate(self.models):
                tree = model.tree_
                features = self.feature_subsets[step]
                Xs = X if features is None else X[:, features]
                scale = self.learning_rate * self.gammas[step] * self.weights[step]
                values = tree.value[:, 0, 0]
                current = np.zeros(len(y), dtype=int)
                for _ in range(tree.max_depth):
                    split_feature = tree.feature[current]
                    internal = tree.children_left[current] != -1
                    if not internal.any():
                        break
                    go_left = Xs[np.arange(len(y)), np.maximum(split_feature, 0)].astype(np.float32) <= tree.threshold[current]
                    following = np.where(go_left, tree.children_left[current], tree.children_right[current])
                    following = np.where(internal, following, current)
                    gain = (
                        self.loss_per_object(y, predictions + scale * values[current])
                        - self.loss_per_object(y, predictions + scale * values[following])
                    )
                    original = split_feature if features is None else features[np.maximum(split_feature, 0)]
                    importance += np.bincount(original[internal], weights=gain[internal], minlength=self.n_features_)
                    current = following
                predictions += scale * values[current]
            importance = np.maximum(importance, 0)
        else:
            raise ValueError("type должен быть 'split' или 'gain'")

        total = importance.sum()
        return importance / total if total > 0 else importance

    def plot_history(self, keys: str | Iterable[str]) -> None:
        if isinstance(keys, str):
            keys = [keys]
        plt.figure(figsize=(8, 4))
        for key in keys:
            plt.plot(self.history[key], label=key)
        if self.best_iteration_ is not None:
            plt.axvline(self.best_iteration_, color="gray", linestyle="--", label="лучшая итерация")
        plt.xlabel("Номер базовой модели")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == self.classes_[1], self.predict_proba(X)[:, 1])


BoostingClassifier = Boosting
