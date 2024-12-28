from abc import ABC, abstractmethod
from typing import Any, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from utils import (
    KAPPA_SCORER,
    add_fold_column,
    calc_initial_th,
    eval_preds_percentile,
    percentile_rounder,
)


def custom_cross_val_score(
    estimator,
    df,
    add_df,
    feature_cols,
    target_col,
    cv,
    scoring=KAPPA_SCORER,
    weight_col=None,
    fit_params=None,
    verbose=True,
):

    fit_params = fit_params or {}
    scores = []
    models = []
    raw_oofs = np.zeros(len(df))
    oofs = np.zeros(len(df))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df, df[target_col])):
        # データの分割
        X_train, X_val = (
            df.iloc[train_idx][feature_cols],
            df.iloc[val_idx][feature_cols],
        )
        y_train, y_val = df.iloc[train_idx][target_col], df.iloc[val_idx][target_col]

        # もしadd_dataのsiiカラムにnanが存在しなければ、データを追加
        if add_df["sii"].isna().sum() == 0:
            X_train = pd.concat(
                [X_train, add_df[add_df["fold"] == fold_idx][feature_cols]], axis=0
            )
            y_train = pd.concat(
                [y_train, add_df[add_df["fold"] == fold_idx][target_col]], axis=0
            )

        model = clone(estimator)
        model.fit(X_train, y_train, **fit_params)

        # 予測と評価
        y_pred_round, y_pred = model.predict(X_val)
        raw_oofs[val_idx] = y_pred
        oofs[val_idx] = y_pred_round
        raw_score = cohen_kappa_score(
            y_val, np.digitize(y_pred, [0.5, 1.5, 2.5]), weights="quadratic"
        )
        score = cohen_kappa_score(y_val, y_pred_round, weights="quadratic")
        scores.append(score)
        models.append(model)

        if verbose:
            print(
                f"Fold {fold_idx + 1}: train num = {len(X_train)}, raw Score = {raw_score:.4f}, Score = {score:.4f}"
            )

    return np.array(scores), models, raw_oofs, oofs


def get_model_and_objective(model_name) -> Tuple[Any, Any]:

    if model_name == "CustomLGBMRegressor":
        defolt_params = {
            "objective": "tweedie",
            "verbosity": -1,
            "n_iter": 1000,
            "boosting_type": "gbdt",
            "learning_rate": 5e-3,
        }
        return CustomLGBMRegressor, lgb_objective, defolt_params
    elif model_name == "CustomXGBRegressor":
        defolt_params = {
            "objective": "reg:tweedie",
            "verbosity": 0,
            "n_estimators": 1000,
            "booster": "gbtree",
            "learning_rate": 5e-3,
            "device": "cuda",
            "enable_categorical": True,
        }
        return CustomXGBRegressor, xgb_objective, defolt_params
    elif model_name == "CustomLGBMClassifier":
        defolt_params = {
            "objective": "multiclass",
            "num_class": 4,
            "metric": "multi_logloss",
            "verbosity": -1,
            "n_iter": 1000,
            "boosting_type": "gbdt",
            "learning_rate": 5e-3,
        }
        return CustomLGBMClassifier, lgb_classification_objective, defolt_params
    elif model_name == "CustomXGBClassifier":
        defolt_params = {
            "objective": "multi:softmax",  # XGBoostのマルチクラス分類用objective
            "num_class": 4,
            "eval_metric": "mlogloss",  # XGBoostでのmetric名
            "verbosity": 0,  # XGBoostでは0が静音モード
            "n_estimators": 1000,  # XGBoostではn_estimatorsを使用
            "booster": "gbtree",  # XGBoostでのbooster指定
            "learning_rate": 5e-3,
            "device": "cuda",
            "enable_categorical": True,  # カテゴリカル変数のサポートを有効化
        }
        return CustomXGBClassifier, xgb_classification_objective, defolt_params
    else:
        raise ValueError(f"Unsupported model: {model_name}")


"""# objective"""


def lgb_objective(
    trial, df_train, df_train_oof, feature_cols, target_col, num_cols, seed
):

    params = {
        "objective": "tweedie",
        "verbosity": -1,
        "n_iter": 1000,
        "random_state": seed,
        "boosting_type": "gbdt",
        "learning_rate": 5e-3,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if df_train_oof["sii"].isna().sum() == 0:
        df_train_oof = add_fold_column(
            df_train_oof, target_col="sii", n_splits=5, seed=seed
        )
    estimator = CustomLGBMRegressor(**params, num_cols=num_cols)

    val_scores, _, _, _ = custom_cross_val_score(
        estimator,
        df_train,
        df_train_oof,
        feature_cols,
        target_col,
        cv=cv,
        verbose=False,
    )

    return np.mean(val_scores)


def xgb_objective(
    trial, df_train, df_train_oof, feature_cols, target_col, num_cols, seed
):

    params = {
        "objective": "reg:tweedie",  # XGBoostではreg:tweedieを使用
        "verbosity": 0,  # XGBoostでは0が静かモード
        "n_estimators": 1000,  # XGBoostではn_estimatorsを使用
        "random_state": seed,
        "booster": "gbtree",  # XGBoostではgbtreeを使用
        "learning_rate": 5e-3,
        "device": "cuda",
        "enable_categorical": True,
        # 正則化パラメータ
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),  # L1正則化
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10.0, log=True
        ),  # L2正則化
        # ツリー構造パラメータ
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "max_leaves": trial.suggest_int("max_leaves", 16, 256),  # num_leavesに相当
        # サンプリングパラメータ
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.4, 1.0
        ),  # colsample_bynodeに相当
        "subsample": trial.suggest_float(
            "subsample", 0.4, 1.0
        ),  # bagging_fractionに相当
        "subsample_freq": trial.suggest_int(
            "subsample_freq", 1, 7
        ),  # bagging_freqに相当
        # その他のパラメータ
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 5, 100
        ),  # min_data_in_leafに相当
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if df_train_oof["sii"].isna().sum() == 0:
        df_train_oof = add_fold_column(
            df_train_oof, target_col="sii", n_splits=5, seed=seed
        )
    estimator = CustomXGBRegressor(**params, num_cols=num_cols)

    val_scores, _, _, _ = custom_cross_val_score(
        estimator,
        df_train,
        df_train_oof,
        feature_cols,
        target_col,
        cv=cv,
        verbose=False,
    )

    return np.mean(val_scores)


def lgb_classification_objective(
    trial, df_train, df_train_oof, feature_cols, target_col, num_cols, seed
):  # seedパラメータを追加

    params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_iter": 1000,
        "random_state": seed,
        "boosting_type": "gbdt",
        "learning_rate": 5e-3,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if df_train_oof["sii"].isna().sum() == 0:
        df_train_oof = add_fold_column(
            df_train_oof, target_col="sii", n_splits=5, seed=seed
        )
    estimator = CustomLGBMClassifier(**params, num_cols=num_cols)

    val_scores, _, _, _ = custom_cross_val_score(
        estimator,
        df_train,
        df_train_oof,
        feature_cols,
        target_col,
        cv=cv,
        verbose=False,
    )

    return np.mean(val_scores)


def xgb_classification_objective(
    trial, df_train, df_train_oof, feature_cols, target_col, num_cols, seed
):  # seedパラメータを追加

    params = {
        "objective": "multi:softmax",  # XGBoostのマルチクラス分類用objective
        "num_class": 4,
        "eval_metric": "mlogloss",  # XGBoostでのmetric名
        "verbosity": 0,  # XGBoostでは0が静音モード
        "n_estimators": 1000,  # XGBoostではn_estimatorsを使用
        "random_state": seed,
        "booster": "gbtree",  # XGBoostでのbooster指定
        "learning_rate": 5e-3,
        "device": "cuda",
        "enable_categorical": True,  # カテゴリカル変数のサポートを有効化
        # 正則化パラメータ
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),  # L1正則化
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 10.0, log=True
        ),  # L2正則化
        # ツリー構造パラメータ
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "max_leaves": trial.suggest_int("max_leaves", 16, 256),  # num_leaves相当
        # サンプリングパラメータ
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.4, 1.0
        ),  # colsample_bynode相当
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),  # bagging_fraction相当
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),  # bagging_freq相当
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 5, 100
        ),  # min_data_in_leaf相当
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if df_train_oof["sii"].isna().sum() == 0:
        df_train_oof = add_fold_column(
            df_train_oof, target_col="sii", n_splits=5, seed=seed
        )
    estimator = CustomXGBClassifier(**params, num_cols=num_cols)

    val_scores, _, _, _ = custom_cross_val_score(
        estimator,
        df_train,
        df_train_oof,
        feature_cols,
        target_col,
        cv=cv,
        verbose=False,
    )

    return np.mean(val_scores)


"""# models"""


class BaseCustomEstimator(ABC):
    """
    Abstract base class for all custom estimators with threshold optimization
    Main goal is preventing overfit on validation data.
    """

    def __init__(self, num_cols=None, **kwargs):
        self.num_cols = num_cols
        self.model = self._create_model(**kwargs)
        self.imputer = SimpleImputer(strategy="median")

    @abstractmethod
    def _create_model(self, **kwargs):
        """Create and return the specific model instance"""
        pass

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.imputer = SimpleImputer(
            strategy="median",
        )

        X[self.num_cols] = self.imputer.fit_transform(X[self.num_cols])
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)

        if hasattr(self.model, "predict_proba"):
            # Classifier case
            y_pred = self.model.predict_proba(X, **kwargs)
            y_pred = np.sum(y_pred * np.arange(4), axis=1)
        else:
            # Regressor case
            y_pred = self.model.predict(X, **kwargs)

        oof_initial_thresholds = calc_initial_th(y, y_pred)
        self.optimizer = minimize(
            eval_preds_percentile,
            x0=oof_initial_thresholds,
            args=(y, y_pred),
            method="Nelder-Mead",
            bounds=[(0, 1), (0, 1), (0, 1)],
        )

    def predict(self, X, **kwargs):
        X[self.num_cols] = self.imputer.transform(X[self.num_cols])

        if hasattr(self.model, "predict_proba"):
            # Classifier case
            proba = self.model.predict_proba(X, **kwargs)
            y_pred = np.sum(proba * np.arange(4), axis=1)
        else:
            # Regressor case
            y_pred = self.model.predict(X, **kwargs)

        y_pred_round = percentile_rounder(y_pred, self.optimizer.x)
        return y_pred_round, y_pred

    def predict_proba(self, X, **kwargs):
        """
        Classifier用のメソッド
        Regressorの場合は使用不可
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("This estimator has no predict_proba method")
        return self.predict(X, **kwargs)

    def get_params(self, deep=True):
        return {**self.model.get_params(deep), "num_cols": self.num_cols}

    def set_params(self, **params):
        if "num_cols" in params:
            self.num_cols = params.pop("num_cols")
        self.model.set_params(**params)
        return self


class CustomLGBMRegressor(BaseCustomEstimator):
    """Custom LightGBM Regressor with threshold optimization"""

    def _create_model(self, **kwargs):
        return lgb.LGBMRegressor(**kwargs)


class CustomXGBRegressor(BaseCustomEstimator):
    """Custom XGBoost Regressor with threshold optimization"""

    def _create_model(self, **kwargs):
        return xgb.XGBRegressor(**kwargs)


class CustomLGBMClassifier(BaseCustomEstimator):
    """Custom LightGBM Classifier with threshold optimization"""

    def _create_model(self, **kwargs):
        return lgb.LGBMClassifier(**kwargs)


class CustomXGBClassifier(BaseCustomEstimator):
    """Custom XGBoost Classifier with threshold optimization"""

    def _create_model(self, **kwargs):
        return xgb.XGBClassifier(**kwargs)
