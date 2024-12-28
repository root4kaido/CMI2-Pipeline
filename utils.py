import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold

KAPPA_SCORER = make_scorer(
    cohen_kappa_score,
    greater_is_better=True,
    weights="quadratic",
)


def eval_preds_percentile(percentiles, y_true, y_pred):
    y_pred_discrete = percentile_rounder(y_pred, percentiles)
    score = cohen_kappa_score(y_true, y_pred_discrete, weights="quadratic")
    return -score


def percentile_rounder(y_pred, percentiles):
    # パーセンタイル値を計算して閾値として使用
    thresholds = np.percentile(y_pred, [p * 100 for p in percentiles])

    # 予測値を離散化
    y_pred_discrete = np.zeros_like(y_pred)
    for i in range(len(thresholds) + 1):
        if i == 0:
            mask = y_pred < thresholds[0]
            y_pred_discrete[mask] = i
        elif i == len(thresholds):
            mask = y_pred >= thresholds[-1]
            y_pred_discrete[mask] = i
        else:
            mask = (y_pred >= thresholds[i - 1]) & (y_pred < thresholds[i])
            y_pred_discrete[mask] = i

    return y_pred_discrete


def calc_initial_th(y, y_pred):
    tmp_df = pd.DataFrame({"sii": y, "prediction": y_pred})
    oof_initial_thresholds = (
        tmp_df.groupby("sii")["prediction"].mean().iloc[1:].values.tolist()
    )

    oof_threshold_percentiles = [
        (tmp_df["prediction"] <= threshold).mean()
        for threshold in oof_initial_thresholds
    ]

    return oof_threshold_percentiles


def add_fold_column(df, target_col, n_splits=5, shuffle=True, seed=42):

    # Initialize array with -1
    folds = np.full(len(df), -1)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    # Fill fold array
    for fold_id, (_, valid_idx) in enumerate(skf.split(df, df[target_col])):
        folds[valid_idx] = fold_id

    # Add fold column to DataFrame
    df_with_folds = df.copy()
    df_with_folds["fold"] = folds

    return df_with_folds


def voting(array_2d):
    # axis=0で各列について最頻値を計算
    # mode関数は(最頻値, 出現回数)のタプルを返すため[0]で最頻値のみを取得
    return stats.mode(array_2d, axis=0)[0]
