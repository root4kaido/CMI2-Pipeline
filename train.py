# -*- coding: utf-8 -*-

import logging
import os
import pickle
import warnings
from pathlib import Path

import hydra
import numpy as np
import optuna
import pandas as pd
import polars as pl
from lightning.pytorch import seed_everything
from models import custom_cross_val_score, get_model_and_objective
from preprocess import preprocess
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from utils import KAPPA_SCORER, add_fold_column, voting

warnings.filterwarnings("ignore")

cat_cols = [
    "Basic_Demos-Enroll_Season",
    "CGAS-Season",
    "Physical-Season",
    "Fitness_Endurance-Season",
    "FGC-Season",
    "BIA-Season",
    "PAQ_A-Season",
    "PAQ_C-Season",
    "SDS-Season",
    "PreInt_EduHx-Season",
    "Basic_Demos-Sex",
    "FGC-FGC_CU_Zone",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD_Zone",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL_Zone",
    "BIA-BIA_Activity_Level_num",
    "BIA-BIA_Frame_num",
    "PreInt_EduHx-computerinternet_hoursday",
]

num_cols = [
    "Basic_Demos-Age",
    "Basic_Demos-Sex",
    "CGAS-CGAS_Score",
    "Physical-BMI",
    "Physical-Height",
    "Physical-Weight",
    "Physical-Waist_Circumference",
    "Physical-Diastolic_BP",
    "Physical-HeartRate",
    "Physical-Systolic_BP",
    "Fitness_Endurance-Max_Stage",
    "Fitness_Endurance-Time_Mins",
    "Fitness_Endurance-Time_Sec",
    "FGC-FGC_CU",
    "FGC-FGC_CU_Zone",
    "FGC-FGC_GSND",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD",
    "FGC-FGC_GSD_Zone",
    "FGC-FGC_PU",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL",
    "FGC-FGC_TL_Zone",
    "BIA-BIA_Activity_Level_num",
    "BIA-BIA_BMC",
    "BIA-BIA_BMI",
    "BIA-BIA_BMR",
    "BIA-BIA_DEE",
    "BIA-BIA_ECW",
    "BIA-BIA_FFM",
    "BIA-BIA_FFMI",
    "BIA-BIA_FMI",
    "BIA-BIA_Fat",
    "BIA-BIA_Frame_num",
    "BIA-BIA_ICW",
    "BIA-BIA_LDM",
    "BIA-BIA_LST",
    "BIA-BIA_SMM",
    "BIA-BIA_TBW",
    "PAQ_A-PAQ_A_Total",
    "PAQ_C-PAQ_C_Total",
    "SDS-SDS_Total_Raw",
    "SDS-SDS_Total_T",
    "PreInt_EduHx-computerinternet_hoursday",
]

target_col = "sii"


def make_pseudo_data(df_train: pd.DataFrame, path: str):

    df_train_oof = df_train[df_train["sii"].isna()].copy()

    if (path is not None) and (os.path.exists(path)):
        print(f"oof path: {path}")
        oof = pd.read_csv(path)
        df_train_oof["sii"] = oof["sii"].to_numpy()

    return df_train_oof


@hydra.main(version_base=None, config_path="config/", config_name="train")
def main(cfg):

    root = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log = logging.getLogger(__name__)

    SEED = 42
    seed_everything(SEED)

    """### PreProcess"""

    df_train = pd.read_csv(root / "train.csv")
    df_train, feature_cols = preprocess(df_train, "train", root, cat_cols, num_cols)

    """### Drop Rows with Missing Targets"""

    df_train_oof = make_pseudo_data(df_train, cfg.oof_path)
    df_train = df_train.dropna(subset=[target_col])

    """### Optuna - Hyperparameter Tuning"""

    total_models = []
    total_scores = []
    total_oofs = []

    for SEED in [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]:

        seed_everything(SEED)
        model_class, objective, default_params = get_model_and_objective(cfg.model_name)

        """### Optuna - Hyperparameter Tuning"""

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial,
                df_train,
                df_train_oof,
                feature_cols,
                target_col,
                num_cols,
                seed=SEED,
            ),
            n_trials=30,
            show_progress_bar=True,
        )

        """### Train Model"""

        params = default_params
        params["random_state"] = SEED
        params.update(study.best_params)

        model = model_class(**params, num_cols=num_cols)

        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        if df_train_oof["sii"].isna().sum() == 0:
            df_train_oof = add_fold_column(
                df_train_oof, target_col="sii", n_splits=5, seed=SEED
            )

        val_scores, models, raw_oofs, oofs = custom_cross_val_score(
            model,
            df_train,
            df_train_oof,
            feature_cols,
            target_col,
            cv=cv,
            weight_col=df_train["PCIAT-PCIAT_Total"],
            scoring=KAPPA_SCORER,
        )

        pred_df = pl.DataFrame(
            {
                "id": df_train.index.to_list(),
                "sii": df_train[target_col].to_numpy(),
                "raw_pred": raw_oofs,
                "pred": oofs,
            }
        )
        pred_df.write_csv(output_dir / f"oof_seed{SEED}.csv")

        total_models.append(models)
        total_scores.append(np.mean(val_scores))
        total_oofs.append(oofs)
        # break

    log.info(f"total_scores: {total_scores}")

    total_oofs = np.array(total_oofs)
    total_oofs = voting(total_oofs)
    voting_score = cohen_kappa_score(
        df_train["sii"].to_numpy().reshape(-1),
        total_oofs.reshape(-1),
        weights="quadratic",
    )
    log.info(
        f"Voting Score: {voting_score:.4f}, Score Mean: {np.mean(total_scores):.4f}"
    )

    # データを保存する
    with open(output_dir / "total_models.pkl", "wb") as f:
        pickle.dump(total_models, f)


if __name__ == "__main__":
    main()
