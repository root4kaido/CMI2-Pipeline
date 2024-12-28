# -*- coding: utf-8 -*-

import logging
import os
import pickle
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from lightning.pytorch import seed_everything
from preprocess import preprocess
from utils import voting

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


@hydra.main(version_base=None, config_path="config/", config_name="inference")
def main(cfg):

    root = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log = logging.getLogger(__name__)
    phase = cfg.phase

    SEED = 42
    seed_everything(SEED)

    with open(cfg.model_path, "rb") as f:
        total_models = pickle.load(f)

    """### PreProcess"""

    df_test = pd.read_csv(root / f"{phase}.csv")
    df_test, feature_cols = preprocess(df_test, phase, root, cat_cols, num_cols)

    if phase == "train":
        df_test = df_test[df_test[target_col].isna()]

    """### Prediction"""

    test_pred = []
    for models in total_models:
        for model in models:
            proba, _ = model.predict(df_test[feature_cols])
            test_pred.append(proba)
    test_pred = np.array(test_pred)

    test_pred_voted = voting(test_pred)
    df_test["sii"] = test_pred_voted.reshape(-1)

    if phase == "train":
        np.save(output_dir / "pseudo_submission", test_pred)
        df_test["sii"].to_csv(output_dir / "pseudo_submission.csv")
    else:
        np.save(output_dir / "submission", test_pred)
        df_test["sii"].to_csv(output_dir / "submission.csv")


if __name__ == "__main__":
    main()
