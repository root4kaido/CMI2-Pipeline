#!/bin/bash

# pretrain
python3 train.py model_name=CustomLGBMRegressor exp_name=lgbm_regressor_pretrain
python3 inference.py model_name=CustomLGBMRegressor exp_name=lgbm_regressor_pretrain model_path=/home/user/work/CMI2024/Git/src/results/lgbm_regressor_pretrain/total_models.pkl phase=pseudo

# train
python3 train.py model_name=CustomLGBMRegressor exp_name=lgbm_regressor oof_path=/home/user/work/CMI2024/Git/src/results/lgbm_regressor_pretrain/pseudo_submission.csv
python3 train.py model_name=CustomXGBRegressor exp_name=xgb_regressor oof_path=/home/user/work/CMI2024/Git/src/results/lgbm_regressor_pretrain/pseudo_submission.csv
python3 train.py model_name=CustomLGBMClassifier exp_name=lgbm_classifier oof_path=/home/user/work/CMI2024/Git/src/results/lgbm_regressor_pretrain/pseudo_submission.csv
python3 train.py model_name=CustomXGBClassifier exp_name=xgb_classifier oof_path=/home/user/work/CMI2024/Git/src/results/lgbm_regressor_pretrain/pseudo_submission.csv

