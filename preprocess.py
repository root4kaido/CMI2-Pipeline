import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

use_cols = [
    "X",
    "Y",
    "Z",
    "enmo",
    "anglez",
    "non-wear_flag",
    "light",
    "battery_voltage",
    "time_of_day",
    "weekday",
    "quarter",
    # "relative_date_PCIAT",
]
statistics_cols = [
    "count",
    "null_count",
    "mean",
    "std",
    "min",
    "25%",
    "50%",
    "75%",
    "max",
]

# 各カラムと統計量の組み合わせを作成
metric_names = []
for col in use_cols:
    for stat in statistics_cols:
        metric_name = f"stats_{col}_{stat}"
        metric_names.append(metric_name)


def process_file(filename: str, dirname: str) -> tuple:
    # ファイルを読み込んでstep列を削除
    df = pl.read_parquet(os.path.join(dirname, filename, "part-0.parquet")).drop("step")

    stats = (
        df.describe()[["statistic"] + use_cols]
        .filter(pl.col("statistic").is_in(statistics_cols))
        .melt(id_vars="statistic")
        .with_columns(metric=pl.col("variable") + "_" + pl.col("statistic"))
        .select(["metric", "value"])
    )

    return stats["value"].to_numpy(), filename.split("=")[1]


def load_time_series(dirname: str, suffix=None) -> pl.DataFrame:
    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(lambda fname: process_file(fname, dirname), ids),
                total=len(ids),
            )
        )

    stats, indexes = zip(*results)

    # DataFrameの作成
    df = pl.DataFrame(
        {
            **{
                f"{metric_names[i]}": [row[i] for row in stats]
                for i in range(len(stats[0]))
            },
            "id": indexes,
        }
    )

    return df.to_pandas()


def preprocess(df, phase, root, cat_cols, num_cols):
    """### Time Series Data"""

    ts_df = load_time_series(root / f"series_{phase}.parquet")

    time_series_cols = ts_df.columns.tolist()
    time_series_cols.remove("id")

    df = pd.merge(df, ts_df, how="left", on="id")
    df = df.set_index("id")

    """### Cast Data"""

    new_cat_cols = [col + "_cate" for col in cat_cols]
    num_cols = num_cols + time_series_cols
    feature_cols = num_cols + new_cat_cols

    df[num_cols] = df[num_cols].astype(np.float32)

    cate_df = df[cat_cols].copy()
    cate_df.columns = new_cat_cols
    df = pd.concat([df, cate_df], axis=1)

    df[new_cat_cols] = df[new_cat_cols].astype("category")

    return df, feature_cols
