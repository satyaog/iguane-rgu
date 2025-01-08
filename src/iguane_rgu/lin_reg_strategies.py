import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from iguane_rgu.gpu_data import GPU_DATA, NORMALIZED_GPU_DATA, REF_GPU
from iguane_rgu.milabench_data import compute_score


def prepare_perf_ram_df(df: pd.DataFrame):
    plot_df = []

    for bench in sorted(df.index.get_level_values("bench").unique()):
        bench_data = df.loc[:, bench, :]

        if bench == "lightning-gpus":
            bench_data = bench_data[bench_data["ngpu"] == 4]

        if bench == "ppo":
            bench_data = bench_data[
                bench_data.index.get_level_values("batch_size") != 7808
            ]

        bench_data.reset_index(inplace=True)
        bench_data = bench_data.loc[
            :, ("gpu", "perf", "batch_size", "peak_memory", "weight")
        ]
        bench_data["bench"] = bench
        plot_df.append(bench_data)

    plot_df = pd.concat(plot_df)

    return plot_df


def mean_per_bench_lin_reg(df: pd.DataFrame, scores: pd.DataFrame):
    column = "peak_memory"
    ref_column = "memgb"

    mean_coef = []
    sum_weights = 0

    def linreg(X: pd.DataFrame, y: pd.DataFrame):
        # Create a Linear Regression model
        model = LinearRegression(fit_intercept=False, positive=False)

        # Fit the model on the data
        model.fit(X, y)

        return model

    for bench in sorted(df["bench"].unique()):
        _data = df.loc[df["bench"] == bench]
        _data = _data.loc[_data[column].notna()]
        assert _data["weight"].iloc[0] == _data["weight"].mean()
        sum_weights += _data["weight"].iloc[0]

    for bench in sorted(df["bench"].unique()):
        _data = df.loc[df["bench"] == bench]

        if not _data[_data[column].isna()].empty:
            logging.debug("\n" + str(_data[_data[column].isna()]))

        _data = _data.loc[_data[column].notna()]

        bench_coef = []
        bench_intercept = []
        for gpu in df["gpu"].unique():
            _gpu_data = _data[_data["gpu"] == gpu]

            if _gpu_data.empty:
                bench_coef.append(0)
                bench_intercept.append(0)
                continue

            model = linreg(
                _gpu_data.loc[:, (column,)]
                / (GPU_DATA.loc[REF_GPU, ref_column] * 1024),
                np.exp(
                    _gpu_data.apply(
                        compute_score,
                        axis=1,
                    )
                    * _data["weight"].iloc[0]
                    / sum_weights
                )
                / scores[REF_GPU],
            )

            bench_coef.append(model.coef_[0])
            bench_intercept.append(model.intercept_)

        bench_coef = sum(bench_coef) / len(bench_coef)
        bench_intercept = sum(bench_intercept) / len(bench_intercept)

        _ref_data = _data[_data["gpu"] == REF_GPU]

        model_ref = linreg(
            _ref_data.loc[:, (column,)] / (GPU_DATA.loc[REF_GPU, ref_column] * 1024),
            np.exp(
                _ref_data.apply(compute_score, axis=1)
                * _data["weight"].iloc[0]
                / sum_weights
            )
            / scores[REF_GPU],
        )

        logging.debug(f"{bench}\t\ty = {bench_coef}x + {bench_intercept}")
        logging.debug(
            f"{bench}\t(ref)\ty = {model_ref.coef_[0]}x + {model_ref.intercept_}"
        )

        # Coefficients (beta values)
        mean_coef.append(bench_coef)

    return np.mean(mean_coef)


def _linreg(GPUS, normalized_scores, dropped_columns):
    # Features (independent variables)
    X = GPUS

    y = normalized_scores

    # Create a Linear Regression model
    model = LinearRegression(fit_intercept=False, positive=True)

    # Fit the model on the data
    model.fit(X, y)

    # Coefficients (beta values)
    coefficients = pd.DataFrame(
        [model.coef_] * 4,
        index=["raw", "sum to 1", "floor 0.05", "floor sum to 1"],
        columns=X.columns.tolist(),
    )

    if "tf32" not in coefficients:
        coefficients["tf32"] = coefficients["fp16"] / 2
        coefficients["fp16"] = coefficients["fp16"] / 2

    return coefficients


def _linreg_custom(GPUS, normalized_scores, dropped_columns):
    # Features (independent variables)
    GPUS["fp16+fp32+tf32"] = 0.0
    GPUS["memgb+membw"] = 0.0

    for column in ["fp16", "fp32", "tf32"]:
        if column in dropped_columns:
            continue
        GPUS["fp16+fp32+tf32"] = GPUS["fp16+fp32+tf32"] + GPUS[column]

    for column in ["memgb", "membw"]:
        if column in dropped_columns:
            continue
        GPUS["memgb+membw"] = GPUS["memgb+membw"] + GPUS[column]

    for column in ["fp16", "fp32", "tf32", "memgb", "membw"]:
        if column in dropped_columns:
            continue
        GPUS = GPUS.drop(columns=[column])

    X = GPUS

    y = normalized_scores

    # Create a Linear Regression model
    model = LinearRegression(fit_intercept=False, positive=True)

    # Fit the model on the data
    model.fit(X, y)

    # Coefficients (beta values)
    coefficients = pd.DataFrame(
        [model.coef_] * 4,
        index=["raw", "sum to 1", "floor 0.05", "floor sum to 1"],
        columns=X.columns.tolist(),
    )

    coefficients["fp16"] = coefficients["fp16+fp32+tf32"] * 2 / 5
    coefficients["fp32"] = coefficients["fp16+fp32+tf32"] * 1 / 5
    coefficients["tf32"] = coefficients["fp16+fp32+tf32"] * 2 / 5

    coefficients["memgb"] = coefficients["memgb+membw"] / 2
    coefficients["membw"] = coefficients["memgb+membw"] / 2

    coefficients = coefficients.drop(columns=["fp16+fp32+tf32", "memgb+membw"])

    return coefficients


def _linreg_ram(GPUS, normalized_scores, dropped_columns, mean_m_ram):
    ram_coef = mean_m_ram

    # Features (independent variables)
    X = GPUS

    # Target variable (dependent variable)
    y = normalized_scores - ram_coef * NORMALIZED_GPU_DATA["memgb"]

    # Create a Linear Regression model
    model = LinearRegression(fit_intercept=False, positive=True)

    # Fit the model on the data
    model.fit(X, y)

    # Coefficients (beta values)
    coefficients = pd.DataFrame(
        [model.coef_] * 4,
        index=["raw", "sum to 1", "floor 0.05", "floor sum to 1"],
        columns=X.columns.tolist(),
    )

    if "tf32" not in coefficients:
        coefficients["tf32"] = coefficients["fp16"] / 2
        coefficients["fp16"] = coefficients["fp16"] / 2

    coefficients["memgb"] = ram_coef

    return coefficients


def _linreg_all_minval(GPUS, normalized_scores, dropped_columns):
    # Features (independent variables)
    X = GPUS

    y = normalized_scores

    for dropped_column in dropped_columns:
        y = y - 0.05 * NORMALIZED_GPU_DATA[dropped_column]

    # Create a Linear Regression model
    model = LinearRegression(fit_intercept=False, positive=True)

    # Fit the model on the data
    model.fit(X, y)

    # Coefficients (beta values)
    coefficients = pd.DataFrame(
        [model.coef_] * 4,
        index=["raw", "sum to 1", "floor 0.05", "floor sum to 1"],
        columns=X.columns.tolist(),
    )

    if "tf32" not in coefficients:
        coefficients["tf32"] = coefficients["fp16"] / 2
        coefficients["fp16"] = coefficients["fp16"] / 2

    for dropped_column in dropped_columns:
        coefficients[dropped_column] = 0.05

    return coefficients
