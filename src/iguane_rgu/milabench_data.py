import logging
from pathlib import Path
import numpy as np
import pandas as pd

from iguane_rgu.gpu_data import GPU_ALIASES, REF_GPU
from iguane_rgu.milabench_weights import WEIGHTS


def compute_score(df: pd.DataFrame):
    weights = df["weight"]

    if len(df.shape) > 1:
        sum_weights = weights.sum()

    else:
        sum_weights = weights

    return (np.log(df["perf"] + 1) * weights).sum() / sum_weights


def compute_optimal_scores(df: pd.DataFrame, normalize=False):
    df = df.copy(deep=True)

    for bench in sorted(df.index.get_level_values("bench").unique()):
        for gpu in df.index.get_level_values("gpu").unique():
            bench_data = df.loc[gpu, bench, :]
            df.drop([(gpu, bench, _idx) for _idx in bench_data.index], inplace=True)

            # Ignore lightning-gpus with less than 4 GPUs used
            if bench == "lightning-gpus":
                bench_data = bench_data[bench_data["ngpu"] == 4]

            # From all batch sizes, ppo has an outlier high score with 7808,
            # which is the smallest batch size tested. Since lower batch sizes
            # were not tested, we chose to ignore this particular batch size
            if bench == "ppo":
                bench_data = bench_data[
                    bench_data.index.get_level_values("batch_size") != 7808
                ]

            if bench_data.empty or bench_data["perf"].isna().all():
                missing_bench = pd.Series(
                    {
                        "gpu": gpu,
                        "bench": bench,
                        "batch_size": 0,
                        "perf": 0.0,
                        "weight": df.loc[:, bench, :]["weight"].max(),
                    },
                )
                df = pd.concat(
                    [
                        df,
                        missing_bench.to_frame().T.set_index(
                            ["gpu", "bench", "batch_size"]
                        ),
                    ]
                )
                continue

            if len(bench_data.loc[bench_data["perf"].idxmax()].shape) > 1:
                best = bench_data.loc[bench_data["perf"].idxmax()].mean()
            else:
                best = bench_data.loc[bench_data["perf"].idxmax()].copy()
            best["gpu"] = gpu
            best["bench"] = bench
            best["batch_size"] = bench_data["perf"].idxmax()

            df = pd.concat(
                [df, best.to_frame().T.set_index(["gpu", "bench", "batch_size"])]
            )

    df.sort_index(level=2, inplace=True)
    df.sort_index(level=1, inplace=True)
    df.sort_index(level=0, inplace=True)

    optim_df = df.infer_objects()

    logging.debug("\n" + str(optim_df))

    scores = optim_df.groupby("gpu").apply(compute_score)
    scores = np.exp(scores)

    if normalize:
        return scores / scores.loc[REF_GPU]

    return scores


def load_milabench_data(report_file: str | Path):
    report_file = Path(report_file)

    df = pd.read_csv(str(report_file))
    # drop weight == 0
    df = df[df["weight"] != 0]

    for bench, weight in WEIGHTS.items():
        if df[df["bench"] == bench].empty:
            logging.warning(f"Could not find bench: {(bench, weight)}")
            continue
        df.loc[df["bench"] == bench, "weight"] = weight

    # drop weight == 0
    df = df[df["weight"] != 0]
    df.set_index(["gpu", "bench", "batch_size"], inplace=True)

    # convert str
    df["sem%"] = [float(sem[:-1]) / 100 for sem in df["sem%"]]
    df["std%"] = [float(std[:-1]) / 100 for std in df["std%"]]

    for gpu, aliases in GPU_ALIASES.items():
        for alias in aliases:
            aliases_data = df[df.index.get_level_values("gpu") == alias].copy(deep=True)

            df.drop(aliases_data.index, inplace=True)
            aliases_data.reset_index(inplace=True)
            aliases_data.loc[:, "gpu"] = gpu
            aliases_data.set_index(["gpu", "bench", "batch_size"], inplace=True)
            aliases_data.drop(
                aliases_data.index[[i in df.index for i in aliases_data.index]],
                inplace=True,
            )

            df = pd.concat([df, aliases_data])

    return df.sort_index(level=2).sort_index(level=1).sort_index(level=0)
