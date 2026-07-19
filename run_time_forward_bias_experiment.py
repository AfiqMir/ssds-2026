"""Leakage-safe time-forward residual calibration for the station stack.

The existing station-aware training script and its outputs are read-only inputs.
For every validation fold, calibration residuals come only from earlier-origin
OOF models and timestamps no later than that fold's training cutoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


STATION_DIR = Path("output_station_aware")
CAT_DIR = Path("output_catboost_experiments")
OUTPUT_DIR = Path("output_time_forward_bias")
OOF_PATH = STATION_DIR / "oof_predictions.parquet"
BASE_SUBMISSION_PATH = STATION_DIR / "submission_station_stack.csv"
TEST_FEATURES_PATH = CAT_DIR / "test_features.parquet"
SAMPLE_PATH = Path("sample_submission.csv")
PREDICTION_COLUMN = "prediction_station_stack"

FOLDS = (
    ("sep_2023", "2023-09-18 18:00:00", "2023-09-19 06:00:00"),
    ("may_2024", "2024-05-18 18:00:00", "2024-05-19 06:00:00"),
    ("sep_2024", "2024-09-18 18:00:00", "2024-09-19 06:00:00"),
    ("jan_2025", "2025-01-18 18:00:00", "2025-01-19 06:00:00"),
)


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def prepare_oof() -> pd.DataFrame:
    frame = pd.read_parquet(OOF_PATH)[
        ["fold", "datetime", "nama_pos", "tma_mdpl", PREDICTION_COLUMN]
    ].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    order = {fold: index for index, (fold, _, _) in enumerate(FOLDS)}
    frame["fold_order"] = frame["fold"].map(order)
    frame["residual"] = frame["tma_mdpl"] - frame[PREDICTION_COLUMN]
    if frame["fold_order"].isna().any():
        raise ValueError("Unknown fold in station-stack OOF")
    return frame


def calibration_history(
    oof: pd.DataFrame, target_fold: str, cutoff: pd.Timestamp
) -> pd.DataFrame:
    fold_order = {fold: index for index, (fold, _, _) in enumerate(FOLDS)}
    history = oof[
        (oof["fold_order"] < fold_order[target_fold])
        & (oof["datetime"] <= cutoff)
    ].copy()
    return history.sort_values("fold_order").drop_duplicates(
        ["datetime", "nama_pos"], keep="last"
    )


def residual_lookup(
    history: pd.DataFrame,
    cutoff: pd.Timestamp,
    half_life: float,
    scope: str,
    shrinkage_rows: float,
) -> tuple[pd.Series, pd.Series | None]:
    work = history.copy()
    age_days = (cutoff - work["datetime"]).dt.total_seconds() / 86_400.0
    work["weight"] = np.exp(-np.log(2.0) * age_days / half_life)
    lower = work.groupby("nama_pos", observed=True)["residual"].transform(
        lambda values: values.quantile(0.10)
    )
    upper = work.groupby("nama_pos", observed=True)["residual"].transform(
        lambda values: values.quantile(0.90)
    )
    work["robust_residual"] = work["residual"].clip(lower, upper)
    station = work.groupby("nama_pos", observed=True).apply(
        lambda group: np.average(group["robust_residual"], weights=group["weight"]),
        include_groups=False,
    )
    if scope == "station":
        return station, None

    work["month"] = work["datetime"].dt.month
    grouped = work.groupby(["nama_pos", "month"], observed=True)
    local = grouped.apply(
        lambda group: np.average(group["robust_residual"], weights=group["weight"]),
        include_groups=False,
    )
    effective_rows = grouped["weight"].sum() / grouped["weight"].max()
    station_for_local = local.index.get_level_values("nama_pos").map(station)
    fraction = effective_rows / (effective_rows + shrinkage_rows)
    shrunk = fraction * local + (1.0 - fraction) * station_for_local.to_numpy()
    return station, shrunk


def map_correction(
    target: pd.DataFrame,
    station: pd.Series,
    station_month: pd.Series | None,
) -> np.ndarray:
    fallback = target["nama_pos"].map(station).fillna(0.0).to_numpy(dtype=float)
    if station_month is None:
        return fallback
    keys = pd.MultiIndex.from_arrays(
        [target["nama_pos"], target["datetime"].dt.month],
        names=["nama_pos", "month"],
    )
    local = station_month.reindex(keys).to_numpy(dtype=float)
    return np.where(np.isfinite(local), local, fallback)


def precompute_oof_corrections(oof: pd.DataFrame) -> dict[tuple, np.ndarray]:
    corrections: dict[tuple, np.ndarray] = {}
    settings = [
        (half_life, scope, shrinkage)
        for half_life in (90.0, 180.0, 365.0, 730.0)
        for scope, shrinkages in (("station", (0.0,)), ("station_month", (30.0, 90.0)))
        for shrinkage in shrinkages
    ]
    for half_life, scope, shrinkage in settings:
        values = np.zeros(len(oof), dtype=float)
        for fold, cutoff_text, _ in FOLDS:
            target = oof[oof["fold"] == fold]
            history = calibration_history(oof, fold, pd.Timestamp(cutoff_text))
            if history.empty:
                continue
            station, station_month = residual_lookup(
                history, pd.Timestamp(cutoff_text), half_life, scope, shrinkage
            )
            values[target.index] = map_correction(target, station, station_month)
        corrections[(half_life, scope, shrinkage)] = values
    return corrections


def select_configuration(
    oof: pd.DataFrame, corrections: dict[tuple, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, object]]:
    truth = oof["tma_mdpl"].to_numpy(dtype=float)
    base = oof[PREDICTION_COLUMN].to_numpy(dtype=float)
    base_fold = {
        fold: rmse(group["tma_mdpl"], group[PREDICTION_COLUMN])
        for fold, group in oof.groupby("fold", sort=False)
    }
    base_mean = float(np.mean(list(base_fold.values())))
    base_pooled = rmse(truth, base)
    rows: list[dict[str, object]] = []
    starts = {
        fold: pd.Timestamp(valid_start) for fold, _, valid_start in FOLDS
    }
    horizon = np.array(
        [
            (timestamp - starts[fold]).total_seconds() / 86_400.0
            for fold, timestamp in zip(oof["fold"], oof["datetime"])
        ],
        dtype=float,
    )
    for (half_life, scope, shrinkage), correction in corrections.items():
        for tau_days in (180.0, 365.0, 730.0, 1e9):
            basis = correction * np.exp(-horizon / tau_days)
            for alpha in np.linspace(0.0, 1.0, 21):
                prediction = base + alpha * basis
                work = oof[["fold", "tma_mdpl"]].copy()
                work["prediction"] = prediction
                scores = {
                    fold: rmse(group["tma_mdpl"], group["prediction"])
                    for fold, group in work.groupby("fold", sort=False)
                }
                deltas = {
                    fold: scores[fold] - base_fold[fold] for fold in scores
                }
                pooled = rmse(truth, prediction)
                rows.append(
                    {
                        "half_life_days": half_life,
                        "scope": scope,
                        "shrinkage_rows": shrinkage,
                        "tau_days": tau_days,
                        "alpha": float(alpha),
                        "mean_fold_rmse": float(np.mean(list(scores.values()))),
                        "pooled_rmse": pooled,
                        "mean_fold_delta": float(np.mean(list(deltas.values()))),
                        "pooled_delta": pooled - base_pooled,
                        "worst_fold_delta": float(max(deltas.values())),
                        **{f"rmse_{fold}": score for fold, score in scores.items()},
                        **{f"delta_{fold}": delta for fold, delta in deltas.items()},
                    }
                )
    grid = pd.DataFrame(rows)
    robust = grid[
        (grid["mean_fold_delta"] < 0.0)
        & (grid["pooled_delta"] < 0.0)
        & (grid["worst_fold_delta"] <= 0.01)
        & (grid["delta_sep_2023"] <= 0.0)
        & (grid["delta_sep_2024"] <= 0.01)
    ].sort_values(["mean_fold_rmse", "pooled_rmse", "worst_fold_delta"])
    if robust.empty:
        selected = grid[grid["alpha"] == 0.0].iloc[0]
        note = "No robust correction; retained station-stack control."
    else:
        selected = robust.iloc[0]
        note = "Selected using time-forward OOF only under robustness constraints."
    result = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in selected.to_dict().items()
    }
    result.update(
        {
            "selection_note": note,
            "control_mean_fold_rmse": base_mean,
            "control_pooled_rmse": base_pooled,
        }
    )
    return grid, result


def create_submission(
    oof: pd.DataFrame, selection: dict[str, object]
) -> tuple[Path, pd.DataFrame]:
    test = pd.read_parquet(TEST_FEATURES_PATH)[
        ["id", "datetime", "nama_pos"]
    ].copy()
    test["datetime"] = pd.to_datetime(test["datetime"])
    cutoff = pd.read_parquet(CAT_DIR / "train_features.parquet", columns=["datetime"])[
        "datetime"
    ].max()
    history = oof[oof["datetime"] <= cutoff].sort_values("fold_order")
    history = history.drop_duplicates(["datetime", "nama_pos"], keep="last")
    station, station_month = residual_lookup(
        history,
        cutoff,
        float(selection["half_life_days"]),
        str(selection["scope"]),
        float(selection["shrinkage_rows"]),
    )
    raw_correction = map_correction(test, station, station_month)
    horizon = (test["datetime"] - test["datetime"].min()).dt.total_seconds() / 86_400.0
    correction = (
        float(selection["alpha"])
        * raw_correction
        * np.exp(-horizon.to_numpy(dtype=float) / float(selection["tau_days"]))
    )
    base = pd.read_csv(BASE_SUBMISSION_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    if not base["id"].equals(sample["id"]) or not test["id"].equals(sample["id"]):
        raise ValueError("Input order does not match sample submission")
    prediction = base["tma_mdpl"].to_numpy(dtype=float) + correction
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite calibrated prediction")
    output = pd.DataFrame({"id": sample["id"], "tma_mdpl": prediction})
    path = OUTPUT_DIR / "submission_time_forward_bias.csv"
    output.to_csv(path, index=False)
    details = test.copy()
    details["base_prediction"] = base["tma_mdpl"].to_numpy(dtype=float)
    details["raw_correction"] = raw_correction
    details["applied_correction"] = correction
    details["prediction"] = prediction
    return path, details


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = prepare_oof().reset_index(drop=True)
    corrections = precompute_oof_corrections(oof)
    grid, selection = select_configuration(oof, corrections)
    key = (
        float(selection["half_life_days"]),
        str(selection["scope"]),
        float(selection["shrinkage_rows"]),
    )
    starts = {fold: pd.Timestamp(start) for fold, _, start in FOLDS}
    horizon = np.array(
        [
            (timestamp - starts[fold]).total_seconds() / 86_400.0
            for fold, timestamp in zip(oof["fold"], oof["datetime"])
        ]
    )
    oof["raw_bias_correction"] = corrections[key]
    oof["prediction_time_forward_bias"] = (
        oof[PREDICTION_COLUMN].to_numpy(dtype=float)
        + float(selection["alpha"])
        * corrections[key]
        * np.exp(-horizon / float(selection["tau_days"]))
    )
    submission_path, details = create_submission(oof, selection)
    selection["submission"] = str(submission_path)
    grid.to_csv(OUTPUT_DIR / "calibration_grid.csv", index=False)
    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    details.to_parquet(OUTPUT_DIR / "test_corrections.parquet", index=False)
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
