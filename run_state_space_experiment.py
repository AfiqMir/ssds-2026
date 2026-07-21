"""Leakage-safe per-station state-space experiment.

The experiment filters each station's deviation from a train-only seasonal
median with a robust local-level/local-linear-trend Kalman filter.  The final
state is projected over each validation/test horizon with exponential decay.
State configuration, correction strength, and blend weight are selected only
from the four rolling-origin OOF folds; API_TEST is not read here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_lightgbm_residual_experiment import apply_seasonal_lookup, rmse, seasonal_lookup
from run_station_aware_experiment import FOLDS, write_submission


CAT_DIR = Path("output_catboost_experiments")
STATION_DIR = Path("output_station_aware")
OUTPUT_DIR = Path("output_state_space")

TRAIN_PATH = CAT_DIR / "train_features.parquet"
TEST_PATH = CAT_DIR / "test_features.parquet"
RAW_SUBMISSION_PATH = CAT_DIR / "submission_rmse_selected.csv"
STATION_OOF_PATH = STATION_DIR / "oof_predictions.parquet"
STACK_SUBMISSION_PATH = STATION_DIR / "submission_station_stack.csv"
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

STEP_HOURS = 6.0
STATE_DECAY_DAYS = (90.0, 180.0, 365.0)
ALPHAS = np.arange(0.0, 1.2001, 0.05)
BLEND_WEIGHTS = np.arange(0.0, 0.3001, 0.01)
MAX_STATE_WORST_FOLD_DELTA = 0.02
MAX_BLEND_WORST_FOLD_DELTA = 0.005


@dataclass(frozen=True)
class FilterSpec:
    name: str
    level_noise_ratio: float
    slope_noise_ratio: float
    trend_damping_days: float


FILTER_SPECS = (
    FilterSpec("level_slow", 0.001, 0.0, 30.0),
    FilterSpec("level_medium", 0.010, 0.0, 30.0),
    FilterSpec("level_fast", 0.050, 0.0, 30.0),
    FilterSpec("trend_slow", 0.005, 0.000001, 30.0),
    FilterSpec("trend_medium", 0.020, 0.000010, 30.0),
)


def robust_scale(values: np.ndarray, floor: float = 0.05) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return floor
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(max((q3 - q1) / 1.349, floor))


def filter_station(
    station: pd.DataFrame, baseline: np.ndarray, spec: FilterSpec
) -> tuple[np.ndarray, float]:
    """Return final [level, slope] and a robust residual scale."""
    ordered = station.assign(_baseline=baseline).sort_values("datetime")
    measurement = (
        ordered["tma_mdpl"].to_numpy(dtype=float)
        - ordered["_baseline"].to_numpy(dtype=float)
    )
    timestamps = ordered["datetime"].to_numpy(dtype="datetime64[ns]")
    differences = np.diff(measurement)
    observation_scale = max(robust_scale(differences) / np.sqrt(2.0), 0.05)
    observation_variance = observation_scale**2

    initial = measurement[: min(30, len(measurement))]
    state = np.array([float(np.nanmedian(initial)), 0.0], dtype=float)
    covariance = np.diag([10.0 * observation_variance, observation_variance])
    if spec.slope_noise_ratio == 0.0:
        covariance[1, 1] = 0.0

    previous_time = timestamps[0]
    identity = np.eye(2)
    for value, timestamp in zip(measurement, timestamps, strict=True):
        elapsed_hours = float((timestamp - previous_time) / np.timedelta64(1, "h"))
        elapsed_steps = max(elapsed_hours / STEP_HOURS, 1.0 if timestamp != previous_time else 0.0)
        transition = np.array([[1.0, elapsed_steps], [0.0, 1.0]])
        process_noise = observation_variance * np.diag(
            [
                spec.level_noise_ratio * max(elapsed_steps, 1.0),
                spec.slope_noise_ratio * max(elapsed_steps, 1.0),
            ]
        )
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise
        if np.isfinite(value):
            innovation_variance = float(covariance[0, 0] + observation_variance)
            innovation = float(value - state[0])
            innovation_limit = 4.0 * np.sqrt(max(innovation_variance, 1e-12))
            innovation = float(np.clip(innovation, -innovation_limit, innovation_limit))
            gain = covariance[:, 0] / innovation_variance
            state = state + gain * innovation
            # Joseph form keeps the small 2x2 covariance positive semi-definite.
            update = identity - np.outer(gain, np.array([1.0, 0.0]))
            covariance = (
                update @ covariance @ update.T
                + np.outer(gain, gain) * observation_variance
            )
        previous_time = timestamp
    return state, robust_scale(measurement)


def forecast_basis(
    state: np.ndarray,
    residual_scale: float,
    cutoff: pd.Timestamp,
    target_times: pd.Series,
    spec: FilterSpec,
    decay_days: float,
) -> np.ndarray:
    horizon_steps = (
        (target_times - cutoff).dt.total_seconds().to_numpy(dtype=float)
        / (STEP_HOURS * 3_600.0)
    )
    horizon_steps = np.maximum(horizon_steps, 0.0)
    horizon_days = horizon_steps * STEP_HOURS / 24.0
    if spec.slope_noise_ratio > 0.0:
        rho = np.exp(-1.0 / (spec.trend_damping_days * 24.0 / STEP_HOURS))
        trend_multiplier = (1.0 - np.power(rho, horizon_steps)) / (1.0 - rho)
    else:
        trend_multiplier = np.zeros_like(horizon_steps)
    projected = state[0] + state[1] * trend_multiplier
    projected = np.clip(projected, -4.0 * residual_scale, 4.0 * residual_scale)
    return projected * np.exp(-horizon_days / decay_days)


def state_bases_for_split(
    fit: pd.DataFrame, target: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    lookup = seasonal_lookup(fit)
    fit = fit.copy()
    fit["seasonal_baseline"] = apply_seasonal_lookup(fit, lookup)
    result = target[["datetime", "nama_pos"]].copy()

    for spec in FILTER_SPECS:
        for decay_days in STATE_DECAY_DAYS:
            result[f"basis_{spec.name}_{int(decay_days)}d"] = np.nan
        for station, target_station in target.groupby("nama_pos", sort=True, observed=True):
            fit_station = fit[fit["nama_pos"] == station]
            if fit_station.empty:
                state = np.zeros(2, dtype=float)
                scale = 0.05
            else:
                state, scale = filter_station(
                    fit_station,
                    fit_station["seasonal_baseline"].to_numpy(dtype=float),
                    spec,
                )
            for decay_days in STATE_DECAY_DAYS:
                column = f"basis_{spec.name}_{int(decay_days)}d"
                result.loc[target_station.index, column] = forecast_basis(
                    state,
                    scale,
                    cutoff,
                    target_station["datetime"],
                    spec,
                    decay_days,
                )
    basis_columns = [column for column in result if column.startswith("basis_")]
    if result[basis_columns].isna().any().any():
        raise ValueError("State-space basis contains missing values")
    return result


def build_oof_bases(train: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold, train_end, _, _ in FOLDS:
        cutoff = pd.Timestamp(train_end)
        fit = train[train["datetime"] <= cutoff].copy()
        fold_target = reference[reference["fold"] == fold].copy()
        print(f"[{fold}] filtering {len(fit):,} rows for {len(fold_target):,} targets", flush=True)
        bases = state_bases_for_split(fit, fold_target, cutoff)
        bases.insert(0, "fold", fold)
        parts.append(bases)
    return pd.concat(parts, ignore_index=True)


def fold_scores(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    work = frame[["fold", "tma_mdpl"]].copy()
    work["prediction"] = prediction
    return {
        fold: rmse(group["tma_mdpl"], group["prediction"])
        for fold, group in work.groupby("fold", sort=False)
    }


def select_state(frame: pd.DataFrame, folds: set[str] | None = None) -> dict[str, object]:
    work = frame if folds is None else frame[frame["fold"].isin(folds)]
    control_scores = fold_scores(work, work["prediction_station_anchor"].to_numpy())
    truth = work["tma_mdpl"].to_numpy(dtype=float)
    raw = work["prediction_raw_rmse_selected"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    basis_columns = [column for column in work if column.startswith("basis_")]
    for basis_column in basis_columns:
        basis = work[basis_column].to_numpy(dtype=float)
        for alpha in ALPHAS:
            prediction = raw + alpha * basis
            scores = fold_scores(work, prediction)
            rows.append(
                {
                    "basis_column": basis_column,
                    "alpha": float(alpha),
                    "mean_fold_rmse": float(np.mean(list(scores.values()))),
                    "pooled_rmse": rmse(truth, prediction),
                    "worst_fold_delta_vs_station_anchor": float(
                        max(scores[name] - control_scores[name] for name in scores)
                    ),
                    **{f"rmse_{name}": value for name, value in scores.items()},
                }
            )
    grid = pd.DataFrame(rows)
    eligible = grid[
        grid["worst_fold_delta_vs_station_anchor"] <= MAX_STATE_WORST_FOLD_DELTA
    ]
    chosen = (eligible if not eligible.empty else grid).sort_values(
        ["mean_fold_rmse", "pooled_rmse"]
    ).iloc[0]
    return chosen.to_dict() | {"grid": grid}


def select_blend(
    frame: pd.DataFrame,
    state_selection: dict[str, object],
    folds: set[str] | None = None,
) -> dict[str, object]:
    work = frame if folds is None else frame[frame["fold"].isin(folds)]
    basis = work[str(state_selection["basis_column"])].to_numpy(dtype=float)
    candidate = (
        work["prediction_raw_rmse_selected"].to_numpy(dtype=float)
        + float(state_selection["alpha"]) * basis
    )
    control = work["prediction_station_stack"].to_numpy(dtype=float)
    control_scores = fold_scores(work, control)
    truth = work["tma_mdpl"].to_numpy(dtype=float)
    rows: list[dict[str, float]] = []
    for weight in BLEND_WEIGHTS:
        prediction = (1.0 - weight) * control + weight * candidate
        scores = fold_scores(work, prediction)
        rows.append(
            {
                "state_weight": float(weight),
                "station_stack_weight": float(1.0 - weight),
                "mean_fold_rmse": float(np.mean(list(scores.values()))),
                "pooled_rmse": rmse(truth, prediction),
                "worst_fold_delta_vs_station_stack": float(
                    max(scores[name] - control_scores[name] for name in scores)
                ),
                **{f"rmse_{name}": value for name, value in scores.items()},
            }
        )
    grid = pd.DataFrame(rows)
    eligible = grid[
        grid["worst_fold_delta_vs_station_stack"] <= MAX_BLEND_WORST_FOLD_DELTA
    ]
    chosen = (eligible if not eligible.empty else grid).sort_values(
        ["mean_fold_rmse", "pooled_rmse"]
    ).iloc[0]
    return chosen.to_dict() | {"grid": grid}


def leave_one_fold_out_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_folds = list(frame["fold"].drop_duplicates())
    for held_out in all_folds:
        training_folds = set(all_folds) - {held_out}
        state = select_state(frame, training_folds)
        blend = select_blend(frame, state, training_folds)
        held = frame[frame["fold"] == held_out]
        candidate = (
            held["prediction_raw_rmse_selected"].to_numpy(dtype=float)
            + float(state["alpha"])
            * held[str(state["basis_column"])].to_numpy(dtype=float)
        )
        control = held["prediction_station_stack"].to_numpy(dtype=float)
        prediction = (
            float(blend["station_stack_weight"]) * control
            + float(blend["state_weight"]) * candidate
        )
        rows.append(
            {
                "held_out_fold": held_out,
                "basis_column": state["basis_column"],
                "alpha": float(state["alpha"]),
                "state_weight": float(blend["state_weight"]),
                "held_out_rmse": rmse(held["tma_mdpl"], prediction),
                "control_rmse": rmse(held["tma_mdpl"], control),
            }
        )
    audit = pd.DataFrame(rows)
    audit["delta_vs_control"] = audit["held_out_rmse"] - audit["control_rmse"]
    return audit


def metric_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        scores = fold_scores(frame, frame[column].to_numpy(dtype=float))
        rows.extend(
            {"model": column.removeprefix("prediction_"), "fold": fold, "rmse": score}
            for fold, score in scores.items()
        )
        rows.append(
            {
                "model": column.removeprefix("prediction_"),
                "fold": "mean_fold_rmse",
                "rmse": float(np.mean(list(scores.values()))),
            }
        )
        rows.append(
            {
                "model": column.removeprefix("prediction_"),
                "fold": "pooled_rmse",
                "rmse": rmse(frame["tma_mdpl"], frame[column]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    required = [
        TRAIN_PATH,
        TEST_PATH,
        RAW_SUBMISSION_PATH,
        STATION_OOF_PATH,
        STACK_SUBMISSION_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    reference = pd.read_parquet(STATION_OOF_PATH)[
        [
            "fold",
            "datetime",
            "nama_pos",
            "tma_mdpl",
            "prediction_raw_rmse_selected",
            "prediction_station_anchor",
            "prediction_station_stack",
        ]
    ].copy()
    for frame in (train, test, reference):
        frame["datetime"] = pd.to_datetime(frame["datetime"])

    cache_path = OUTPUT_DIR / "oof_state_bases.parquet"
    if cache_path.exists():
        bases = pd.read_parquet(cache_path)
        print("Reusing cached OOF state bases", flush=True)
    else:
        bases = build_oof_bases(train, reference)
        bases.to_parquet(cache_path, index=False)
    oof = reference.merge(
        bases,
        on=["fold", "datetime", "nama_pos"],
        validate="one_to_one",
    )

    selected_state = select_state(oof)
    state_grid = selected_state.pop("grid")
    selected_blend = select_blend(oof, selected_state)
    blend_grid = selected_blend.pop("grid")
    audit = leave_one_fold_out_audit(oof)

    oof["prediction_state_space"] = (
        oof["prediction_raw_rmse_selected"]
        + float(selected_state["alpha"])
        * oof[str(selected_state["basis_column"])]
    )
    oof["prediction_state_space_blend"] = (
        float(selected_blend["station_stack_weight"])
        * oof["prediction_station_stack"]
        + float(selected_blend["state_weight"]) * oof["prediction_state_space"]
    )
    metrics = metric_rows(
        oof,
        [
            "prediction_station_anchor",
            "prediction_station_stack",
            "prediction_state_space",
            "prediction_state_space_blend",
        ],
    )

    test_bases = state_bases_for_split(train, test, train["datetime"].max())
    raw_submission = pd.read_csv(RAW_SUBMISSION_PATH)
    stack_submission = pd.read_csv(STACK_SUBMISSION_PATH)
    raw_test = test[["id"]].merge(raw_submission, on="id", validate="one_to_one")[
        "tma_mdpl"
    ].to_numpy(dtype=float)
    stack_test = test[["id"]].merge(stack_submission, on="id", validate="one_to_one")[
        "tma_mdpl"
    ].to_numpy(dtype=float)
    state_test = (
        raw_test
        + float(selected_state["alpha"])
        * test_bases[str(selected_state["basis_column"])].to_numpy(dtype=float)
    )
    blend_test = (
        float(selected_blend["station_stack_weight"]) * stack_test
        + float(selected_blend["state_weight"]) * state_test
    )

    write_submission(OUTPUT_DIR / "submission_state_space.csv", test["id"], state_test)
    write_submission(
        OUTPUT_DIR / "submission_state_space_blend.csv", test["id"], blend_test
    )
    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    state_grid.to_csv(OUTPUT_DIR / "state_grid.csv", index=False)
    blend_grid.to_csv(OUTPUT_DIR / "blend_grid.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "leave_one_fold_out_audit.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    test_bases.to_parquet(OUTPUT_DIR / "test_state_bases.parquet", index=False)

    station_audit = test[["nama_pos"]].copy()
    station_audit["prediction"] = blend_test
    prediction_ranges = station_audit.groupby("nama_pos")["prediction"].agg(["min", "max"])
    train_ranges = train.groupby("nama_pos")["tma_mdpl"].agg(
        train_min="min", train_max="max"
    )
    prediction_ranges.join(train_ranges).to_csv(
        OUTPUT_DIR / "submission_station_ranges.csv"
    )

    audit_pass = bool((audit["delta_vs_control"] <= MAX_BLEND_WORST_FOLD_DELTA).all())
    blend_improves = bool(
        float(selected_blend["mean_fold_rmse"])
        < float(
            metrics[
                (metrics["model"] == "station_stack")
                & (metrics["fold"] == "mean_fold_rmse")
            ]["rmse"].iloc[0]
        )
    )
    summary = {
        "selected_state": selected_state,
        "selected_blend": selected_blend,
        "leave_one_fold_out": audit.to_dict(orient="records"),
        "audit_pass": audit_pass,
        "blend_improves_mean_fold_rmse": blend_improves,
        "recommendation": (
            "state_space_blend_candidate"
            if audit_pass and blend_improves and float(selected_blend["state_weight"]) > 0
            else "retain_station_stack_control"
        ),
        "metrics": metrics.to_dict(orient="records"),
        "submission": str(OUTPUT_DIR / "submission_state_space_blend.csv"),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
