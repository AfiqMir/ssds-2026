"""Leakage-safe global/local LightGBM residual hybrid experiment.

The local component fits one compact LightGBM model per station on residuals
from a station/month/hour seasonal median.  It is compared and blended with the
existing station-anchor CatBoost and global LightGBM residual predictions on the
same four rolling-origin folds.  API_TEST is deliberately not used for tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_lightgbm_residual_experiment import (
    apply_seasonal_lookup,
    isolated_spike_weights,
    prepare_categories,
    rmse,
    seasonal_lookup,
)
from run_station_aware_experiment import FOLDS, tune_station_anchor, write_submission


CAT_DIR = Path("output_catboost_experiments")
GLOBAL_DIR = Path("output_lightgbm_residual")
STATION_DIR = Path("output_station_aware")
OUTPUT_DIR = Path("output_global_local_hybrid")

TRAIN_PATH = CAT_DIR / "train_features.parquet"
TEST_PATH = CAT_DIR / "test_features.parquet"
FEATURE_METADATA_PATH = CAT_DIR / "feature_metadata.json"
CAT_OOF_PATH = CAT_DIR / "oof_predictions.parquet"
GLOBAL_OOF_PATH = GLOBAL_DIR / "oof_predictions.parquet"
GLOBAL_SUBMISSION_PATH = GLOBAL_DIR / "submission_lightgbm_residual.csv"
STATION_ANCHOR_SUBMISSION_PATH = STATION_DIR / "submission_station_anchor.csv"
STATION_SUMMARY_PATH = STATION_DIR / "experiment_summary.json"

MIN_STATION_ROWS = 180
MAX_ESTIMATORS = 1_500
EARLY_STOPPING_ROUNDS = 100
RANDOM_STATE = 17
WEIGHT_STEP = 0.01
MAX_NON_CAT_WEIGHT = 0.60
MAX_WORST_FOLD_DELTA = 0.01


def make_local_model(n_estimators: int = MAX_ESTIMATORS) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=6,
        min_child_samples=30,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.20,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def local_feature_spec(metadata: dict[str, object]) -> tuple[list[str], list[str]]:
    # nama_pos is constant within a local model. Static station descriptors are
    # also constant, but leaving numeric constants in is harmless; LightGBM
    # simply never splits on them.
    features = [name for name in metadata["features"] if name != "nama_pos"]
    categorical = [name for name in metadata["categorical"] if name != "nama_pos"]
    return features, categorical


def validate_local_models(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    iteration_rows: list[dict[str, object]] = []
    for fold, train_end, valid_start, valid_end in FOLDS:
        fit = train[train["datetime"] <= pd.Timestamp(train_end)].copy()
        valid = train[
            train["datetime"].between(
                pd.Timestamp(valid_start), pd.Timestamp(valid_end), inclusive="both"
            )
        ].copy()
        counts = fit.groupby("nama_pos", observed=True).size()
        eligible = counts[counts >= MIN_STATION_ROWS].index
        fit = fit[fit["nama_pos"].isin(eligible)].copy()
        valid = valid[valid["nama_pos"].isin(eligible)].copy()
        lookup = seasonal_lookup(fit)
        fit_baseline = pd.Series(apply_seasonal_lookup(fit, lookup), index=fit.index)
        valid_baseline = pd.Series(apply_seasonal_lookup(valid, lookup), index=valid.index)
        fold_parts: list[pd.DataFrame] = []
        print(
            f"[{fold}] local models: fit={len(fit):,} valid={len(valid):,} "
            f"stations={len(eligible)}",
            flush=True,
        )
        for station, valid_station in valid.groupby("nama_pos", sort=True, observed=True):
            fit_station = fit[fit["nama_pos"] == station]
            weights, _ = isolated_spike_weights(fit_station)
            y_fit = (
                fit_station["tma_mdpl"].to_numpy(dtype=float)
                - fit_baseline.loc[fit_station.index].to_numpy(dtype=float)
            )
            y_valid = (
                valid_station["tma_mdpl"].to_numpy(dtype=float)
                - valid_baseline.loc[valid_station.index].to_numpy(dtype=float)
            )
            model = make_local_model()
            model.fit(
                fit_station[features],
                y_fit,
                sample_weight=weights,
                categorical_feature=categorical,
                eval_set=[(valid_station[features], y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            prediction = (
                valid_baseline.loc[valid_station.index].to_numpy(dtype=float)
                + model.predict(valid_station[features])
            )
            result = valid_station[["datetime", "nama_pos", "tma_mdpl"]].copy()
            result.insert(0, "fold", fold)
            result["prediction_local_lightgbm"] = prediction
            fold_parts.append(result)
            iteration_rows.append(
                {
                    "fold": fold,
                    "nama_pos": station,
                    "fit_rows": len(fit_station),
                    "valid_rows": len(valid_station),
                    "best_iteration": int(model.best_iteration_),
                    "rmse": rmse(valid_station["tma_mdpl"], prediction),
                }
            )
        fold_result = pd.concat(fold_parts, ignore_index=True)
        print(
            f"  local LightGBM RMSE={rmse(fold_result['tma_mdpl'], fold_result['prediction_local_lightgbm']):.6f}",
            flush=True,
        )
        parts.append(fold_result)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(iteration_rows)


def score_prediction(frame: pd.DataFrame, column: str) -> dict[str, float]:
    scores = {
        fold: rmse(group["tma_mdpl"], group[column])
        for fold, group in frame.groupby("fold", sort=False)
    }
    return {
        **scores,
        "mean_fold_rmse": float(np.mean(list(scores.values()))),
        "pooled_rmse": rmse(frame["tma_mdpl"], frame[column]),
    }


def blend_grid(frame: pd.DataFrame, folds: set[str] | None = None) -> pd.DataFrame:
    work = frame if folds is None else frame[frame["fold"].isin(folds)]
    base_by_fold = {
        fold: rmse(group["tma_mdpl"], group["prediction_station_stack"])
        for fold, group in work.groupby("fold", sort=False)
    }
    # For error = cat_error + wg * (global-cat) + wl * (local-cat), MSE is a
    # quadratic in the two weights.  Precomputing its six terms avoids scanning
    # all OOF rows for every point in the grid and keeps LOFO auditing cheap.
    quadratic: dict[str, dict[str, float]] = {}
    for fold, group in work.groupby("fold", sort=False):
        error = (
            group["prediction_station_anchor"].to_numpy(dtype=float)
            - group["tma_mdpl"].to_numpy(dtype=float)
        )
        global_delta = (
            group["prediction_global_lightgbm"].to_numpy(dtype=float)
            - group["prediction_station_anchor"].to_numpy(dtype=float)
        )
        local_delta = (
            group["prediction_local_lightgbm"].to_numpy(dtype=float)
            - group["prediction_station_anchor"].to_numpy(dtype=float)
        )
        quadratic[fold] = {
            "n": float(len(group)),
            "e2": float(np.dot(error, error)),
            "g2": float(np.dot(global_delta, global_delta)),
            "l2": float(np.dot(local_delta, local_delta)),
            "eg": float(np.dot(error, global_delta)),
            "el": float(np.dot(error, local_delta)),
            "gl": float(np.dot(global_delta, local_delta)),
        }
    rows: list[dict[str, float]] = []
    weights = np.arange(0.0, MAX_NON_CAT_WEIGHT + WEIGHT_STEP / 2, WEIGHT_STEP)
    for global_weight in weights:
        for local_weight in weights:
            if global_weight + local_weight > MAX_NON_CAT_WEIGHT + 1e-12:
                continue
            cat_weight = 1.0 - global_weight - local_weight
            fold_sse = {
                fold: (
                    values["e2"]
                    + global_weight**2 * values["g2"]
                    + local_weight**2 * values["l2"]
                    + 2.0 * global_weight * values["eg"]
                    + 2.0 * local_weight * values["el"]
                    + 2.0 * global_weight * local_weight * values["gl"]
                )
                for fold, values in quadratic.items()
            }
            fold_scores = {
                fold: float(np.sqrt(max(0.0, sse / quadratic[fold]["n"])))
                for fold, sse in fold_sse.items()
            }
            total_n = sum(values["n"] for values in quadratic.values())
            pooled = float(np.sqrt(max(0.0, sum(fold_sse.values()) / total_n)))
            rows.append(
                {
                    "catboost_weight": float(cat_weight),
                    "global_lightgbm_weight": float(global_weight),
                    "local_lightgbm_weight": float(local_weight),
                    "mean_fold_rmse": float(np.mean(list(fold_scores.values()))),
                    "pooled_rmse": pooled,
                    "worst_fold_delta_vs_control": float(
                        max(fold_scores[f] - base_by_fold[f] for f in fold_scores)
                    ),
                    **{f"rmse_{fold}": value for fold, value in fold_scores.items()},
                }
            )
    return pd.DataFrame(rows)


def select_blend(grid: pd.DataFrame) -> dict[str, float]:
    eligible = grid[grid["worst_fold_delta_vs_control"] <= MAX_WORST_FOLD_DELTA]
    chosen = (eligible if not eligible.empty else grid).sort_values(
        ["mean_fold_rmse", "pooled_rmse"]
    ).iloc[0]
    return {key: float(value) for key, value in chosen.items()}


def leave_one_fold_out_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_folds = list(frame["fold"].drop_duplicates())
    for held_out in all_folds:
        selected = select_blend(blend_grid(frame, set(all_folds) - {held_out}))
        held = frame[frame["fold"] == held_out]
        prediction = (
            selected["catboost_weight"] * held["prediction_station_anchor"]
            + selected["global_lightgbm_weight"] * held["prediction_global_lightgbm"]
            + selected["local_lightgbm_weight"] * held["prediction_local_lightgbm"]
        )
        rows.append(
            {
                "held_out_fold": held_out,
                "catboost_weight": selected["catboost_weight"],
                "global_lightgbm_weight": selected["global_lightgbm_weight"],
                "local_lightgbm_weight": selected["local_lightgbm_weight"],
                "held_out_rmse": rmse(held["tma_mdpl"], prediction),
                "control_rmse": rmse(
                    held["tma_mdpl"], held["prediction_station_stack"]
                ),
            }
        )
    audit = pd.DataFrame(rows)
    audit["delta_vs_control"] = audit["held_out_rmse"] - audit["control_rmse"]
    return audit


def train_final_local(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    iterations: pd.DataFrame,
) -> np.ndarray:
    lookup = seasonal_lookup(train)
    train_baseline = pd.Series(apply_seasonal_lookup(train, lookup), index=train.index)
    test_baseline = pd.Series(apply_seasonal_lookup(test, lookup), index=test.index)
    median_iterations = iterations.groupby("nama_pos")["best_iteration"].median()
    output = pd.Series(np.nan, index=test.index, dtype=float)
    for station, test_station in test.groupby("nama_pos", sort=True, observed=True):
        fit_station = train[train["nama_pos"] == station]
        if len(fit_station) < MIN_STATION_ROWS or station not in median_iterations.index:
            output.loc[test_station.index] = test_baseline.loc[test_station.index]
            continue
        weights, _ = isolated_spike_weights(fit_station)
        target = (
            fit_station["tma_mdpl"].to_numpy(dtype=float)
            - train_baseline.loc[fit_station.index].to_numpy(dtype=float)
        )
        n_estimators = max(1, int(round(median_iterations.loc[station])))
        model = make_local_model(n_estimators=n_estimators)
        model.fit(
            fit_station[features],
            target,
            sample_weight=weights,
            categorical_feature=categorical,
        )
        output.loc[test_station.index] = (
            test_baseline.loc[test_station.index].to_numpy(dtype=float)
            + model.predict(test_station[features])
        )
    if not np.isfinite(output).all():
        raise ValueError("Local final prediction contains NaN or infinity")
    return output.to_numpy(dtype=float)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        TRAIN_PATH,
        TEST_PATH,
        FEATURE_METADATA_PATH,
        CAT_OOF_PATH,
        GLOBAL_OOF_PATH,
        GLOBAL_SUBMISSION_PATH,
        STATION_ANCHOR_SUBMISSION_PATH,
        STATION_SUMMARY_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    features, categorical = local_feature_spec(metadata)
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    train, test = prepare_categories(train, test, metadata["categorical"])

    local_oof_cache = OUTPUT_DIR / "local_oof_cache.parquet"
    iteration_cache = OUTPUT_DIR / "local_model_iterations_cache.csv"
    if local_oof_cache.exists() and iteration_cache.exists():
        print("Loading cached local OOF predictions", flush=True)
        local_oof = pd.read_parquet(local_oof_cache)
        iterations = pd.read_csv(iteration_cache)
    else:
        local_oof, iterations = validate_local_models(train, features, categorical)
        local_oof.to_parquet(local_oof_cache, index=False)
        iterations.to_csv(iteration_cache, index=False)
    cat_oof, _ = tune_station_anchor(pd.read_parquet(CAT_OOF_PATH))
    global_oof = pd.read_parquet(GLOBAL_OOF_PATH)[
        ["fold", "datetime", "nama_pos", "prediction_raw_residual"]
    ].rename(columns={"prediction_raw_residual": "prediction_global_lightgbm"})
    oof = cat_oof.merge(
        global_oof,
        on=["fold", "datetime", "nama_pos"],
        validate="one_to_one",
    ).merge(
        local_oof,
        on=["fold", "datetime", "nama_pos", "tma_mdpl"],
        how="left",
        validate="one_to_one",
    )
    if len(oof) != len(cat_oof):
        raise ValueError("OOF components do not align exactly")
    missing_local = oof["prediction_local_lightgbm"].isna()
    if missing_local.any():
        print(
            f"Local fallback to global LightGBM for {int(missing_local.sum()):,} OOF rows",
            flush=True,
        )
        oof.loc[missing_local, "prediction_local_lightgbm"] = oof.loc[
            missing_local, "prediction_global_lightgbm"
        ]
    station_summary = json.loads(STATION_SUMMARY_PATH.read_text(encoding="utf-8"))
    control_global_weight = float(
        station_summary["selected_blend"]["lightgbm_weight"]
    )
    oof["prediction_station_stack"] = (
        (1.0 - control_global_weight) * oof["prediction_station_anchor"]
        + control_global_weight * oof["prediction_global_lightgbm"]
    )

    grid = blend_grid(oof)
    selected = select_blend(grid)
    oof["prediction_global_local_hybrid"] = (
        selected["catboost_weight"] * oof["prediction_station_anchor"]
        + selected["global_lightgbm_weight"] * oof["prediction_global_lightgbm"]
        + selected["local_lightgbm_weight"] * oof["prediction_local_lightgbm"]
    )
    audit = leave_one_fold_out_audit(oof)
    metric_columns = [
        "prediction_station_anchor",
        "prediction_global_lightgbm",
        "prediction_local_lightgbm",
        "prediction_station_stack",
        "prediction_global_local_hybrid",
    ]
    metric_rows = []
    for column in metric_columns:
        scores = score_prediction(oof, column)
        metric_rows.extend(
            {"model": column.removeprefix("prediction_"), "fold": key, "rmse": value}
            for key, value in scores.items()
        )
    metrics = pd.DataFrame(metric_rows)

    local_test = train_final_local(train, test, features, categorical, iterations)
    global_submission = pd.read_csv(GLOBAL_SUBMISSION_PATH)
    station_anchor_submission = pd.read_csv(STATION_ANCHOR_SUBMISSION_PATH)
    global_test = test[["id"]].merge(
        global_submission, on="id", validate="one_to_one"
    )["tma_mdpl"].to_numpy(dtype=float)
    station_anchor_test = test[["id"]].merge(
        station_anchor_submission, on="id", validate="one_to_one"
    )["tma_mdpl"].to_numpy(dtype=float)
    hybrid_test = (
        selected["catboost_weight"] * station_anchor_test
        + selected["global_lightgbm_weight"] * global_test
        + selected["local_lightgbm_weight"] * local_test
    )
    write_submission(OUTPUT_DIR / "submission_local_lightgbm.csv", test["id"], local_test)
    write_submission(
        OUTPUT_DIR / "submission_global_local_hybrid.csv", test["id"], hybrid_test
    )

    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    iterations.to_csv(OUTPUT_DIR / "local_model_iterations.csv", index=False)
    grid.to_csv(OUTPUT_DIR / "blend_grid.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "leave_one_fold_out_audit.csv", index=False)
    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    summary = {
        "selected_blend": selected,
        "leave_one_fold_out": audit.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "submission": str(OUTPUT_DIR / "submission_global_local_hybrid.csv"),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)
    print(audit.to_string(index=False), flush=True)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
