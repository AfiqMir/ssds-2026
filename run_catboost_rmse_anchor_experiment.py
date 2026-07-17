"""Evaluate RMSE-aligned tree selection and non-recursive state anchoring.

This experiment reuses the feature-engineering functions in
``notebook-catboost v2.ipynb`` so that the only model changes are measurable:

1. Compare tree-count selection by normalized MAE (the original notebook),
   normalized RMSE, and competition RMSE on the original TMA scale.
2. Tune a decaying correction based on the last observed station-level anomaly
   without feeding predictions back recursively.

All hyperparameter selection uses rolling-origin out-of-fold predictions.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error


SOURCE_NOTEBOOK = Path("notebook-catboost v2.ipynb")
OUTPUT_DIR = Path("output_catboost_experiments")
TRAIN_CACHE = OUTPUT_DIR / "train_features.parquet"
TEST_CACHE = OUTPUT_DIR / "test_features.parquet"
FEATURE_METADATA = OUTPUT_DIR / "feature_metadata.json"

MAX_ITERATIONS = 1_800
STAGED_EVAL_PERIOD = 20
FINAL_SEEDS = (17, 41, 83)

PIPELINE_DEFINITION_CELLS = (2, 4, 6, 8, 10, 12)


def original_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_pipeline_namespace() -> dict:
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    module = types.ModuleType("catboost_v2_notebook_runtime")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    for cell_index in PIPELINE_DEFINITION_CELLS:
        exec(notebook.cells[cell_index].source, namespace)
    return namespace


def build_or_load_features(namespace: dict):
    if TRAIN_CACHE.exists() and TEST_CACHE.exists() and FEATURE_METADATA.exists():
        train_features = pd.read_parquet(TRAIN_CACHE)
        test_features = pd.read_parquet(TEST_CACHE)
        metadata = json.loads(FEATURE_METADATA.read_text(encoding="utf-8"))
        columns = metadata["features"]
        print(
            f"Loaded cached features: train={train_features.shape}, "
            f"test={test_features.shape}, model_features={len(columns)}",
            flush=True,
        )
        return train_features, test_features, columns

    paths = namespace["Paths"](root=Path("."), artifacts=OUTPUT_DIR)
    config = namespace["ModelConfig"]()
    train, test, sample, coordinates, environment = namespace["load_inputs"](paths)
    namespace["audit_inputs"](train, test, sample, coordinates, environment)
    train_features, test_features, _ = namespace["build_feature_tables"](
        train, test, coordinates, environment, paths, config
    )
    columns = namespace["feature_columns"](train_features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_features.to_parquet(TRAIN_CACHE, index=False)
    test_features.to_parquet(TEST_CACHE, index=False)
    FEATURE_METADATA.write_text(
        json.dumps(
            {
                "features": columns,
                "categorical": [
                    column
                    for column in namespace["CATEGORICAL_COLUMNS"]
                    if column in columns
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Built features: train={train_features.shape}, "
        f"test={test_features.shape}, model_features={len(columns)}",
        flush=True,
    )
    return train_features, test_features, columns


def make_full_path_model(config, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=MAX_ITERATIONS,
        learning_rate=config.learning_rate,
        depth=config.depth,
        loss_function=f"Huber:delta={config.huber_delta}",
        eval_metric="RMSE",
        custom_metric=["MAE"],
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=seed,
        random_strength=0.8,
        bootstrap_type="Bayesian",
        bagging_temperature=0.7,
        allow_writing_files=False,
        thread_count=-1,
        verbose=100,
    )


def make_final_model(config, seed: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        loss_function=f"Huber:delta={config.huber_delta}",
        eval_metric="RMSE",
        l2_leaf_reg=config.l2_leaf_reg,
        random_seed=seed,
        random_strength=0.8,
        bootstrap_type="Bayesian",
        bagging_temperature=0.7,
        allow_writing_files=False,
        thread_count=-1,
        verbose=100,
    )


def predict_original_scale(model, pool, stations, stats, namespace, iteration):
    prediction_z = model.predict(pool, ntree_end=iteration)
    return namespace["denormalize_prediction"](
        prediction_z, stations, stats
    )


def safe_seasonal_baseline(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    reference = train.assign(
        month=train["datetime"].dt.month,
        hour=train["datetime"].dt.hour,
    )
    lookup = reference.groupby(
        ["nama_pos", "month", "hour"], observed=True
    )["tma_mdpl"].median()
    fallback = reference.groupby("nama_pos", observed=True)["tma_mdpl"].median()
    key = pd.MultiIndex.from_arrays(
        [
            target["nama_pos"],
            target["datetime"].dt.month,
            target["datetime"].dt.hour,
        ]
    )
    prediction = lookup.reindex(key).to_numpy(dtype=float).copy()
    missing = np.isnan(prediction)
    prediction[missing] = (
        target.loc[missing, "nama_pos"].map(fallback).to_numpy(dtype=float)
    )
    return prediction


def best_selection_iterations(
    model,
    validation_pool,
    truth,
    truth_normalized,
    stations,
    stats,
    namespace,
) -> dict[str, int]:
    best_iterations = {
        "mae_selected": 1,
        "normalized_rmse_selected": 1,
        "raw_rmse_selected": 1,
    }
    best_scores = {key: float("inf") for key in best_iterations}

    for step, prediction_z in enumerate(
        model.staged_predict(
            validation_pool,
            eval_period=STAGED_EVAL_PERIOD,
        ),
        start=1,
    ):
        iteration = min(step * STAGED_EVAL_PERIOD, model.tree_count_)
        prediction = namespace["denormalize_prediction"](
            prediction_z, stations, stats
        )
        scores = {
            "mae_selected": float(
                mean_absolute_error(truth_normalized, prediction_z)
            ),
            "normalized_rmse_selected": original_rmse(
                truth_normalized, prediction_z
            ),
            "raw_rmse_selected": original_rmse(truth, prediction),
        }
        for method, score in scores.items():
            if score < best_scores[method]:
                best_scores[method] = score
                best_iterations[method] = iteration

    refinement_iterations: set[int] = set()
    for best_iteration in best_iterations.values():
        lower = max(1, best_iteration - STAGED_EVAL_PERIOD + 1)
        upper = min(model.tree_count_, best_iteration + STAGED_EVAL_PERIOD - 1)
        refinement_iterations.update(range(lower, upper + 1))

    for iteration in sorted(refinement_iterations):
        prediction_z = model.predict(validation_pool, ntree_end=iteration)
        prediction = namespace["denormalize_prediction"](
            prediction_z, stations, stats
        )
        scores = {
            "mae_selected": float(
                mean_absolute_error(truth_normalized, prediction_z)
            ),
            "normalized_rmse_selected": original_rmse(
                truth_normalized, prediction_z
            ),
            "raw_rmse_selected": original_rmse(truth, prediction),
        }
        for method, score in scores.items():
            if score < best_scores[method]:
                best_scores[method] = score
                best_iterations[method] = iteration

    return best_iterations


def state_features_for_fold(fit, validation, stats, namespace):
    last_rows = (
        fit.sort_values("datetime")
        .groupby("nama_pos", as_index=False, observed=True)
        .tail(1)
        .copy()
    )
    baseline = safe_seasonal_baseline(fit, last_rows)
    last_rows["state_anomaly"] = last_rows["tma_mdpl"].to_numpy() - baseline
    last_rows["state_scale"] = last_rows["nama_pos"].map(stats["scale"])
    last_rows["state_anomaly"] = last_rows["state_anomaly"].clip(
        lower=-3.0 * last_rows["state_scale"],
        upper=3.0 * last_rows["state_scale"],
    )
    lookup_anomaly = last_rows.set_index("nama_pos")["state_anomaly"]
    lookup_time = last_rows.set_index("nama_pos")["datetime"]
    anomaly = validation["nama_pos"].map(lookup_anomaly).to_numpy(dtype=float)
    last_time = validation["nama_pos"].map(lookup_time)
    horizon_days = (
        (validation["datetime"] - last_time).dt.total_seconds() / 86_400.0
    ).to_numpy(dtype=float)
    return anomaly, horizon_days


def validate_full_model_paths(
    train_features: pd.DataFrame,
    columns: list[str],
    namespace: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = namespace["ModelConfig"]()
    folds = namespace["FOLDS"]
    categorical = [
        column
        for column in namespace["CATEGORICAL_COLUMNS"]
        if column in columns
    ]
    metrics_rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []
    model_dir = OUTPUT_DIR / "validation_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold_name, train_end, valid_start, valid_end in folds:
        fit = train_features[
            train_features["datetime"] <= pd.Timestamp(train_end)
        ].copy()
        validation = train_features[
            train_features["datetime"].between(
                pd.Timestamp(valid_start),
                pd.Timestamp(valid_end),
                inclusive="both",
            )
        ].copy()
        eligible = (
            fit.groupby("nama_pos", observed=True)
            .size()
            .loc[lambda count: count >= 90]
            .index
        )
        fit = fit[fit["nama_pos"].isin(eligible)]
        validation = validation[validation["nama_pos"].isin(eligible)]

        stats = namespace["fit_target_stats"](fit)
        y_fit = namespace["normalize_target"](fit, stats)
        y_validation_z = namespace["normalize_target"](validation, stats)
        sample_weights, spike_count = namespace["isolated_spike_weights"](fit)

        fit_pool = Pool(
            namespace["_xy"](fit, columns),
            y_fit,
            cat_features=categorical,
            weight=sample_weights,
        )
        validation_pool = Pool(
            namespace["_xy"](validation, columns),
            y_validation_z,
            cat_features=categorical,
        )

        print(
            f"\n[{fold_name}] training full {MAX_ITERATIONS}-tree path: "
            f"fit={len(fit):,}, validation={len(validation):,}",
            flush=True,
        )
        model_path = model_dir / f"{fold_name}.cbm"
        model = make_full_path_model(config, seed=config.seeds[0])
        if model_path.exists():
            model.load_model(str(model_path))
            print(f"[{fold_name}] reusing saved full model path", flush=True)
        else:
            model.fit(fit_pool, eval_set=validation_pool, use_best_model=False)
            model.save_model(str(model_path))

        truth = validation["tma_mdpl"].to_numpy(dtype=float)
        method_iterations = best_selection_iterations(
            model,
            validation_pool,
            truth,
            y_validation_z,
            validation["nama_pos"],
            stats,
            namespace,
        )
        best_mae_iteration = method_iterations["mae_selected"]
        best_normalized_rmse_iteration = method_iterations[
            "normalized_rmse_selected"
        ]
        best_raw_iteration = method_iterations["raw_rmse_selected"]
        predictions = {}
        row = {
            "fold": fold_name,
            "train_rows": len(fit),
            "validation_rows": len(validation),
            "stations": validation["nama_pos"].nunique(),
            "spikes_downweighted": spike_count,
        }
        for method, iteration in method_iterations.items():
            prediction = predict_original_scale(
                model,
                validation_pool,
                validation["nama_pos"],
                stats,
                namespace,
                iteration,
            )
            predictions[method] = prediction
            row[f"{method}_iteration"] = iteration
            row[f"{method}_rmse"] = original_rmse(truth, prediction)
            row[f"{method}_mae"] = float(
                mean_absolute_error(truth, prediction)
            )

        anomaly, horizon_days = state_features_for_fold(
            fit, validation, stats, namespace
        )
        oof = validation[["datetime", "nama_pos", "tma_mdpl"]].copy()
        oof.insert(0, "fold", fold_name)
        for method, prediction in predictions.items():
            oof[f"prediction_{method}"] = prediction
        oof["state_anomaly"] = anomaly
        oof["horizon_days"] = horizon_days
        oof_parts.append(oof)
        metrics_rows.append(row)

        print(
            f"[{fold_name}] MAE-selected RMSE={row['mae_selected_rmse']:.6f} "
            f"(iter={best_mae_iteration}); normalized-RMSE-selected="
            f"{row['normalized_rmse_selected_rmse']:.6f} "
            f"(iter={best_normalized_rmse_iteration}); raw-RMSE-selected="
            f"{row['raw_rmse_selected_rmse']:.6f} "
            f"(iter={best_raw_iteration})",
            flush=True,
        )

        pd.DataFrame(metrics_rows).to_csv(
            OUTPUT_DIR / "tree_selection_metrics.csv", index=False
        )
        pd.concat(oof_parts, ignore_index=True).to_parquet(
            OUTPUT_DIR / "oof_predictions.parquet", index=False
        )

    return pd.DataFrame(metrics_rows), pd.concat(oof_parts, ignore_index=True)


def tune_state_anchor(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    prediction_column = "prediction_raw_rmse_selected"
    baseline_by_fold = {
        fold: original_rmse(group["tma_mdpl"], group[prediction_column])
        for fold, group in oof.groupby("fold", sort=False)
    }
    rows: list[dict] = []
    for tau_days in (7, 15, 30, 60, 90, 120, 180, 365):
        decay = np.exp(-oof["horizon_days"].to_numpy() / tau_days)
        correction_basis = decay * oof["state_anomaly"].to_numpy()
        for alpha in np.linspace(0.0, 1.0, 21):
            candidate = oof[prediction_column].to_numpy() + alpha * correction_basis
            candidate_frame = oof[["fold", "tma_mdpl"]].copy()
            candidate_frame["prediction"] = candidate
            fold_scores = {
                fold: original_rmse(group["tma_mdpl"], group["prediction"])
                for fold, group in candidate_frame.groupby("fold", sort=False)
            }
            deltas = {
                fold: fold_scores[fold] - baseline_by_fold[fold]
                for fold in fold_scores
            }
            rows.append(
                {
                    "tau_days": tau_days,
                    "alpha": float(alpha),
                    "mean_fold_rmse": float(np.mean(list(fold_scores.values()))),
                    "mean_fold_delta": float(np.mean(list(deltas.values()))),
                    "worst_fold_delta": float(np.max(list(deltas.values()))),
                    **{f"rmse_{fold}": value for fold, value in fold_scores.items()},
                    **{f"delta_{fold}": value for fold, value in deltas.items()},
                }
            )

    grid = pd.DataFrame(rows)
    aligned_folds = [fold for fold in ("sep_2023", "sep_2024") if fold in baseline_by_fold]
    robust_mask = grid["mean_fold_delta"] < 0
    for fold in aligned_folds:
        robust_mask &= grid[f"delta_{fold}"] <= 0
    robust_mask &= grid["worst_fold_delta"] <= 0.0

    robust = grid[robust_mask].sort_values(
        ["mean_fold_rmse", "worst_fold_delta"]
    )
    if robust.empty:
        selected_row = grid[
            (grid["alpha"] == 0.0) & (grid["tau_days"] == 7)
        ].iloc[0]
        selection_note = "No robust correction; selected alpha=0."
    else:
        selected_row = robust.iloc[0]
        selection_note = (
            "Improves both September-to-May folds, improves mean fold RMSE, "
            "and does not degrade any stress fold."
        )

    selected = {
        "tau_days": int(selected_row["tau_days"]),
        "alpha": float(selected_row["alpha"]),
        "mean_fold_rmse": float(selected_row["mean_fold_rmse"]),
        "mean_fold_delta": float(selected_row["mean_fold_delta"]),
        "worst_fold_delta": float(selected_row["worst_fold_delta"]),
        "selection_note": selection_note,
    }
    for fold in baseline_by_fold:
        selected[f"baseline_rmse_{fold}"] = baseline_by_fold[fold]
        selected[f"corrected_rmse_{fold}"] = float(selected_row[f"rmse_{fold}"])

    grid.to_csv(OUTPUT_DIR / "anchor_grid.csv", index=False)
    (OUTPUT_DIR / "anchor_selection.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    return grid, selected


def final_state_correction(train_features, test_features, stats, namespace, selection):
    last_rows = (
        train_features.sort_values("datetime")
        .groupby("nama_pos", as_index=False, observed=True)
        .tail(1)
        .copy()
    )
    baseline = safe_seasonal_baseline(train_features, last_rows)
    last_rows["state_anomaly"] = last_rows["tma_mdpl"].to_numpy() - baseline
    last_rows["state_scale"] = last_rows["nama_pos"].map(stats["scale"])
    last_rows["state_anomaly"] = last_rows["state_anomaly"].clip(
        lower=-3.0 * last_rows["state_scale"],
        upper=3.0 * last_rows["state_scale"],
    )
    anomaly_lookup = last_rows.set_index("nama_pos")["state_anomaly"]
    time_lookup = last_rows.set_index("nama_pos")["datetime"]
    anomaly = test_features["nama_pos"].map(anomaly_lookup).to_numpy(dtype=float)
    last_time = test_features["nama_pos"].map(time_lookup)
    horizon_days = (
        (test_features["datetime"] - last_time).dt.total_seconds() / 86_400.0
    ).to_numpy(dtype=float)
    return (
        selection["alpha"]
        * np.exp(-horizon_days / selection["tau_days"])
        * anomaly
    )


def train_final_and_submit(
    train_features,
    test_features,
    columns,
    metrics,
    anchor_selection,
    namespace,
):
    config = namespace["ModelConfig"]()
    categorical = [
        column
        for column in namespace["CATEGORICAL_COLUMNS"]
        if column in columns
    ]
    iterations = int(np.median(metrics["raw_rmse_selected_iteration"]))
    stats = namespace["fit_target_stats"](train_features)
    y = namespace["normalize_target"](train_features, stats)
    sample_weights, _ = namespace["isolated_spike_weights"](train_features)
    train_pool = Pool(
        namespace["_xy"](train_features, columns),
        y,
        cat_features=categorical,
        weight=sample_weights,
    )
    test_pool = Pool(
        namespace["_xy"](test_features, columns),
        cat_features=categorical,
    )
    model_dir = OUTPUT_DIR / "final_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_z = []
    for seed in FINAL_SEEDS:
        print(
            f"Training final CatBoost seed={seed}, iterations={iterations}",
            flush=True,
        )
        model = make_final_model(config, seed, iterations)
        model.fit(train_pool)
        model.save_model(str(model_dir / f"catboost_seed_{seed}.cbm"))
        predictions_z.append(model.predict(test_pool))

    prediction_z = np.mean(predictions_z, axis=0)
    prediction = namespace["denormalize_prediction"](
        prediction_z, test_features["nama_pos"], stats
    )
    correction = final_state_correction(
        train_features,
        test_features,
        stats,
        namespace,
        anchor_selection,
    )
    corrected_prediction = prediction + correction

    sample = pd.read_csv("sample_submission.csv")
    base = pd.DataFrame(
        {"id": test_features["id"], "tma_mdpl": prediction}
    ).sort_index()
    corrected = pd.DataFrame(
        {"id": test_features["id"], "tma_mdpl": corrected_prediction}
    ).sort_index()
    if not base["id"].reset_index(drop=True).equals(sample["id"]):
        raise ValueError("Final test order does not match sample submission")
    if not np.isfinite(base["tma_mdpl"]).all() or not np.isfinite(
        corrected["tma_mdpl"]
    ).all():
        raise ValueError("Non-finite final prediction")

    base.to_csv(OUTPUT_DIR / "submission_rmse_selected.csv", index=False)
    corrected.to_csv(OUTPUT_DIR / "submission_rmse_anchor.csv", index=False)
    metadata = {
        "final_iterations": iterations,
        "seeds": list(FINAL_SEEDS),
        "anchor": anchor_selection,
    }
    (OUTPUT_DIR / "final_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def retune_anchor_from_cached_outputs() -> None:
    namespace = load_pipeline_namespace()
    train_features, test_features, _ = build_or_load_features(namespace)
    oof = pd.read_parquet(OUTPUT_DIR / "oof_predictions.parquet")
    _, anchor_selection = tune_state_anchor(oof)
    base_path = OUTPUT_DIR / "submission_rmse_selected.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing cached base submission: {base_path}")
    base = pd.read_csv(base_path)
    stats = namespace["fit_target_stats"](train_features)
    correction = final_state_correction(
        train_features,
        test_features,
        stats,
        namespace,
        anchor_selection,
    )
    corrected = base.copy()
    corrected["tma_mdpl"] = corrected["tma_mdpl"].to_numpy() + correction
    corrected.to_csv(OUTPUT_DIR / "submission_rmse_anchor.csv", index=False)
    metadata_path = OUTPUT_DIR / "final_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    metadata["anchor"] = anchor_selection
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(anchor_selection, indent=2), flush=True)
    print("Re-exported submission_rmse_anchor.csv", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    namespace = load_pipeline_namespace()
    train_features, test_features, columns = build_or_load_features(namespace)
    metrics, oof = validate_full_model_paths(
        train_features, columns, namespace
    )
    print("\nTree-selection metrics:", flush=True)
    print(metrics.to_string(index=False), flush=True)

    _, anchor_selection = tune_state_anchor(oof)
    print("\nSelected anchor:", flush=True)
    print(json.dumps(anchor_selection, indent=2), flush=True)

    train_final_and_submit(
        train_features,
        test_features,
        columns,
        metrics,
        anchor_selection,
        namespace,
    )
    print("\nExperiment complete.", flush=True)


if __name__ == "__main__":
    if "--retune-anchor-only" in sys.argv:
        retune_anchor_from_cached_outputs()
    else:
        main()
