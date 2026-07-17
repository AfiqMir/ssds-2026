"""Train and validate a reproducible CatBoost-XGBoost blend.

The script reuses the non-recursive feature matrices produced by
``tma_pipeline_v2_notebook.ipynb``. Blend weights are learned exclusively from
out-of-fold predictions on September-to-May windows, matching the competition
test season. No test labels or manually selected submission weights are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


FEATURE_DIR = Path("output_v2")
OUTPUT_DIR = Path("output_blend")

TRAIN_FEATURES_PATH = FEATURE_DIR / "train_features.parquet"
TEST_FEATURES_PATH = FEATURE_DIR / "test_features.parquet"
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

# These two validation windows have the same September-to-May season and
# approximately the same 8-month horizon as the competition test set.
FOLDS = [
    ("sep_2023", "2023-09-19", "2024-05-19"),
    ("sep_2024", "2024-09-19", "2025-05-19"),
]

SEEDS = [17, 41, 83]
MIN_LABELS_BEFORE_CUTOFF = 90

CATBOOST_MAX_ITERATIONS = 1_500
XGBOOST_MAX_ITERATIONS = 3_000
EARLY_STOPPING_ROUNDS = 100

NON_FEATURE_COLUMNS = {
    "nama_pos",
    "datetime",
    "split",
    "id",
    "tma_mdpl",
    "MAIN_RIV",
}
CAT_FEATURES = ["nama_pos_cat"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_target_stats(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("nama_pos")["tma_mdpl"]
    center = grouped.median()
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    scale = ((q3 - q1) / 1.349).clip(lower=0.05)
    return pd.DataFrame(
        {"center": center, "scale": scale}, index=center.index
    )


def normalize_target(frame: pd.DataFrame, stats: pd.DataFrame) -> np.ndarray:
    center = frame["nama_pos"].map(stats["center"]).to_numpy()
    scale = frame["nama_pos"].map(stats["scale"]).to_numpy()
    return (frame["tma_mdpl"].to_numpy() - center) / scale


def denormalize_target(
    pred_normalized: np.ndarray,
    station: pd.Series,
    stats: pd.DataFrame,
) -> np.ndarray:
    center = station.map(stats["center"]).to_numpy()
    scale = station.map(stats["scale"]).to_numpy()
    return pred_normalized * scale + center


def prepare_xgboost_categories(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_xgb = train_features.copy()
    test_xgb = test_features.copy()
    categories = sorted(
        set(train_xgb["nama_pos_cat"].dropna().astype(str))
        | set(test_xgb["nama_pos_cat"].dropna().astype(str))
    )
    dtype = pd.CategoricalDtype(categories=categories)
    train_xgb["nama_pos_cat"] = train_xgb["nama_pos_cat"].astype(dtype)
    test_xgb["nama_pos_cat"] = test_xgb["nama_pos_cat"].astype(dtype)
    return train_xgb, test_xgb


def make_catboost(
    seed: int,
    iterations: int,
    use_early_stopping: bool,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="Huber:delta=1.5",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        random_seed=seed,
        cat_features=CAT_FEATURES,
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS if use_early_stopping else None
        ),
        verbose=False,
        allow_writing_files=False,
    )


def make_xgboost(
    seed: int,
    iterations: int,
    use_early_stopping: bool,
) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=32,
        n_estimators=iterations,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10.0,
        early_stopping_rounds=(
            EARLY_STOPPING_ROUNDS if use_early_stopping else None
        ),
        random_state=seed,
        n_jobs=-1,
    )


def optimal_catboost_weight(
    truth: np.ndarray,
    pred_catboost: np.ndarray,
    pred_xgboost: np.ndarray,
) -> float:
    """Return the SSE-optimal CatBoost weight constrained to [0, 1]."""
    delta = pred_catboost - pred_xgboost
    denominator = float(np.dot(delta, delta))
    if denominator <= np.finfo(float).eps:
        return 0.5
    weight = float(np.dot(truth - pred_xgboost, delta) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def check_inputs() -> None:
    missing = [
        str(path)
        for path in (
            TRAIN_FEATURES_PATH,
            TEST_FEATURES_PATH,
            SAMPLE_SUBMISSION_PATH,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required inputs are missing: " + ", ".join(missing)
        )


def export_cached_recent_fold_candidate() -> None:
    """Export a true blend using the latest fold's optimal OOF weight.

    This candidate is intentionally separate from the conservative pooled-OOF
    submission. It is useful because the latest fold has a training-set size
    closer to the final model, but its weight is supported by only one fold.
    """
    required = [
        OUTPUT_DIR / "blend_summary.json",
        OUTPUT_DIR / "fold_metrics.csv",
        OUTPUT_DIR / "oof_predictions.parquet",
        OUTPUT_DIR / "test_component_predictions.parquet",
        SAMPLE_SUBMISSION_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cached blend artifacts are missing: " + ", ".join(missing)
        )

    metrics = pd.read_csv(OUTPUT_DIR / "fold_metrics.csv")
    latest = metrics.iloc[-1]
    latest_fold = str(latest["fold"])
    weight_cat = float(latest["fold_optimal_catboost_weight"])
    weight_xgb = 1.0 - weight_cat

    oof = pd.read_parquet(OUTPUT_DIR / "oof_predictions.parquet")
    latest_oof = oof[oof["fold"] == latest_fold]
    truth = latest_oof["tma_mdpl"].to_numpy()
    pred_recent = (
        weight_cat * latest_oof["pred_catboost"].to_numpy()
        + weight_xgb * latest_oof["pred_xgboost"].to_numpy()
    )

    components = pd.read_parquet(
        OUTPUT_DIR / "test_component_predictions.parquet"
    )
    pred_test = (
        weight_cat * components["pred_catboost"].to_numpy()
        + weight_xgb * components["pred_xgboost"].to_numpy()
    )
    submission = components[["id"]].copy()
    submission["tma_mdpl"] = pred_test
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission IDs do not match sample_submission.csv")
    if not np.isfinite(submission["tma_mdpl"]).all():
        raise ValueError("Submission contains NaN or infinite predictions")

    output_path = OUTPUT_DIR / "submission_blend_recent_fold.csv"
    submission.to_csv(output_path, index=False)

    summary_path = OUTPUT_DIR / "blend_summary.json"
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    summary.update(
        {
            "recent_candidate_fold": latest_fold,
            "recent_candidate_catboost_weight": weight_cat,
            "recent_candidate_xgboost_weight": weight_xgb,
            "recent_candidate_catboost_rmse": float(
                latest["catboost_rmse"]
            ),
            "recent_candidate_blend_rmse": rmse(truth, pred_recent),
            "recent_candidate_selection_note": (
                "Experimental: weight selected from the latest matching-season "
                "fold only; pooled OOF remains the conservative selection."
            ),
        }
    )
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(
        f"Saved {output_path}: CatBoost={weight_cat:.6f}, "
        f"XGBoost={weight_xgb:.6f}, latest-fold RMSE="
        f"{rmse(truth, pred_recent):.6f}"
    )


def main() -> None:
    check_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    test = pd.read_parquet(TEST_FEATURES_PATH)
    train["datetime"] = pd.to_datetime(train["datetime"])
    test["datetime"] = pd.to_datetime(test["datetime"])

    feature_columns = [
        column for column in train.columns if column not in NON_FEATURE_COLUMNS
    ]
    if CAT_FEATURES[0] not in feature_columns:
        raise ValueError(f"Missing categorical feature: {CAT_FEATURES[0]}")
    if train.duplicated(["nama_pos", "datetime"]).any():
        raise ValueError("Duplicate station-datetime rows in training features")
    if test.duplicated(["nama_pos", "datetime"]).any():
        raise ValueError("Duplicate station-datetime rows in test features")

    train_xgb, test_xgb = prepare_xgboost_categories(train, test)

    print(
        f"Loaded train={train.shape}, test={test.shape}, "
        f"features={len(feature_columns)}"
    )

    oof_parts: list[pd.DataFrame] = []
    fold_iteration_rows: list[dict[str, int | str]] = []

    for fold_name, cutoff, validation_end in FOLDS:
        cutoff_ts = pd.Timestamp(cutoff)
        validation_end_ts = pd.Timestamp(validation_end)

        train_mask = train["datetime"] < cutoff_ts
        counts = train.loc[train_mask].groupby("nama_pos").size()
        valid_stations = counts[counts >= MIN_LABELS_BEFORE_CUTOFF].index
        validation_mask = (
            (train["datetime"] >= cutoff_ts)
            & (train["datetime"] < validation_end_ts)
            & train["nama_pos"].isin(valid_stations)
        )

        train_fold = train.loc[train_mask]
        validation_fold = train.loc[validation_mask]
        train_fold_xgb = train_xgb.loc[train_mask]
        validation_fold_xgb = train_xgb.loc[validation_mask]

        if train_fold.empty or validation_fold.empty:
            raise ValueError(f"Fold {fold_name} has no train or validation rows")

        stats = fit_target_stats(train_fold)
        y_train = normalize_target(train_fold, stats)
        y_validation = normalize_target(validation_fold, stats)
        truth = validation_fold["tma_mdpl"].to_numpy()

        print(
            f"[{fold_name}] train={len(train_fold):,}, "
            f"validation={len(validation_fold):,}, "
            f"stations={len(valid_stations)}"
        )

        cat_model = make_catboost(
            seed=SEEDS[0],
            iterations=CATBOOST_MAX_ITERATIONS,
            use_early_stopping=True,
        )
        cat_model.fit(
            train_fold[feature_columns],
            y_train,
            eval_set=(validation_fold[feature_columns], y_validation),
            use_best_model=True,
        )
        pred_cat = denormalize_target(
            cat_model.predict(validation_fold[feature_columns]),
            validation_fold["nama_pos"],
            stats,
        )

        xgb_model = make_xgboost(
            seed=SEEDS[0],
            iterations=XGBOOST_MAX_ITERATIONS,
            use_early_stopping=True,
        )
        xgb_model.fit(
            train_fold_xgb[feature_columns],
            y_train,
            eval_set=[
                (validation_fold_xgb[feature_columns], y_validation)
            ],
            verbose=False,
        )
        pred_xgb = denormalize_target(
            xgb_model.predict(validation_fold_xgb[feature_columns]),
            validation_fold["nama_pos"],
            stats,
        )

        cat_iteration = int(cat_model.get_best_iteration()) + 1
        xgb_iteration = int(xgb_model.best_iteration) + 1
        fold_iteration_rows.append(
            {
                "fold": fold_name,
                "catboost_best_iteration": cat_iteration,
                "xgboost_best_iteration": xgb_iteration,
            }
        )

        fold_oof = validation_fold[
            ["datetime", "nama_pos", "tma_mdpl"]
        ].copy()
        fold_oof.insert(0, "fold", fold_name)
        fold_oof["pred_catboost"] = pred_cat
        fold_oof["pred_xgboost"] = pred_xgb
        oof_parts.append(fold_oof)

        print(
            f"[{fold_name}] CatBoost RMSE={rmse(truth, pred_cat):.6f} "
            f"(iter={cat_iteration}); "
            f"XGBoost RMSE={rmse(truth, pred_xgb):.6f} "
            f"(iter={xgb_iteration})"
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    truth_all = oof["tma_mdpl"].to_numpy()
    pred_cat_all = oof["pred_catboost"].to_numpy()
    pred_xgb_all = oof["pred_xgboost"].to_numpy()
    weight_cat = optimal_catboost_weight(
        truth_all, pred_cat_all, pred_xgb_all
    )
    weight_xgb = 1.0 - weight_cat
    oof["pred_blend"] = (
        weight_cat * oof["pred_catboost"]
        + weight_xgb * oof["pred_xgboost"]
    )

    fold_metrics: list[dict[str, float | int | str]] = []
    for fold_name, fold_frame in oof.groupby("fold", sort=False):
        truth = fold_frame["tma_mdpl"].to_numpy()
        pred_cat = fold_frame["pred_catboost"].to_numpy()
        pred_xgb = fold_frame["pred_xgboost"].to_numpy()
        pred_blend = fold_frame["pred_blend"].to_numpy()
        fold_metrics.append(
            {
                "fold": fold_name,
                "n_validation": len(fold_frame),
                "catboost_rmse": rmse(truth, pred_cat),
                "xgboost_rmse": rmse(truth, pred_xgb),
                "blend_rmse": rmse(truth, pred_blend),
                "catboost_mae": float(
                    mean_absolute_error(truth, pred_cat)
                ),
                "xgboost_mae": float(
                    mean_absolute_error(truth, pred_xgb)
                ),
                "blend_mae": float(
                    mean_absolute_error(truth, pred_blend)
                ),
                "fold_optimal_catboost_weight": optimal_catboost_weight(
                    truth, pred_cat, pred_xgb
                ),
            }
        )

    iteration_frame = pd.DataFrame(fold_iteration_rows)
    metrics_frame = pd.DataFrame(fold_metrics).merge(
        iteration_frame, on="fold", how="left"
    )

    summary = {
        "folds": [fold[0] for fold in FOLDS],
        "n_oof": int(len(oof)),
        "catboost_weight": weight_cat,
        "xgboost_weight": weight_xgb,
        "pooled_catboost_rmse": rmse(truth_all, pred_cat_all),
        "pooled_xgboost_rmse": rmse(truth_all, pred_xgb_all),
        "pooled_blend_rmse": rmse(
            truth_all, oof["pred_blend"].to_numpy()
        ),
        "catboost_final_iterations": int(
            np.median(iteration_frame["catboost_best_iteration"])
        ),
        "xgboost_final_iterations": int(
            np.median(iteration_frame["xgboost_best_iteration"])
        ),
        "seeds": SEEDS,
        "feature_count": len(feature_columns),
    }

    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    metrics_frame.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    with (OUTPUT_DIR / "blend_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nOOF metrics:")
    print(metrics_frame.to_string(index=False))
    print("\nBlend summary:")
    print(json.dumps(summary, indent=2))

    stats_final = fit_target_stats(train)
    y_all = normalize_target(train, stats_final)
    pred_cat_test: list[np.ndarray] = []
    pred_xgb_test: list[np.ndarray] = []

    print("\nTraining final three-seed ensembles ...")
    for seed in SEEDS:
        print(f"  seed={seed}: CatBoost")
        cat_model = make_catboost(
            seed=seed,
            iterations=summary["catboost_final_iterations"],
            use_early_stopping=False,
        )
        cat_model.fit(train[feature_columns], y_all)
        cat_model.save_model(
            str(OUTPUT_DIR / f"catboost_seed_{seed}.cbm")
        )
        pred_cat_test.append(
            cat_model.predict(test[feature_columns])
        )

        print(f"  seed={seed}: XGBoost")
        xgb_model = make_xgboost(
            seed=seed,
            iterations=summary["xgboost_final_iterations"],
            use_early_stopping=False,
        )
        xgb_model.fit(train_xgb[feature_columns], y_all, verbose=False)
        xgb_model.save_model(
            OUTPUT_DIR / f"xgboost_seed_{seed}.ubj"
        )
        pred_xgb_test.append(
            xgb_model.predict(test_xgb[feature_columns])
        )

    pred_cat_normalized = np.mean(pred_cat_test, axis=0)
    pred_xgb_normalized = np.mean(pred_xgb_test, axis=0)
    pred_cat = denormalize_target(
        pred_cat_normalized, test["nama_pos"], stats_final
    )
    pred_xgb = denormalize_target(
        pred_xgb_normalized, test["nama_pos"], stats_final
    )
    pred_blend = weight_cat * pred_cat + weight_xgb * pred_xgb

    submission = test[["id"]].copy()
    submission["tma_mdpl"] = pred_blend
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission IDs do not match sample_submission.csv")
    if not np.isfinite(submission["tma_mdpl"]).all():
        raise ValueError("Submission contains NaN or infinite predictions")
    if submission["id"].duplicated().any():
        raise ValueError("Submission contains duplicate IDs")

    component_predictions = test[["id", "datetime", "nama_pos"]].copy()
    component_predictions["pred_catboost"] = pred_cat
    component_predictions["pred_xgboost"] = pred_xgb
    component_predictions["pred_blend"] = pred_blend
    component_predictions.to_parquet(
        OUTPUT_DIR / "test_component_predictions.parquet", index=False
    )

    submission_path = OUTPUT_DIR / "submission_blend_oof.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved {submission_path} with shape={submission.shape}")
    print(submission["tma_mdpl"].describe().to_string())
    export_cached_recent_fold_candidate()


if __name__ == "__main__":
    if "--export-cached-recent" in sys.argv:
        export_cached_recent_fold_candidate()
    else:
        main()
