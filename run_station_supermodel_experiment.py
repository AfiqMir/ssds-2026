"""Conservative per-station mixture-of-experts experiment.

Each station defaults to the existing station-stack.  A challenger is eligible
only when it beats the control on every available training fold and improves
mean station-level fold RMSE by at least a fixed margin.  Eligible challengers
are shrunk 50% toward the control instead of being hard-routed.

The routing rule is evaluated with an outer leave-one-fold-out audit.  API_TEST
is quarantined and is never read or executed by this script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from run_lightgbm_residual_experiment import rmse
from run_station_aware_experiment import write_submission


OUTPUT_DIR = Path("output_station_supermodel")
STATION_DIR = Path("output_station_aware")
SAMPLE_SUBMISSION_PATH = Path("sample_submission.csv")

CONTROL_NAME = "station_stack"
CONTROL_OOF_PATH = STATION_DIR / "oof_predictions.parquet"
CONTROL_OOF_COLUMN = "prediction_station_stack"
CONTROL_SUBMISSION_PATH = STATION_DIR / "submission_station_stack.csv"

# These values were declared before implementing this experiment, based on the
# conservative exploratory rule: unanimous fold wins, material improvement,
# and partial rather than hard routing.
MIN_MEAN_RMSE_IMPROVEMENT = 0.01
EXPERT_WEIGHT = 0.50
MAX_WORST_FOLD_DELTA = 0.001

KEY_COLUMNS = ["fold", "datetime", "nama_pos"]


@dataclass(frozen=True)
class Expert:
    name: str
    oof_path: Path
    oof_column: str
    submission_path: Path


EXPERTS = (
    Expert(
        "station_anchor",
        STATION_DIR / "oof_predictions.parquet",
        "prediction_station_anchor",
        STATION_DIR / "submission_station_anchor.csv",
    ),
    Expert(
        "station_weighted_stack",
        STATION_DIR / "oof_predictions.parquet",
        "prediction_station_weighted_stack",
        STATION_DIR / "submission_station_weighted_stack.csv",
    ),
    Expert(
        "time_forward_bias",
        Path("output_time_forward_bias/oof_predictions.parquet"),
        "prediction_time_forward_bias",
        Path("output_time_forward_bias/submission_time_forward_bias.csv"),
    ),
    Expert(
        "global_local_hybrid",
        Path("output_global_local_hybrid/oof_predictions.parquet"),
        "prediction_global_local_hybrid",
        Path("output_global_local_hybrid/submission_global_local_hybrid.csv"),
    ),
    Expert(
        "recency_blend",
        Path("output_catboost_recency/oof_predictions.parquet"),
        "prediction_selected",
        Path("output_catboost_recency/submission_recency_blend.csv"),
    ),
    Expert(
        "state_space_blend",
        Path("output_state_space/oof_predictions.parquet"),
        "prediction_state_space_blend",
        Path("output_state_space/submission_state_space_blend.csv"),
    ),
)


def check_inputs() -> None:
    paths = [
        SAMPLE_SUBMISSION_PATH,
        CONTROL_OOF_PATH,
        CONTROL_SUBMISSION_PATH,
        *(expert.oof_path for expert in EXPERTS),
        *(expert.submission_path for expert in EXPERTS),
    ]
    missing = sorted({str(path) for path in paths if not path.exists()})
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))


def load_oof() -> pd.DataFrame:
    frame = pd.read_parquet(CONTROL_OOF_PATH)[
        KEY_COLUMNS + ["tma_mdpl", CONTROL_OOF_COLUMN]
    ].rename(columns={CONTROL_OOF_COLUMN: f"prediction_{CONTROL_NAME}"})
    for expert in EXPERTS:
        column = f"prediction_{expert.name}"
        if expert.oof_path == CONTROL_OOF_PATH:
            values = pd.read_parquet(expert.oof_path)[
                KEY_COLUMNS + [expert.oof_column]
            ].rename(columns={expert.oof_column: column})
        else:
            values = pd.read_parquet(expert.oof_path)[
                KEY_COLUMNS + [expert.oof_column]
            ].rename(columns={expert.oof_column: column})
        frame = frame.merge(values, on=KEY_COLUMNS, validate="one_to_one")
    prediction_columns = [column for column in frame if column.startswith("prediction_")]
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate OOF keys")
    if frame[prediction_columns].isna().any().any():
        raise ValueError("Missing OOF predictions")
    if not np.isfinite(frame[prediction_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite OOF predictions")
    return frame


def station_fold_rmse(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    return frame.groupby(["nama_pos", "fold"], observed=True).apply(
        lambda group: rmse(group["tma_mdpl"], group[prediction_column]),
        include_groups=False,
    )


def select_routes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a frozen route per station from the folds present in frame."""
    control_column = f"prediction_{CONTROL_NAME}"
    control_scores = station_fold_rmse(frame, control_column)
    route_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for station in sorted(frame["nama_pos"].unique()):
        control = control_scores.loc[station].sort_index()
        candidates: list[dict[str, object]] = []
        for expert in EXPERTS:
            column = f"prediction_{expert.name}"
            scores = station_fold_rmse(
                frame[frame["nama_pos"] == station], column
            ).loc[station].sort_index()
            if not scores.index.equals(control.index):
                raise ValueError(f"Fold mismatch for {station}/{expert.name}")
            improvements = control - scores
            unanimous_win = bool((improvements > 0.0).all())
            mean_improvement = float(improvements.mean())
            record = {
                "nama_pos": station,
                "expert": expert.name,
                "folds": int(len(scores)),
                "unanimous_win": unanimous_win,
                "mean_expert_rmse": float(scores.mean()),
                "mean_control_rmse": float(control.mean()),
                "mean_improvement": mean_improvement,
                "min_fold_improvement": float(improvements.min()),
            }
            score_rows.append(record)
            if unanimous_win and mean_improvement >= MIN_MEAN_RMSE_IMPROVEMENT:
                candidates.append(record)

        if candidates:
            selected = min(
                candidates,
                key=lambda row: (float(row["mean_expert_rmse"]), str(row["expert"])),
            )
            selected_expert = str(selected["expert"])
            improvement = float(selected["mean_improvement"])
            minimum_improvement = float(selected["min_fold_improvement"])
        else:
            selected_expert = CONTROL_NAME
            improvement = 0.0
            minimum_improvement = 0.0
        route_rows.append(
            {
                "nama_pos": station,
                "selected_expert": selected_expert,
                "expert_weight": 0.0 if selected_expert == CONTROL_NAME else EXPERT_WEIGHT,
                "control_weight": 1.0 if selected_expert == CONTROL_NAME else 1.0 - EXPERT_WEIGHT,
                "mean_rmse_improvement": improvement,
                "min_fold_rmse_improvement": minimum_improvement,
                "eligible_expert_count": len(candidates),
            }
        )
    return pd.DataFrame(route_rows), pd.DataFrame(score_rows)


def apply_routes(frame: pd.DataFrame, routes: pd.DataFrame) -> np.ndarray:
    route_lookup = routes.set_index("nama_pos")
    output = frame[f"prediction_{CONTROL_NAME}"].to_numpy(dtype=float).copy()
    for station, station_frame in frame.groupby("nama_pos", observed=True):
        route = route_lookup.loc[station]
        expert = str(route["selected_expert"])
        if expert == CONTROL_NAME:
            continue
        weight = float(route["expert_weight"])
        indices = frame.index.get_indexer(station_frame.index)
        if (indices < 0).any():
            raise ValueError("Could not align station route")
        control = station_frame[f"prediction_{CONTROL_NAME}"].to_numpy(dtype=float)
        challenger = station_frame[f"prediction_{expert}"].to_numpy(dtype=float)
        output[indices] = (1.0 - weight) * control + weight * challenger
    return output


def prediction_metrics(
    frame: pd.DataFrame, prediction: np.ndarray, model: str
) -> list[dict[str, object]]:
    work = frame[["fold", "tma_mdpl"]].copy()
    work["prediction"] = prediction
    fold_values = {
        fold: rmse(group["tma_mdpl"], group["prediction"])
        for fold, group in work.groupby("fold", sort=False)
    }
    return [
        *(
            {"model": model, "fold": fold, "rmse": value}
            for fold, value in fold_values.items()
        ),
        {
            "model": model,
            "fold": "mean_fold_rmse",
            "rmse": float(np.mean(list(fold_values.values()))),
        },
        {
            "model": model,
            "fold": "pooled_rmse",
            "rmse": rmse(work["tma_mdpl"], work["prediction"]),
        },
    ]


def leave_one_fold_out_audit(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    audit_rows: list[dict[str, object]] = []
    route_parts: list[pd.DataFrame] = []
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    folds = list(frame["fold"].drop_duplicates())
    for held_out in folds:
        train_meta = frame[frame["fold"] != held_out]
        held = frame[frame["fold"] == held_out]
        routes, _ = select_routes(train_meta)
        held_prediction = apply_routes(held, routes)
        predictions.loc[held.index] = held_prediction
        control_prediction = held[f"prediction_{CONTROL_NAME}"].to_numpy(dtype=float)
        held_rmse = rmse(held["tma_mdpl"], held_prediction)
        control_rmse = rmse(held["tma_mdpl"], control_prediction)
        audit_rows.append(
            {
                "held_out_fold": held_out,
                "selected_station_count": int(
                    (routes["selected_expert"] != CONTROL_NAME).sum()
                ),
                "held_out_rmse": held_rmse,
                "control_rmse": control_rmse,
                "delta_vs_control": held_rmse - control_rmse,
            }
        )
        route_copy = routes.copy()
        route_copy.insert(0, "held_out_fold", held_out)
        route_parts.append(route_copy)
    if predictions.isna().any():
        raise ValueError("LOFO predictions are incomplete")
    return pd.DataFrame(audit_rows), pd.concat(route_parts), predictions.to_numpy()


def load_test_predictions(sample: pd.DataFrame) -> pd.DataFrame:
    output = sample[["id"]].copy()

    def merge_submission(path: Path, name: str) -> None:
        submission = pd.read_csv(path)
        if submission["id"].duplicated().any():
            raise ValueError(f"Duplicate submission IDs: {path}")
        values = output[["id"]].merge(
            submission[["id", "tma_mdpl"]], on="id", how="left", validate="one_to_one"
        )["tma_mdpl"]
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid submission predictions: {path}")
        output[f"prediction_{name}"] = values.to_numpy(dtype=float)

    merge_submission(CONTROL_SUBMISSION_PATH, CONTROL_NAME)
    for expert in EXPERTS:
        merge_submission(expert.submission_path, expert.name)
    return output


def main() -> None:
    check_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = load_oof()

    final_routes, candidate_scores = select_routes(oof)
    fitted_oof_prediction = apply_routes(oof, final_routes)
    audit, audit_routes, lofo_prediction = leave_one_fold_out_audit(oof)
    oof["prediction_supermodel_fitted"] = fitted_oof_prediction
    oof["prediction_supermodel_lofo"] = lofo_prediction

    control_prediction = oof[f"prediction_{CONTROL_NAME}"].to_numpy(dtype=float)
    metrics = pd.DataFrame(
        prediction_metrics(oof, control_prediction, CONTROL_NAME)
        + prediction_metrics(oof, fitted_oof_prediction, "supermodel_fitted")
        + prediction_metrics(oof, lofo_prediction, "supermodel_lofo")
    )
    mean_lookup = metrics.pivot(index="model", columns="fold", values="rmse")
    mean_improves = bool(
        mean_lookup.loc["supermodel_lofo", "mean_fold_rmse"]
        < mean_lookup.loc[CONTROL_NAME, "mean_fold_rmse"]
    )
    pooled_improves = bool(
        mean_lookup.loc["supermodel_lofo", "pooled_rmse"]
        < mean_lookup.loc[CONTROL_NAME, "pooled_rmse"]
    )
    worst_fold_delta = float(audit["delta_vs_control"].max())
    audit_pass = bool(
        mean_improves
        and pooled_improves
        and worst_fold_delta <= MAX_WORST_FOLD_DELTA
    )

    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test_predictions = load_test_predictions(sample)
    test_with_station = sample[["id"]].copy()
    test_with_station["nama_pos"] = sample["id"].str.rsplit("_", n=2).str[0]
    # IDs in this competition encode the station before the final date/time
    # tokens. Validate that parsing recovers the exact routed station universe.
    if set(test_with_station["nama_pos"]) != set(final_routes["nama_pos"]):
        test_features_path = Path("output_catboost_experiments/test_features.parquet")
        test_features = pd.read_parquet(test_features_path)[["id", "nama_pos"]]
        test_with_station = sample[["id"]].merge(
            test_features, on="id", validate="one_to_one"
        )
    test_frame = test_with_station.merge(
        test_predictions, on="id", validate="one_to_one"
    )
    test_output = apply_routes(test_frame, final_routes)
    write_submission(
        OUTPUT_DIR / "submission_station_supermodel.csv",
        test_frame["id"],
        test_output,
    )

    oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False)
    final_routes.to_csv(OUTPUT_DIR / "final_station_routes.csv", index=False)
    candidate_scores.to_csv(OUTPUT_DIR / "candidate_station_scores.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "leave_one_fold_out_audit.csv", index=False)
    audit_routes.to_csv(OUTPUT_DIR / "leave_one_fold_out_routes.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)

    summary = {
        "selection_protocol": {
            "control": CONTROL_NAME,
            "required_condition": "expert beats control on every meta-training fold",
            "minimum_mean_rmse_improvement": MIN_MEAN_RMSE_IMPROVEMENT,
            "expert_weight": EXPERT_WEIGHT,
            "maximum_worst_fold_delta": MAX_WORST_FOLD_DELTA,
            "expert_pool": [expert.name for expert in EXPERTS],
        },
        "pre_api_decision": {
            "api_test_evaluated": False,
            "audit_pass": audit_pass,
            "recommendation": (
                "frozen_candidate_eligible_for_one_way_api_verification"
                if audit_pass
                else "retain_station_stack_control"
            ),
            "reason": (
                "LOFO mean and pooled RMSE improve within the worst-fold tolerance"
                if audit_pass
                else "LOFO improvement or worst-fold stability rule was not satisfied"
            ),
        },
        "final_route_counts": final_routes["selected_expert"].value_counts().to_dict(),
        "leave_one_fold_out": audit.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
        "submission": str(OUTPUT_DIR / "submission_station_supermodel.csv"),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
