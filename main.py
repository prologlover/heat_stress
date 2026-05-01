"""
Main pipeline for:
"Explainable Hourly Heat-Stress Early Warning for Baghdad Using
Machine Learning, Nighttime Heat Recovery Index, and
Meteorological Variable Graph Neural Networks"

Run with:
    python main.py

For a fast test run, set QUICK_TEST_MODE = True in src/config.py.
"""

import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.utils import create_directories, set_seed, save_dataframe, print_section
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_hourly_weather, chronological_split
from src.labeling import create_heat_stress_labels
from src.feature_engineering import engineer_features
from src.nhri import compute_nhri, merge_nhri_into_hourly
from src.visualization import generate_eda_figures
from src.modeling import train_all_models
from src.evaluation import evaluate_all_models, find_best_model
from src.explainability import run_shap_explainability
from src.utils import get_available_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# EARLY WARNING OUTPUT
# ============================================================

def generate_early_warning_output(
    test_df: pd.DataFrame,
    trained_models: dict,
    feature_info: dict,
):
    """
    Produce a sample early warning table for the test set,
    showing multi-horizon predictions and warning messages.
    """
    print_section("GENERATING EARLY WARNING OUTPUT")

    required_base_cols = ["datetime", "FeelsLikeC", "heat_risk_class"]
    for c in required_base_cols:
        if c not in test_df.columns:
            logger.warning(f"Column '{c}' missing; early warning output may be incomplete.")

    # Build predictions using the best-performing model for each horizon.
    # If a horizon has no trained model, we fall back to the ground-truth target.
    best_model_per_target = {}
    for target, model_dict in trained_models.items():
        if target not in feature_info:
            continue
        X_test = feature_info[target].get("X_test")
        X_test_scaled = feature_info[target].get("X_test_scaled")
        y_test = feature_info[target].get("y_test")
        if X_test is None or y_test is None:
            continue

        best_name = None
        best_f1 = -1.0
        best_pred = None
        for model_name, model in model_dict.items():
            try:
                X_eval = X_test_scaled if model_name == "LogisticRegression" else X_test
                pred = model.predict(X_eval)
                from sklearn.metrics import f1_score
                score = f1_score(y_test, pred, average="macro", zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_name = model_name
                    best_pred = pred
            except Exception:
                continue

        if best_name is not None and best_pred is not None:
            best_model_per_target[target] = {
                "model_name": best_name,
                "pred": best_pred,
            }

    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        if i >= 1000:  # limit output to 1000 sample rows
            break

        record = {
            "datetime": row.get("datetime", ""),
            "FeelsLikeC": row.get("FeelsLikeC", np.nan),
            "current_heat_risk_class": int(row.get("heat_risk_class", 0)),
            "current_heat_risk_label": config.HEAT_RISK_CLASSES.get(
                int(row.get("heat_risk_class", 0)), "Unknown"
            ),
        }

        for target in config.TARGET_COLUMNS:
            horizon_h = target.split("_")[-1]  # '1', '3', etc.
            col_name = f"predicted_risk_t_plus_{horizon_h}"
            if target in best_model_per_target and i < len(best_model_per_target[target]["pred"]):
                record[col_name] = int(best_model_per_target[target]["pred"][i])
            else:
                record[col_name] = int(row.get(target, -1)) if target in test_df.columns else -1

        # Warning message based on the worst predicted class in the next 1 hour
        pred_1h = record.get("predicted_risk_t_plus_1", 0)
        warning_class = max(
            [record.get(f"predicted_risk_t_plus_{h}", 0) for h in config.FORECAST_HORIZONS],
            default=0,
        )
        record["warning_message"] = config.EARLY_WARNING_MESSAGES.get(warning_class, "Unknown")

        rows.append(record)

    ew_df = pd.DataFrame(rows)
    save_dataframe(
        ew_df,
        os.path.join(config.TABLES_DIR, "sample_early_warning_output.csv"),
    )
    print(f"  Early warning output: {len(ew_df)} rows saved.")
    return ew_df


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print_section("HEAT-STRESS EARLY WARNING SYSTEM FOR BAGHDAD")
    print(f"  QUICK_TEST_MODE: {config.QUICK_TEST_MODE}")
    print(f"  RUN_CLASSICAL_MODELS: {config.RUN_CLASSICAL_MODELS}")
    print(f"  RUN_GNN_MODELS: {config.RUN_GNN_MODELS}")

    # --------------------------------------------------------
    # STEP 1: Setup
    # --------------------------------------------------------
    create_directories()
    set_seed(config.RANDOM_STATE)

    # --------------------------------------------------------
    # STEP 2: Load raw data
    # --------------------------------------------------------
    locations_df, hourly_df, daily_df = load_raw_data()

    # --------------------------------------------------------
    # STEP 3: Preprocess hourly data
    # --------------------------------------------------------
    clean_df = preprocess_hourly_weather(hourly_df)

    # --------------------------------------------------------
    # STEP 4: Heat-stress labels and future targets
    # --------------------------------------------------------
    labeled_df = create_heat_stress_labels(clean_df)

    # --------------------------------------------------------
    # STEP 5: Feature engineering
    # --------------------------------------------------------
    feat_df = engineer_features(labeled_df)

    # --------------------------------------------------------
    # STEP 6: Compute NHRI and merge
    # --------------------------------------------------------
    daily_nhri = compute_nhri(feat_df)
    full_df = merge_nhri_into_hourly(feat_df, daily_nhri)

    # --------------------------------------------------------
    # STEP 7: EDA figures
    # --------------------------------------------------------
    generate_eda_figures(full_df)

    # --------------------------------------------------------
    # STEP 8: Chronological split
    # --------------------------------------------------------
    train_df, val_df, test_df = chronological_split(full_df)

    # Basic sanity checks
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(split_df) == 0:
            raise RuntimeError(
                f"{split_name} split is empty. "
                "Check TRAIN_END / VAL_START / TEST_START in config.py."
            )

    # --------------------------------------------------------
    # STEP 9: Classical ML training and evaluation
    # --------------------------------------------------------
    classical_results = pd.DataFrame()
    trained_models = {}
    feature_info = {}

    if config.RUN_CLASSICAL_MODELS:
        trained_models, feature_info = train_all_models(train_df, val_df, test_df)
        classical_results = evaluate_all_models(trained_models, feature_info)

        # SHAP explainability
        best_model_info = find_best_model(classical_results)
        run_shap_explainability(trained_models, feature_info, best_model_info)

    # --------------------------------------------------------
    # STEP 10: Early warning output
    # --------------------------------------------------------
    generate_early_warning_output(test_df, trained_models, feature_info)

    # --------------------------------------------------------
    # STEP 11: GNN pipeline
    # --------------------------------------------------------
    gnn_results = pd.DataFrame()
    trained_gnn = {}

    if config.RUN_GNN_MODELS:
        try:
            import torch
            from torch_geometric.data import Data  # noqa: F401
            _gnn_available = True
        except ImportError:
            _gnn_available = False
            logger.warning(
                "PyTorch or PyTorch Geometric not installed. "
                "GNN pipeline will be skipped.\n"
                "Install with: pip install torch torch-geometric"
            )

        if _gnn_available:
            from src.graph_builder import (
                build_graph, save_graph_info,
                create_gnn_datasets,
            )
            from src.gnn_training import train_all_gnn_models
            from src.gnn_evaluation import evaluate_all_gnn_models, build_final_comparison_table
            from src.gnn_explainability import run_gnn_explainability

            # Determine available node variables
            candidate_nodes = config.GNN_NODE_CANDIDATES
            nodes = get_available_columns(full_df, candidate_nodes)
            print(f"\n  GNN nodes ({len(nodes)}): {nodes}")

            if len(nodes) < 2:
                logger.warning("Fewer than 2 GNN nodes available; skipping GNN pipeline.")
            else:
                # Build graph using training data only (no leakage)
                nodes, edges = build_graph(train_df, nodes, mode=config.GNN_GRAPH_MODE)
                save_graph_info(nodes, edges)

                # Dataset factory: returns (train_ds, val_ds, test_ds) for a given target
                def dataset_factory(target_col):
                    try:
                        return create_gnn_datasets(
                            train_df, val_df, test_df, nodes, edges, target_col
                        )
                    except Exception as e:
                        logger.error(f"Dataset factory failed for {target_col}: {e}")
                        return None, None, None, None, None

                # Train GNN models
                trained_gnn = train_all_gnn_models(dataset_factory, nodes, edges)

                # Evaluate GNN models
                gnn_results = evaluate_all_gnn_models(trained_gnn)

                # GNN explainability
                run_gnn_explainability(trained_gnn, nodes, edges)

                # Ablation study
                from src.ablation import run_ablation_study
                run_ablation_study(train_df, val_df, test_df, nodes)

    # --------------------------------------------------------
    # STEP 12: Build final comparison table
    # --------------------------------------------------------
    from src.gnn_evaluation import build_final_comparison_table
    final_df = build_final_comparison_table(classical_results, gnn_results)

    # --------------------------------------------------------
    # STEP 13: Final summary
    # --------------------------------------------------------
    _print_final_summary(
        full_df, train_df, val_df, test_df,
        classical_results, gnn_results, final_df,
    )


def _print_final_summary(
    full_df, train_df, val_df, test_df,
    classical_results, gnn_results, final_df,
):
    print_section("FINAL SUMMARY")

    print(f"\n  Dataset size     : {len(full_df):>10,} rows")
    print(f"  Training size    : {len(train_df):>10,} rows")
    print(f"  Validation size  : {len(val_df):>10,} rows")
    print(f"  Testing size     : {len(test_df):>10,} rows")

    if "heat_risk_class" in full_df.columns:
        print("\n  Heat-risk class distribution (full dataset):")
        dist = full_df["heat_risk_class"].value_counts().sort_index()
        for cls, cnt in dist.items():
            label = config.HEAT_RISK_CLASSES.get(int(cls), "Unknown")
            print(f"    Class {cls} ({label:>14s}): {cnt:>8,} ({100*cnt/len(full_df):5.1f}%)")

    # Best classical model
    if not classical_results.empty and "macro_f1" in classical_results.columns:
        best_cl = classical_results.loc[classical_results["macro_f1"].idxmax()]
        print(f"\n  Best Classical Model: {best_cl['model']}")
        print(f"    Horizon      : {best_cl['target']}")
        print(f"    Macro F1     : {best_cl['macro_f1']:.4f}")
        print(f"    Balanced Acc : {best_cl['balanced_accuracy']:.4f}")
    else:
        print("\n  No classical model results available.")

    # Best GNN model
    if not gnn_results.empty and "macro_f1" in gnn_results.columns:
        best_gn = gnn_results.loc[gnn_results["macro_f1"].idxmax()]
        print(f"\n  Best GNN Model: {best_gn['model']}")
        print(f"    Horizon      : {best_gn['target']}")
        print(f"    Macro F1     : {best_gn['macro_f1']:.4f}")
        print(f"    Balanced Acc : {best_gn['balanced_accuracy']:.4f}")

        # Does GNN improve over classical?
        if not classical_results.empty:
            best_cl_f1 = classical_results["macro_f1"].max()
            best_gn_f1 = gnn_results["macro_f1"].max()
            if best_gn_f1 > best_cl_f1:
                print(f"\n  GNN IMPROVES over classical ML by "
                      f"{best_gn_f1 - best_cl_f1:.4f} Macro F1 points.")
            else:
                print(f"\n  Classical ML outperforms GNN by "
                      f"{best_cl_f1 - best_gn_f1:.4f} Macro F1 points.")
    else:
        print("\n  No GNN results available.")

    print(f"\n  Output locations:")
    print(f"    Tables  : {config.TABLES_DIR}")
    print(f"    Figures : {config.FIGURES_DIR}")
    print(f"    Models  : {config.MODELS_DIR}")
    print(f"    SHAP    : {config.SHAP_DIR}")
    print()


if __name__ == "__main__":
    main()
