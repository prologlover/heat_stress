"""
Classical machine learning modeling module.
Trains Logistic Regression, Decision Tree, Random Forest, XGBoost,
LightGBM, and CatBoost for each forecast horizon.
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.utils import print_section, save_dataframe

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Optional imports
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("XGBoost not installed; skipping.")

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("LightGBM not installed; skipping.")

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False
    logger.warning("CatBoost not installed; skipping.")


# ============================================================
# FEATURE PREPARATION
# ============================================================

_CATEGORICAL_ENCODE = ["season", "winddir16point", "weatherDesc", "isdaytime"]


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of input feature columns, excluding all excluded and
    leaky columns as defined in config.EXCLUDE_FROM_FEATURES.
    """
    exclude = set(config.EXCLUDE_FROM_FEATURES)
    cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("risk_t_plus_")
        and c != "heat_risk_class"
        and c != "heat_risk_label"
    ]
    return cols


def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
):
    """
    Encode categoricals and return aligned (X_train, X_val, X_test).
    Encoders are fitted on training data only.
    """
    # Encode known categorical columns + any residual object/category columns.
    cat_cols = [c for c in _CATEGORICAL_ENCODE if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    def _select(df):
        return df[[c for c in feature_cols if c in df.columns]].copy()

    X_train = _select(train_df)
    X_val = _select(val_df)
    X_test = _select(test_df)

    # Auto-detect additional categorical columns not listed explicitly.
    inferred_cat = [
        c for c in X_train.columns
        if str(X_train[c].dtype) in ("object", "category", "bool")
    ]
    cat_cols = sorted(set(cat_cols + inferred_cat))
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Ordinal-encode categorical columns (fit on train only)
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(X_train[cat_cols].astype(str))
        for X in [X_train, X_val, X_test]:
            X[cat_cols] = enc.transform(X[cat_cols].astype(str))

    # Fill any residual NaNs with training-set statistics only (no leakage)
    medians = X_train[num_cols].median(numeric_only=True) if num_cols else pd.Series(dtype=float)
    if num_cols:
        X_train[num_cols] = X_train[num_cols].fillna(medians)
        X_val[num_cols] = X_val[num_cols].fillna(medians)
        X_test[num_cols] = X_test[num_cols].fillna(medians)

    if cat_cols:
        # Encoded categoricals are numeric now; missing categories mapped to -1
        X_train[cat_cols] = X_train[cat_cols].fillna(-1)
        X_val[cat_cols] = X_val[cat_cols].fillna(-1)
        X_test[cat_cols] = X_test[cat_cols].fillna(-1)

    return X_train, X_val, X_test, medians


# ============================================================
# MODEL FACTORY
# ============================================================

def _build_models(n_estimators: int = 200) -> dict:
    """Return a dictionary of model name -> unfitted model instance."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=config.RANDOM_STATE, multi_class="multinomial", solver="lbfgs",
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", random_state=config.RANDOM_STATE, max_depth=15,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=n_estimators, use_label_encoder=False,
            eval_metric="mlogloss", random_state=config.RANDOM_STATE,
            tree_method="hist", verbosity=0,
        )
    if _HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=config.RANDOM_STATE, verbose=-1,
        )
    if _HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=n_estimators, random_seed=config.RANDOM_STATE,
            verbose=0, auto_class_weights="Balanced",
        )
    return models


# ============================================================
# TRAINING
# ============================================================

def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: list = None,
) -> dict:
    """
    Train all classical ML models for each forecast horizon.

    Parameters
    ----------
    train_df, val_df, test_df : chronologically split DataFrames
    targets                   : list of target column names

    Returns
    -------
    trained_models : dict  {target: {model_name: fitted_model}}
    feature_info   : dict  {target: {'cols': [...], 'medians': pd.Series}}
    """
    print_section("TRAINING CLASSICAL ML MODELS")

    if targets is None:
        if config.QUICK_TEST_MODE:
            targets = config.QUICK_HORIZONS
        else:
            targets = config.TARGET_COLUMNS

    if config.QUICK_TEST_MODE:
        model_names_to_run = config.QUICK_CLASSICAL_MODELS
        n_est = config.QUICK_N_ESTIMATORS
    else:
        model_names_to_run = config.ML_MODELS
        n_est = 200

    feature_cols = get_feature_columns(train_df)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:10]} ...")

    X_train, X_val, X_test, medians = prepare_features(
        train_df, val_df, test_df, feature_cols
    )

    # StandardScaler for Logistic Regression (fit on train only)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    all_models_dict = _build_models(n_estimators=n_est)
    # Filter to requested models
    all_models_dict = {
        k: v for k, v in all_models_dict.items()
        if k in model_names_to_run or not config.QUICK_TEST_MODE
    }

    trained_models = {}
    feature_info = {}

    for target in targets:
        if target not in train_df.columns:
            logger.warning(f"Target '{target}' not found in training data; skipping.")
            continue

        print(f"\n  --- Target: {target} ---")
        y_train = train_df[target].astype(int)
        y_val = val_df[target].astype(int)
        y_test = test_df[target].astype(int)

        trained_models[target] = {}

        for model_name, model in all_models_dict.items():
            if model_name not in model_names_to_run and config.QUICK_TEST_MODE:
                continue

            print(f"    Training {model_name} ...", end=" ", flush=True)

            # Choose scaled or unscaled features
            if model_name == "LogisticRegression":
                Xtr, Xva, Xte = X_train_scaled, X_val_scaled, X_test_scaled
            else:
                Xtr, Xva, Xte = X_train, X_val, X_test

            # Class weighting for XGBoost (sample weights)
            fit_kwargs = {}
            if model_name == "XGBoost":
                sw = compute_sample_weight("balanced", y_train)
                fit_kwargs["sample_weight"] = sw

            try:
                import copy
                m = copy.deepcopy(model)
                m.fit(Xtr, y_train, **fit_kwargs)
                trained_models[target][model_name] = m

                # Save model
                model_path = os.path.join(
                    config.MODELS_DIR, f"model_{model_name}_{target}.joblib"
                )
                joblib.dump(m, model_path)
                print(f"saved -> {model_path}")

            except Exception as e:
                logger.error(f"Failed to train {model_name} for {target}: {e}")
                continue

        feature_info[target] = {
            "cols": feature_cols,
            "medians": medians,
            "scaler": scaler,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "X_train_scaled": X_train_scaled,
            "X_val_scaled": X_val_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    return trained_models, feature_info
