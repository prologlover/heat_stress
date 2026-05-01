"""
Configuration settings for the Heat-Stress Early Warning System for Baghdad.
All project-wide constants, paths, and hyperparameters are defined here.
"""

import os

# ============================================================
# BASE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
SHAP_DIR = os.path.join(OUTPUTS_DIR, "shap")
GNN_DIR = os.path.join(OUTPUTS_DIR, "gnn")

# Sub-directories
CLASSIFICATION_REPORTS_DIR = os.path.join(TABLES_DIR, "classification_reports")
CONFUSION_MATRICES_DIR = os.path.join(FIGURES_DIR, "confusion_matrices")
GNN_CLASSIFICATION_REPORTS_DIR = os.path.join(TABLES_DIR, "gnn_classification_reports")
GNN_CONFUSION_MATRICES_DIR = os.path.join(FIGURES_DIR, "gnn_confusion_matrices")

# ============================================================
# RAW FILE PATHS
# ============================================================
LOCATIONS_CSV = os.path.join(RAW_DIR, "locations.csv")
HOURLY_CSV = os.path.join(RAW_DIR, "weather_data_1hr.csv")
DAILY_CSV = os.path.join(RAW_DIR, "weather_data_24hr.csv")

# ============================================================
# PROCESSED FILE PATHS
# ============================================================
CLEAN_HOURLY_CSV = os.path.join(PROCESSED_DIR, "clean_hourly_weather.csv")
FEATURE_ENGINEERED_CSV = os.path.join(PROCESSED_DIR, "feature_engineered_hourly_weather.csv")
NHRI_DAILY_CSV = os.path.join(PROCESSED_DIR, "nhri_daily.csv")
HOURLY_WITH_NHRI_CSV = os.path.join(PROCESSED_DIR, "hourly_with_nhri.csv")

# ============================================================
# REPRODUCIBILITY
# ============================================================
RANDOM_STATE = 42

# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================
TRAIN_END = "2020-12-31 23:00:00"
VAL_START = "2021-01-01 00:00:00"
VAL_END = "2022-12-31 23:00:00"
TEST_START = "2023-01-01 00:00:00"

# ============================================================
# HEAT-STRESS THRESHOLDS (based on FeelsLikeC)
# ============================================================
HEAT_STRESS_THRESHOLDS = {
    "Normal": (None, 32),          # FeelsLikeC < 32
    "Caution": (32, 40),           # 32 <= FeelsLikeC < 40
    "Danger": (40, 52),            # 40 <= FeelsLikeC < 52
    "Extreme Danger": (52, None),  # FeelsLikeC >= 52
}

HEAT_RISK_CLASSES = {
    0: "Normal",
    1: "Caution",
    2: "Danger",
    3: "Extreme Danger",
}

HEAT_RISK_LABELS_REVERSE = {v: k for k, v in HEAT_RISK_CLASSES.items()}

# ============================================================
# FORECAST HORIZONS
# ============================================================
FORECAST_HORIZONS = [1, 3, 6, 12, 24]  # in hours

TARGET_COLUMNS = [f"risk_t_plus_{h}" for h in FORECAST_HORIZONS]
# ['risk_t_plus_1', 'risk_t_plus_3', 'risk_t_plus_6', 'risk_t_plus_12', 'risk_t_plus_24']

# ============================================================
# EARLY WARNING MESSAGES
# ============================================================
EARLY_WARNING_MESSAGES = {
    0: "Safe conditions",
    1: "Use caution during outdoor activities",
    2: "Avoid long outdoor exposure",
    3: "Severe risk; outdoor activity should be minimized",
}

# ============================================================
# NIGHTTIME HEAT RECOVERY INDEX (NHRI)
# ============================================================
NHRI_THRESHOLDS = [35, 40]  # FeelsLikeC thresholds for NHRI calculation

NHRI_CATEGORY_BINS = [0.0, 0.01, 0.26, 0.51, 0.76, 1.01]
NHRI_CATEGORY_LABELS = [
    "Full nighttime recovery",
    "Low nighttime heat persistence",
    "Moderate nighttime heat persistence",
    "High nighttime heat persistence",
    "Severe nighttime heat persistence",
]

# ============================================================
# CLASSICAL MACHINE LEARNING SETTINGS
# ============================================================
RUN_CLASSICAL_MODELS = True

ML_MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

# Columns to exclude from features (potential leakage or metadata)
EXCLUDE_FROM_FEATURES = [
    "datetime", "date", "time", "loc_id",
    "heat_risk_class", "heat_risk_label",
    "risk_t_plus_1", "risk_t_plus_3", "risk_t_plus_6",
    "risk_t_plus_12", "risk_t_plus_24",
    "weatherIconUrl", "weatherDesc", "winddir16point",
    "tempF", "windspeedMiles", "FeelsLikeF", "HeatIndexF",
    "DewPointF", "WindChillF", "WindGustMiles",
    "visibilityMiles", "pressureInches", "precipInches",
    "maxtempF", "mintempF", "avgtempF",
    "sunrise", "sunset", "moonrise", "moonset",
    "moon_phase", "moon_illumination",
    "nhri_category_35", "nhri_category_40",
]

# ============================================================
# GNN SETTINGS
# ============================================================
RUN_GNN_MODELS = True

GNN_GRAPH_MODE = "hybrid"  # Options: "correlation", "expert", "hybrid"

GNN_MODELS = ["GCN", "GAT", "GraphSAGE"]

GNN_HORIZONS = TARGET_COLUMNS  # All five horizons

GNN_CORRELATION_THRESHOLD = 0.35

GNN_HIDDEN_DIM = 64
GNN_DROPOUT = 0.3
GNN_LR = 0.001
GNN_MAX_EPOCHS = 100
GNN_PATIENCE = 10
GNN_BATCH_SIZE = 64
GNN_NUM_CLASSES = 4

# Candidate node variables for the meteorological variable graph
GNN_NODE_CANDIDATES = [
    "tempC", "FeelsLikeC", "HeatIndexC", "DewPointC", "WindChillC",
    "humidity", "pressureMB", "windspeedKmph", "WindGustKmph",
    "cloudcover", "visibilityKm", "precipMM", "uvIndex",
    "NHRI_35", "NHRI_40",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# Expert-defined edges (meteorological domain knowledge)
GNN_EXPERT_EDGES = [
    ("tempC", "FeelsLikeC"),
    ("tempC", "HeatIndexC"),
    ("tempC", "DewPointC"),
    ("tempC", "WindChillC"),
    ("humidity", "HeatIndexC"),
    ("humidity", "DewPointC"),
    ("humidity", "FeelsLikeC"),
    ("windspeedKmph", "WindChillC"),
    ("windspeedKmph", "FeelsLikeC"),
    ("WindGustKmph", "windspeedKmph"),
    ("pressureMB", "cloudcover"),
    ("cloudcover", "visibilityKm"),
    ("precipMM", "humidity"),
    ("precipMM", "cloudcover"),
    ("uvIndex", "tempC"),
    ("uvIndex", "HeatIndexC"),
    ("NHRI_35", "FeelsLikeC"),
    ("NHRI_40", "FeelsLikeC"),
    ("NHRI_35", "HeatIndexC"),
    ("NHRI_40", "HeatIndexC"),
    ("hour_sin", "tempC"),
    ("hour_cos", "tempC"),
    ("month_sin", "tempC"),
    ("month_cos", "tempC"),
]

# ============================================================
# QUICK TEST MODE
# ============================================================
QUICK_TEST_MODE = False
# When True:
# - Train only RandomForest (classical) and GCN (GNN)
# - Train only risk_t_plus_1
# - Use fewer estimators and fewer epochs

QUICK_CLASSICAL_MODELS = ["RandomForest"]
QUICK_GNN_MODELS = ["GCN"]
QUICK_HORIZONS = ["risk_t_plus_1"]
QUICK_N_ESTIMATORS = 50
QUICK_GNN_MAX_EPOCHS = 10

# ============================================================
# FIGURE SETTINGS
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
