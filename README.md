# Explainable Hourly Heat-Stress Early Warning for Baghdad

**Using Machine Learning, Nighttime Heat Recovery Index, and Meteorological Variable Graph Neural Networks**

---

## Research Motivation

Baghdad, Iraq is among the most heat-stressed cities in the world, regularly experiencing Feels-Like temperatures exceeding 52 °C during summer months. Accurate, multi-horizon heat-stress early warning is critical for public health response, occupational safety, and climate adaptation planning.

This project builds a complete machine learning research pipeline that:
1. Labels hourly observations into four heat-stress classes.
2. Proposes and computes a novel **Nighttime Heat Recovery Index (NHRI)** capturing cumulative overnight heat stress.
3. Trains classical ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost) for 1-h, 3-h, 6-h, 12-h, and 24-h heat-stress forecasting.
4. Builds a **Meteorological Variable Graph Neural Network (GNN)** — a novel graph model where nodes are weather variables and edges are statistical or expert-defined variable relationships.
5. Applies SHAP explainability to classical models and permutation/attention explainability to GNN models.
6. Conducts an ablation study isolating the contribution of NHRI and the effect of different graph construction modes.

---

## Dataset

| File | Description |
|------|-------------|
| `data/raw/locations.csv` | Location metadata (Baghdad: lat=33.3152, lon=44.3661) |
| `data/raw/weather_data_1hr.csv` | Hourly weather observations (2009–2024) |
| `data/raw/weather_data_24hr.csv` | Daily weather summaries |

Key variables in the hourly data: `tempC`, `FeelsLikeC`, `HeatIndexC`, `DewPointC`, `WindChillC`, `humidity`, `pressureMB`, `windspeedKmph`, `WindGustKmph`, `cloudcover`, `visibilityKm`, `precipMM`, `uvIndex`.

---

## Heat-Stress Risk Labels

Labels are assigned based on the **Feels-Like Temperature** (FeelsLikeC):

| Class | Label | FeelsLikeC Range |
|-------|-------|-----------------|
| 0 | Normal | < 32 °C |
| 1 | Caution | 32–39 °C |
| 2 | Danger | 40–51 °C |
| 3 | Extreme Danger | ≥ 52 °C |

Future prediction targets are created by shifting the current class label by the forecast horizon:
`risk_t_plus_1`, `risk_t_plus_3`, `risk_t_plus_6`, `risk_t_plus_12`, `risk_t_plus_24`

---

## Nighttime Heat Recovery Index (NHRI)

The NHRI is a novel index proposed in this study to quantify the **cumulative overnight heat burden**:

$$\text{NHRI}_d^{\theta} = \frac{\text{nighttime hours with FeelsLike} \geq \theta}{\text{total nighttime hours}}$$

Two thresholds are computed: **NHRI₃₅** (θ = 35 °C) and **NHRI₄₀** (θ = 40 °C).

| NHRI Value | Category |
|------------|----------|
| 0.00 | Full nighttime recovery |
| 0.01–0.25 | Low nighttime heat persistence |
| 0.26–0.50 | Moderate nighttime heat persistence |
| 0.51–0.75 | High nighttime heat persistence |
| 0.76–1.00 | Severe nighttime heat persistence |

---

## Classical ML Models

Six supervised classifiers are trained for each of the five forecast horizons:

| Model | Notes |
|-------|-------|
| Logistic Regression | Multinomial, balanced class weights |
| Decision Tree | max_depth=15, balanced class weights |
| Random Forest | 200 estimators, balanced class weights |
| XGBoost | Sample weights for class balance |
| LightGBM | Balanced class weights |
| CatBoost | Auto balanced class weights |

**Primary metrics**: Macro F1-score and Balanced Accuracy (robust to class imbalance).

---

## Meteorological Variable Graph GNN

Because the available dataset contains only **one geographical location (Baghdad)**, this study does not construct a spatial graph over multiple weather stations. Instead, it proposes a **Meteorological Variable Graph** where each node represents a weather variable and edges represent statistical or expert-defined relationships. This allows graph neural networks to model interactions among temperature, humidity, heat index, wind, pressure, and nighttime heat recovery features.

### Graph Definition

**Nodes** — each represents one meteorological variable:
`tempC`, `FeelsLikeC`, `HeatIndexC`, `DewPointC`, `WindChillC`, `humidity`, `pressureMB`, `windspeedKmph`, `WindGustKmph`, `cloudcover`, `visibilityKm`, `precipMM`, `uvIndex`, `NHRI_35`, `NHRI_40`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`

**Edges** — connections between variables via three modes:

| Mode | Description |
|------|-------------|
| Correlation | Pearson |r| ≥ 0.35 between variables (computed on training data only) |
| Expert | Meteorological domain knowledge (e.g. humidity↔HeatIndexC, FeelsLikeC↔NHRI) |
| Hybrid | Union of correlation and expert edges (default) |

**Node features** — for each node at time t:
`[current_t, lag_1h, lag_3h, lag_6h, lag_24h, roll_mean_3h, roll_mean_6h, roll_mean_24h]`

### GNN Architectures

| Model | Key Component |
|-------|--------------|
| VariableGCN | GCNConv (3 layers) + global mean pooling |
| VariableGAT | GATConv (multi-head attention, supports weight extraction) |
| VariableGraphSAGE | SAGEConv (3 layers) + global mean pooling |

---

## Explainability Methods

### Tabular Models (Classical ML)
- **SHAP TreeExplainer**: Summary plots, bar plots, dependence plots
- **Permutation importance**: Fallback when SHAP is unavailable

### Graph Models (GNN)
- **GAT attention weights**: Edge-level importance from GATConv layers
- **Permutation node importance**: Drop in macro F1 when a node's features are shuffled
- **Edge masking importance**: Drop in macro F1 when individual edges are removed

---

## Evaluation Metrics

| Metric | Reason |
|--------|--------|
| Macro F1-score | **Primary** — class-imbalance robust |
| Balanced Accuracy | **Primary** — class-imbalance robust |
| Accuracy | Supplementary |
| Macro Precision / Recall | Supplementary |
| Weighted F1-score | Supplementary |
| ROC-AUC (macro OVR) | Supplementary |

---

## Ablation Study

Five conditions are compared to isolate contributions of NHRI and graph construction mode:

| Condition | Description |
|-----------|-------------|
| 1 | Classical ML **without** NHRI features |
| 2 | Classical ML **with** NHRI features |
| 3 | GNN with correlation-only graph |
| 4 | GNN with expert-only graph |
| 5 | GNN with hybrid graph (default) |

---

## Project Folder Structure

```
heat_stress_baghdad/
│
├── data/
│   ├── raw/
│   │   ├── locations.csv
│   │   ├── weather_data_1hr.csv
│   │   └── weather_data_24hr.csv
│   └── processed/
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   ├── shap/
│   └── gnn/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── config.py            – All settings and hyperparameters
│   ├── utils.py             – Shared utilities
│   ├── data_loader.py       – Raw CSV loading and summary
│   ├── preprocessing.py     – Cleaning and chronological splitting
│   ├── labeling.py          – Heat-risk class assignment
│   ├── feature_engineering.py – Time, lag, rolling, interaction features
│   ├── nhri.py              – Nighttime Heat Recovery Index
│   ├── visualization.py     – All figures (EDA + evaluation)
│   ├── modeling.py          – Classical ML training
│   ├── evaluation.py        – Classical ML evaluation
│   ├── explainability.py    – SHAP and permutation importance
│   ├── graph_builder.py     – Meteorological variable graph + PyG dataset
│   ├── gnn_model.py         – GCN, GAT, GraphSAGE model definitions
│   ├── gnn_training.py      – GNN training with early stopping
│   ├── gnn_evaluation.py    – GNN evaluation and final comparison table
│   ├── gnn_explainability.py – GAT attention, node/edge importance
│   └── ablation.py          – Ablation study
│
├── main.py                  – Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
# Install PyTorch first (see https://pytorch.org/get-started/)
# CPU-only example:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install the rest:
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

### 3. Quick test mode

Edit `src/config.py` and set:
```python
QUICK_TEST_MODE = True
```
This trains only RandomForest + VariableGCN for `risk_t_plus_1` in a fraction of the time.

---

## PyTorch Geometric Installation Notes

### Windows

```bash
pip install torch torch-geometric
```

If that fails, visit https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html for the wheel that matches your PyTorch version and CUDA version.

### Google Colab

```bash
pip install torch torch-geometric
```

---

## Expected Outputs

### Tables (`outputs/tables/`)
`dataset_summary.csv`, `heat_risk_distribution.csv`, `future_target_distributions.csv`, `split_summary.csv`, `nhri_daily.csv`, `nhri_summary.csv`, `model_comparison.csv`, `shap_feature_importance.csv`, `sample_early_warning_output.csv`, `gnn_node_list.csv`, `gnn_edge_list.csv`, `gnn_model_comparison.csv`, `gnn_node_importance.csv`, `gnn_edge_importance.csv`, `gnn_gat_attention_edges.csv`, `ablation_study_results.csv`, `final_model_comparison_all_models.csv`

### Figures (`outputs/figures/`)
`class_distribution.png`, `monthly_feelslike.png`, `yearly_max_feelslike.png`, `monthly_danger_frequency.png`, `hourly_heat_risk_frequency.png`, `day_night_heat_comparison.png`, `yearly_nhri_35.png`, `yearly_nhri_40.png`, `nhri_category_distribution.png`, `correlation_heatmap.png`, `macro_f1_comparison.png`, `balanced_accuracy_comparison.png`, confusion matrices, SHAP plots, GNN training curves, GNN explainability plots, ablation plots

### Models (`outputs/models/`)
Classical: `model_{name}_{target}.joblib`
GNN: `gnn_{model}_{target}.pt`

---

## Important Research Notes

1. **This is NOT a spatial GNN** — there is only one geographical location (Baghdad).
2. **This IS a variable-relationship GNN** — nodes are meteorological variables, not stations.
3. **The graph structure is static** — the same graph topology is used for every timestamp.
4. **Node features are dynamic** — they change at each hourly timestep.
5. **The target is future heat-stress class** — multi-horizon classification.
6. **Correlation graph uses training data only** — no leakage from validation or test.
7. **GNN results must be compared against strong tabular baselines** (XGBoost, LightGBM, CatBoost).
8. **GNN is scientifically useful** even if it does not outperform tabular models, because it provides relationship-level interpretability through attention and node importance.
9. **No random splitting** — all splits are strictly chronological to respect time-series structure.
10. **No SMOTE before splitting** — class imbalance is handled with class weights inside each model.

---

## Citation

If you use this code in your research, please cite the corresponding paper:

> [Paper title and authors — to be filled in upon publication]
