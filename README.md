# Student Performance Category Prediction

A machine learning project that predicts student academic performance categories — **High**, **Medium**, or **Low** — using a rich set of behavioral, academic, lifestyle, and AI usage features. While AI usage is one of several dimensions explored, the core objective is to understand the full landscape of factors that drive student outcomes.

---

## Dataset Overview

| Property | Detail |
|---|---|
| Source file | `ai_impact_student_performance_dataset.csv` |
| Rows | 8,000 students |
| Columns | 26 |
| Target variable | `performance_category` (High / Medium / Low) |
| Missing values | `ai_tools_used` (1,362), `ai_usage_purpose` (1,346) — all others complete |

### Feature Groups

**Demographics**
- `student_id`, `age`, `gender`, `grade_level`

**AI Usage** (8 features)
- `uses_ai`, `ai_usage_time_minutes`, `ai_tools_used`, `ai_usage_purpose`
- `ai_dependency_score`, `ai_generated_content_percentage`, `ai_prompts_per_week`, `ai_ethics_score`

**Academic Performance**
- `last_exam_score`, `assignment_scores_avg`, `concept_understanding_score`
- `final_score`, `passed`

**Study Habits & Engagement**
- `study_hours_per_day`, `study_consistency_index`, `improvement_rate`
- `attendance_percentage`, `tutoring_hours`, `class_participation_score`

**Lifestyle**
- `sleep_hours`, `social_media_hours`

### Target Distribution

| Category | Count | % of Dataset | Score Range |
|---|---|---|---|
| Medium | 4,705 | 58.8% | 50.1 – 75.0 |
| Low | 2,542 | 31.8% | 12.7 – 50.0 |
| High | 753 | 9.4% | 75.1 – 95.8 |

The dataset is moderately imbalanced, with High-performing students representing fewer than 1 in 10 records.

---

## Exploratory Data Analysis (EDA)

### Missing Value Treatment

The 1,362 nulls in `ai_tools_used` and 1,346 in `ai_usage_purpose` split into two structurally different groups:

- **900 active AI users** with missing tool/purpose fields — filled using the mode within each `performance_category` cohort, preserving group-level patterns.
- **462 non-AI users** (`uses_ai = 0`) — assigned the explicit category `'None'`, reflecting true non-usage rather than missing data.

A data integrity check also identified students flagged as `uses_ai = 0` who had tool names recorded; their `uses_ai` flag was corrected to 1.

### Key Observations from EDA

- `final_score` maps directly and cleanly onto `performance_category` bands (High: 75–96, Medium: 50–75, Low: 13–50), confirming the target is score-derived.
- `passed` (binary) has a hard threshold at 40.0 — students with `final_score ≥ 40` all have `passed = 1`, making it a redundant feature.
- The pass rate is 88.9%, reflecting that the Low category partially overlaps with the passing threshold.
- `uses_ai` showed near-zero feature importance across all three models, suggesting that simply using AI (yes/no) is not a differentiating factor — what matters more is *how* it is used.
- Social media hours showed a weak negative correlation with final scores.
- Study hours showed a moderate positive correlation with final scores.

---

## Preprocessing

Features dropped before modeling:

| Feature | Reason |
|---|---|
| `student_id` | Identifier, no predictive value |
| `age`, `gender` | Excluded from this analysis scope |
| `final_score` | Direct numeric representation of the target — leakage |
| `passed` | Binary derivative of `final_score` — leakage |
| `last_exam_score` | Highly correlated with target, near-direct proxy — leakage |
| `assignment_scores_avg` | Same as above — leakage |
| `concept_understanding_score` | Same as above — leakage |

> **Note on leakage:** Even though `last_exam_score`, `assignment_scores_avg`, and `concept_understanding_score` are not `final_score` itself, they are sufficiently correlated with the target bands that including them causes models to split almost entirely on score thresholds rather than learning behavioral patterns. Feature importance scores confirmed this — those three features alone accounted for 93.6% of all splits in the Decision Tree, 64% in Random Forest, and 33.7% in XGBoost.

**Encoding:** Categorical features (`grade_level`, `ai_tools_used`, `ai_usage_purpose`) were one-hot encoded via a scikit-learn `ColumnTransformer` inside a `Pipeline`, ensuring the encoder was fit only on training data.

**Class imbalance:** Addressed using `class_weight='balanced'` in all three models, which adjusts loss weights inversely proportional to class frequencies.

**Train/test split:** 80/20 stratified split with `random_state=42`, preserving the High/Medium/Low ratio in both sets.

---

## Models

Three tree-based classifiers were trained and tuned using 5-fold cross-validated `GridSearchCV` with `f1_macro` as the optimization metric.

### 1. Decision Tree (DT)

A single decision tree, interpretable but prone to overfitting.

**Best hyperparameters:** `criterion=gini`, `max_depth=7`, `min_samples_leaf=2`, `min_samples_split=2`

| | Initial | Optimized |
|---|---|---|
| Accuracy | 75.9% | 75.8% |
| Macro F1 | 0.71 | 0.73 |
| High F1 | 0.61 | 0.61 |
| Low F1 | 0.74 | 0.80 |
| Medium F1 | 0.80 | 0.76 |

The optimized model improved recall for the minority High class (0.33 → 0.85) at the cost of Medium precision — a trade-off driven by `class_weight='balanced'` and the shallower depth constraint.

---

### 2. Random Forest (RF)

An ensemble of 200 decision trees with bagging and feature subsampling.

**Best hyperparameters:** `n_estimators=200`, `max_depth=15`, `min_samples_leaf=4`, `min_samples_split=10`

| | Initial | Optimized |
|---|---|---|
| Training Accuracy | 100% | 96.2% |
| Testing Accuracy | 81.9% | 80.5% |
| Macro F1 | 0.72 | 0.81 |
| High F1 | 0.48 | 0.72 |
| Low F1 | 0.81 | 0.83 |
| Medium F1 | 0.86 | 0.87 |

The initial model perfectly memorized the training data (train score 1.0), a clear sign of overfitting. GridSearch narrowed this gap. RF showed meaningful improvement over DT on the High class, where ensemble averaging reduces variance.

---

### 3. XGBoost

A gradient boosted ensemble with built-in regularization via learning rate and tree depth constraints.

**Best hyperparameters:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.15`, `subsample=1.0`, `colsample_bytree=1.0`

| | Initial | Optimized |
|---|---|---|
| Training Accuracy | 100% | 86.5% |
| Testing Accuracy | 84.7% | 81.1% |
| Macro F1 | 0.80 | 0.81 |
| High F1 | 0.70 | 0.71 |
| Low F1 | 0.84 | 0.85 |
| Medium F1 | 0.87 | 0.88 |

The GridSearch selected `max_depth=3`, which aggressively constrains tree depth and reduced the train/test gap from 0.153 to 0.054 — the best generalization behavior of the three models.

---

## Model Comparison

| Metric | DT | RF | XGBoost |
|---|---|---|---|
| Macro F1 (optimized) | 0.73 | 0.81 | **0.81** |
| High class F1 | 0.61 | 0.72 | **0.71** |
| Train/Test gap | ~0.00 | 0.157 | **0.054** |
| Leaky feature dominance | 93.6% | 64.0% | **36.4%** |
| Interpretability | High | Low | Medium |

**Recommended model: XGBoost.** It matches RF on macro F1, generalizes better (smallest train/test gap), and naturally distributes importance across more features due to its shallow depth regularization — making it more trustworthy for interpreting what actually drives predictions.

---

## Feature Importance Insights

Across all models and after accounting for leakage concerns, the most consistently informative non-score features were:

| Feature | Observation |
|---|---|
| `ai_tools_used` (Gemini, ChatGPT+Gemini) | Top AI-related predictors — tool choice matters more than AI usage alone |
| `ai_usage_purpose_Homework` | Students using AI for homework showed distinct performance patterns |
| `ai_generated_content_percentage` | Higher AI-generated content correlated with performance differences |
| `improvement_rate` | Consistent across all models as a behavioral signal |
| `attendance_percentage` | Traditional engagement metric remains relevant |
| `study_consistency_index` | Regularity of study more predictive than raw hours |
| `social_media_hours` | Weak negative signal |
| `uses_ai` (binary) | **Zero importance** in all models — whether a student uses AI is less important than how they use it |

---

## Limitations

**Score-based leakage risk.** `last_exam_score`, `assignment_scores_avg`, and `concept_understanding_score` were identified as near-direct proxies for `performance_category`. When included, they dominate feature importance and mask genuine behavioral signals. The analysis above reflects models where these were retained — results should be interpreted with this caveat. Ideally, a separate model trained without these features would better isolate the contribution of AI usage and lifestyle factors.

**Class imbalance.** The High category (9.4% of data, 151 test samples) is underrepresented. Even with `class_weight='balanced'`, all models struggle to predict High-performing students reliably. A broader dataset with more High-category students would improve minority class performance.

**Imputation assumption.** Missing `ai_tools_used` and `ai_usage_purpose` values for active AI users were filled with the within-cohort mode. This preserves group-level distributions but may introduce bias for individual students who genuinely used a different tool or purpose.

**Binary `uses_ai` signal.** The `uses_ai` flag had zero importance in every model, which may partly reflect how it was constructed — students with tool names but `uses_ai=0` were corrected, but the underlying data collection process may have introduced inconsistencies that dilute this feature's signal.

**Cross-sectional snapshot.** The dataset captures a single point in time. Performance trends, changes in AI adoption habits, and the cumulative effect of AI usage over a semester cannot be captured by this model.

**Generalizability.** The dataset contains students from mixed grade levels (10th, 11th, 12th, 1st–3rd Year university). No subgroup analysis was performed — it is possible that AI usage has different effects at different educational stages.

**No causal inference.** All findings are associative. A student using Gemini for homework assistance performing better does not mean Gemini caused that performance. Confounding factors (motivation, prior ability, access to resources) are not fully controlled for.

---

## Repository Structure

```
├── ai_impact_student_performance_dataset.csv   # Raw data
├── cleaned_ai_impact_dataset.csv               # Cleaned data (output of AA.py)
├── AA.py                                       # EDA and data cleaning
├── AA_DT.py                                    # Decision Tree model
├── AA_RF.py                                    # Random Forest model
├── AA_XGB.py                                   # XGBoost model
└── README.md
```

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
```

Install with:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```
