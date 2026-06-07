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
- AI content generated is higher for students in the 'Low' category and lower for students in the 'High' category. 

---

## Preprocessing

Features dropped before modeling:

| Feature | Reason |
|---|---|
| `student_id` | Identifier, no predictive value |
| `age`, `gender` | Excluded from this analysis scope |
| `final_score` | Direct numeric representation of the target — leakage |
| `passed` | Binary derivative of `final_score` — leakage |

**Encoding:** Categorical features (`grade_level`, `ai_tools_used`, `ai_usage_purpose`) were one-hot encoded via a scikit-learn `ColumnTransformer` inside a `Pipeline`, ensuring the encoder was fit only on training data.

**Class imbalance:** Addressed using `class_weight='balanced'` in all three models, which adjusts loss weights inversely proportional to class frequencies.

**Train/test split:** 80/20 stratified split with `random_state=42`, preserving the High/Medium/Low ratio in both sets.

---

## Models

Three tree-based classifiers were trained and tuned using 5-fold cross-validated `GridSearchCV` with `f1_macro` as the optimization metric.

### 1. Decision Tree (DT)

A single decision tree, interpretable but prone to overfitting. An initial model was used without any hyperparameter tuning and a second decision tree which is optimized ont the best hyperparameters. 

**Best hyperparameters:** `criterion=gini`, `max_depth=7`, `min_samples_leaf=2`, `min_samples_split=2`

| | Initial Model | Optimized Model |
|---|---|---|
| Accuracy | 75.9% | 75.8% |
| Macro F1 | 0.71 | 0.73 |
| High F1 | 0.59 | 0.61 |
| Low F1 | 0.74 | 0.80 |
| Medium F1 | 0.79 | 0.76 |

F1 scores used for evaluation as it balances both recall and precision. The optimized decision tree model shows it is slightly superior to the initial model. 

---

### 2. Random Forest (RF)

An ensemble of 200 decision trees with bagging and feature subsampling.

**Best hyperparameters:** `n_estimators=200`, `max_depth=15`, `min_samples_leaf=4`, `min_samples_split=10`

| | Initial | Optimized |
|---|---|---|
| Training Accuracy | 100% | 96.2% |
| Testing Accuracy | 81.9% | 84% |
| Macro F1 | 0.72 | 0.81 |
| High F1 | 0.48 | 0.72 |
| Low F1 | 0.81 | 0.83 |
| Medium F1 | 0.86 | 0.87 |

The initial model perfectly memorized the training data (train score 1.0), a clear sign of overfitting. GridSearch narrowed this gap. RF showed meaningful improvement over DT on the High class, where ensemble averaging reduces variance. This model appears to be superior in classifiying students in the right performance categiry thatn Decsion tree model. 

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
| Interpretability | High | Low | Medium |

**Recommended model: XGBoost.** It matches RF on macro F1, generalizes better (smallest train/test gap), and naturally distributes importance across more features due to its shallow depth regularization — making it more trustworthy for interpreting what actually drives predictions.

**NOTE:** No models ever mistake a 'Low' category student to a 'High' category student and vice versa. Although there are some misclassification to its immediate adjacent hierarchies. Example, 'High' is mistaken as 'Medium' and vice versa and 'Low' is mistaken for 'Medium' and vice versa. This proves that the models are not randomnly guessing the student performance. 

---

## Feature Importance Insights

Across all models and after accounting for leakage concerns, the most consistently informative non-score features were:

| Feature | Observation |
|---|---|
| `ai_tools_used` (Gemini, ChatGPT+Gemini) | Top AI-related predictors — tool choice matters more than AI usage alone |
| `last_exam_score` | The most relevant feature for classification across all three models |
| `avg_assignment_score` | The second most relevant feature for classification across all three models  |
| `social_media_hours` | Weak negative signal |
| `uses_ai` (binary) | **Zero importance** in all models — whether a student uses AI is less important than how they use it |

---

## Limitations

**Class imbalance.** The High category (9.4% of data, 151 test samples) is underrepresented. Even with `class_weight='balanced'`, all models struggle to predict High-performing students reliably. A broader dataset with more High-category students would improve minority class performance.

**Imputation assumption.** Missing `ai_tools_used` and `ai_usage_purpose` values for active AI users were filled with the within-cohort mode. This preserves group-level distributions but may introduce bias for individual students who genuinely used a different tool or purpose.

**Binary `uses_ai` signal.** The `uses_ai` flag had zero importance in every model, which may partly reflect how it was constructed — students with tool names but `uses_ai=0` were corrected, but the underlying data collection process may have introduced inconsistencies that dilute this feature's signal.

**Generalizability.** The dataset contains students from mixed grade levels (10th, 11th, 12th, 1st–3rd Year university). No subgroup analysis was performed — it is possible that AI usage has different effects at different educational stages.

**No causal inference.** All findings are associative. A student using Gemini for homework assistance performing better does not mean Gemini caused that performance. Confounding factors (motivation, prior ability, access to resources) are not fully controlled for.

---
## Analysis

### 1. Model Trade-offs and Strategic Deployment

The optimized XGBoost and Random Forest architectures represent the peak of global predictive performance, both achieving a dominant Macro F1-score of 81%. However, selecting the "ideal" model is not a purely mathematical choice; it shifts dynamically based on the strategic priorities of educational professionals:

* **Scenario A: Comprehensive Population Mapping (Deploy XGBoost):** If an institution prioritizes system-wide operational accuracy and balanced classification across the entire student body, **XGBoost** is the optimal choice. It yields the highest stability and accuracy for the baseline majority cohort, capturing **90% of Medium-performing students** (845 out of 941) while maintaining a highly regularized, low-variance footprint (the lowest train/test gap at 5.4%).
* **Scenario B: Aggressive "Zero-Tolerance" Intervention (Deploy Decision Tree):** If an institution’s core mandate is to maximize support for vulnerable cohorts—either to elevate the school's overall performance metrics or to ensure no student fails—the **Optimized Decision Tree** is the superior tool. Despite a lower overall Macro F1-score (72%), the single tree acts as an aggressive diagnostic net, achieving a peak **Recall of 90% for the Low performance category** (capturing 459 out of 508 at-risk students). It generates more false alarms, but it minimizes the dangerous error of leaving a struggling student unnoticed.

---

### 2. Feature Importance vs. Operational Utility

Across all three modeling frameworks, a clear hierarchy emerges regarding feature importance, but a stark divide exists between predictive power and real-world utility:

#### The Mathematical Hierarchy
The algorithms heavily anchor their decision-making on three core academic pillars: `last_exam_score`, `assignment_scores_avg`, and `concept_understanding_score`. In the sequential XGBoost framework, this structural reliance undergoes a telling adjustment: `cat__ai_tools_used_Gemini` replaces `concept_understanding_score` in the top flight.

#### The Operational Reality
From a practical instructional standpoint, there is a fundamental difference in how these features can be leveraged:

* **`concept_understanding_score` (The Abstract Metric):** This variable is highly abstract, subjective, varies significantly from student to student, and is usually only uncovered *after* a major assessment has been completed. It is a lagging indicator that is operationally difficult to monitor in real time.
* **`last_exam_score` & `assignment_scores_avg` (The Actionable Trackers):** These features represent concrete, objective, and continuously updated transactional data automatically logged by school administration databases. 

#### Direct Prescriptive Impact
By tracking continuous assignment averages and prior exam performance, educators and counselors can build a dynamic **Early Warning System (EWS)**. Prior exam scores allow institutions to identify baseline vulnerabilities on day one of a semester, while running assignment averages act as live telemetry. This empowers professionals to execute timely, prescriptive interventions—such as targeted homework clinics or prerequisite review sessions—proactively redirecting a student's academic trajectory before they drift into a lower performance category.

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
