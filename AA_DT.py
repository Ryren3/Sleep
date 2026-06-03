import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

# 1. Ingest clean dataset
df = pd.read_csv('cleaned_ai_impact_dataset.csv')

# ==============================================================================
# 🔥 FIX FOR PANDAS 3.0 & SCIKIT-LEARN TYPE MISMATCH:
# Find all columns that are read as 'string' or 'object' and cast them explicitly
# to standard 'object' so scikit-learn's underlying C-extensions can parse them.
# ==============================================================================
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].astype('object')

print(df)
print('='*80)
print(df.info())
print('='*80)
print(df.describe())
print('='*80)

# 2. Separate Features and Targets (Dropping target leakage metrics)
# final_score removed becuase it is a direct numeric representation of the target variable performance_category, and passed is also a direct binary representation of performance_category. Both would lead to perfect predictions without learning any meaningful patterns.
X = df.drop(columns=['age','gender','student_id', 'final_score', 'passed', 'performance_category'])


X = X.astype({
    col: 'object'
    for col in X.select_dtypes(include=['string']).columns
})


# 3. Target Variable Label Encoding
y_encoder = LabelEncoder()
df['performance_category'] = y_encoder.fit_transform(df['performance_category'])
y = df['performance_category']  

# 4. Stratified Split to preserve class distributions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Extract our safe 'object' text features for the preprocessor
cat_cols = list(X.select_dtypes(include=['object']).columns)

# 6. Build the Column Preprocessor Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
], remainder='passthrough')

# 7. Initialize Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
])

# 8. Train Initial Model (This will now run completely error-free!)
pipeline.fit(X_train, y_train)

# 9. Initial Model Metrics
y_pred = pipeline.predict(X_test)
print("Initial Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print('='*80)
print("Initial Model Accuracy:", accuracy_score(y_test, y_pred))
print('='*80)
print("Initial Model Classification Report:")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))
print('='*80)

# 10. Plot Initial Tree using Extracted Attributes
trained_tree = pipeline.named_steps['classifier']
encoded_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

plt.figure(figsize=(20, 10))
plot_tree(trained_tree, filled=True, feature_names=encoded_feature_names, class_names=y_encoder.classes_, max_depth=3)
plt.title("Initial Decision Tree Structure (Capped Depth for Readability)")
plt.show()

# 11. Feature Importances for the Initial Model
print('='*80)
print("Feature Importances for Initial Model:")
print('='*80)
initial_importances = pipeline.named_steps['classifier'].feature_importances_
initial_feature_df = pd.Series(initial_importances, index=encoded_feature_names).sort_values(ascending=False)
print(initial_feature_df.head(10))
print('='*80)

# ==============================================================================
# DECISION TREE WITH HYPERPARAMETER GRID SEARCH
# ==============================================================================
print("Decision Tree with Grid Search Running...")
print('='*80)

param_grid = {
    'classifier__max_depth': [3, 5, 7, 10, 11],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print('Best CV Macro F1-Score:', grid_search.best_score_)
print('='*80)

# 12. Evaluate Optimized Grid Search Model
y_pred_gs = best_pipeline.predict(X_test)
print("Optimized Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gs))
print('='*80)
print("Optimized Model Accuracy:", accuracy_score(y_test, y_pred_gs))
print('='*80)
print("Optimized Model Classification Report:")
print(classification_report(y_test, y_pred_gs, target_names=y_encoder.classes_))
print('='*80)

# 13. Plot Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, y_pred_gs), annot=True, fmt='d', cmap='Blues', 
            xticklabels=y_encoder.classes_, yticklabels=y_encoder.classes_)
plt.title("Confusion Matrix - Optimized Decision Tree")
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

# 14. Optimized Feature Importances
print('='*80)
print("Feature Importances for Optimized Model:")
print('='*80)
gs_encoded_features = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
gs_importances = best_pipeline.named_steps['classifier'].feature_importances_
gs_feature_df = pd.Series(gs_importances, index=gs_encoded_features).sort_values(ascending=False)
print(gs_feature_df.head(10))
print('='*80)
print('End of AA_DT.py')