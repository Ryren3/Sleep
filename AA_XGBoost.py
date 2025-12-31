import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv('ai_impact_student_performance_dataset.csv')
df = df.dropna()
df1 = df.drop(columns=['student_id', 'age','gender','grade_level','final_score','passed','performance_category'])
X = df1

y_encoder = LabelEncoder()
df['performance_category'] = y_encoder.fit_transform(df['performance_category'])
y = df['performance_category']  # Target variable

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Encoding using piipeline

cat_cols = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough'
)


# XGBoost model without hyperparameter tuning

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(eval_metric='mlogloss', random_state=42))
])

model = pipeline

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('='*80)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))
print('='*80)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print('='*80)
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))
print('='*80)
print('Feature Importances:')

feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = pipeline.named_steps['classifier'].feature_importances_

feature_df = pd.Series(importances, index = feature_names).sort_values(ascending=False)

print('Top 10 features:')
print(feature_df.head(10))
print('='*80)


##################################################
# XGBoost model with grid search
##################################################


param = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.15, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

gs = GridSearchCV(estimator = model, param_grid = param, scoring = 'f1_macro', verbose= 1, cv = 5)

model_gs = gs.fit(X_train, y_train)
y_pred_gs = model_gs.predict(X_test)
print('='*80)
print("Best Hyperparameters:", model_gs.best_params_)
print('='*80)
print("Classification Report after Hyperparameter Tuning:")
print(classification_report(y_test, y_pred_gs, target_names=y_encoder.classes_))
print('='*80)
print("Confusion Matrix after Hyperparameter Tuning:")
print(confusion_matrix(y_test, y_pred_gs))
print('='*80)
print("Training Accuracy after Hyperparameter Tuning:", model_gs.score(X_train, y_train))
print("Testing Accuracy after Hyperparameter Tuning:", model_gs.score(X_test, y_test))
print('='*80)
print('Feature Importances after Hyperparameter Tuning:')
feature_names_gs = model_gs.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
importances_gs = model_gs.best_estimator_.named_steps['classifier'].feature_importances_
feature_df_gs = pd.Series(importances_gs, index = feature_names_gs).sort_values(ascending=False)
print('Top 10 features after Hyperparameter Tuning:')
print(feature_df_gs.head(10))
print('='*80)






