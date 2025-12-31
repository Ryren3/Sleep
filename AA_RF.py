import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

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

Preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough'
)

# Fiting model

pipeline = Pipeline(steps=[('preprocessor', Preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])

model = pipeline

# RF model without hyperparameter tuning

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('='*80)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))
print('='*80)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print('='*80)
print("Trainin score:", model.score(X_train, y_train))
print("Testing score:", model.score(X_test, y_test))
print('='*80)
print("Feature Importances:")
importances = model.named_steps['classifier'].feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feature_importances)
print('='*80)

# RF model with hyperparameter tuning using GridSearchCV

param = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [5, 15, 20, 25],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

model_gs = GridSearchCV(estimator = model, param_grid=param, cv=5, scoring= 'f1_macro' ,n_jobs=-1, verbose=2)
# verbose is a parameter that tracks the progress of the model training in grid search.
model_gs.fit(X_train, y_train)

y_pred_gs = model_gs.predict(X_test)

print('='*80)
print("Classification Report (GridSearchCV):")
print(classification_report(y_test, y_pred_gs, target_names=y_encoder.classes_))
print('='*80)
print("Confusion Matrix (GridSearchCV):")
print(confusion_matrix(y_test, y_pred_gs))
print('='*80)
print("Best Parameters:")
print(model_gs.best_params_)
print('='*80)
print("CV score:", model_gs.best_score_)
print('='*80)
print('Training Accuracy:', model_gs.score(X_train, y_train))
print('Testing Accuracy:', model_gs.score(X_test, y_test))
print('='*80)
print("Feature Importances (GridSearchCV):")
importances_gs = model_gs.best_estimator_.named_steps['classifier'].feature_importances_
feature_names = model_gs.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
feature_importances_gs = pd.Series(importances_gs, index=feature_names).sort_values(ascending=False)
print(feature_importances_gs)
print('='*80)



