import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

df = pd.read_csv('ai_impact_student_performance_dataset.csv')
df = df.dropna()
df1 = df.drop(columns=['student_id', 'age','gender','grade_level','final_score','passed','performance_category'])

X = df1
x_cat = X.select_dtypes(include=['object']).columns
for cols in x_cat:
    le = LabelEncoder()
    X[cols] = le.fit_transform(X[cols])


y_encoder = LabelEncoder()
df['performance_category'] = y_encoder.fit_transform(df['performance_category'])
y = df['performance_category']  # Target variable

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# RF model without hyperparameter tuning

model = RandomForestClassifier(random_state=42, class_weight='balanced')
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
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feature_importances)
print('='*80)

# RF model with hyperparameter tuning using GridSearchCV

param = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 15, 20, 25],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
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
importances_gs = model_gs.best_estimator_.feature_importances_
feature_importances_gs = pd.Series(importances_gs, index=feature_names).sort_values(ascending=False)
print(feature_importances_gs)
print('='*80)



