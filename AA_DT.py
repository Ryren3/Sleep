import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree

df = pd.read_csv('ai_impact_student_performance_dataset.csv')
df = df.dropna()

# df1 has no target variables
X = df.drop(columns=['student_id', 'age','gender','grade_level','final_score','passed','performance_category'])

x_cat = X.select_dtypes(include=['object']).columns

encoders = {}
for col in x_cat:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le



print(X)

y_encoder = LabelEncoder()
df['performance_category'] = y_encoder.fit_transform(df['performance_category'])
y = df['performance_category']  # Target variable

# Split the dataset into training and testing sets
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42,class_weight='balanced')
# Train the model
dt_classifier.fit(X_train, y_train)
# Make predictions
y_pred = dt_classifier.predict(X_test)
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print('='*80)
acc =accuracy_score(y_test, y_pred)
print("Accuracy", acc)
print('='*80)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=y_encoder.classes_))
print('='*80)
train_acc = dt_classifier.score(X_train, y_train)
test_acc = dt_classifier.score(X_test, y_test)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)

'''
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.show()
'''

#Plotting the decision tree
'''plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=dt_classifier.classes_)
plt.show()'''


#####################################################################

# Decision tree with grid search

print('='*80)
print("Decision Tree with Grid Search")
print('='*80)

param_grid = {
    'max_depth': [3, 5, 7, 10,11],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1,2,4],
    'criterion': ['gini', 'entropy']
}

dt_classifier_gs = DecisionTreeClassifier(random_state=42,class_weight='balanced')

grid_search = GridSearchCV(estimator=dt_classifier_gs, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
grid_search.fit(X_train, y_train)

best_dt_classifier = grid_search.best_estimator_


print('='*80)
print("Best Parameters:", grid_search.best_params_)

print('Best CV score:', grid_search.best_score_)
y_pred_gs = best_dt_classifier.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gs))
print('='*80)
acc_gs =accuracy_score(y_test, y_pred_gs)
print("Accuracy", acc_gs)
print('='*80)
print("Classification Report:")
print(classification_report(y_test, y_pred_gs, target_names=y_encoder.classes_))
print('='*80)
train_acc_gs = best_dt_classifier.score(X_train, y_train)
test_acc_gs = best_dt_classifier.score(X_test, y_test)
print("Train accuracy:", train_acc_gs)
print("Test accuracy:", test_acc_gs)

print('='*80)
print("Feature Importances for Initial Model:")
print('='*80)
importance_df_initial = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Features:")
print(importance_df_initial.head(10))



print('='*80)
print("Feature Importances for Grid Search Best Model:")
print('='*80)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': best_dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(importance_df.head(10))


# AI related analysis

df_ai = df.filter(like='ai', axis=1)
print(df_ai.columns)
df_ai['performance_category'] = df['performance_category']
print('='*80)
print("AI-related Features:")
print(df_ai)
print('='*80)

# Plots which AI tools used my students in each performance category
'''sns.countplot(data=df_ai, x = 'performance_category', hue='ai_tools_used')
plt.title('Performance Category by AI Usage')
plt.show()'''

# Boxplot to show the relation between ai_generated content and performance category
'''sns.boxplot(data=df_ai, x='ai_generated_content_percentage', y='performance_category')
plt.title('AI Generated Content vs Performance Category')
plt.show()'''






