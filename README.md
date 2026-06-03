# Data Analysis on the influence of AI on student academic performance

## Dataset Description
- The original data set contains 26 columns and aroung 5520 rows. Some columns will be deleted to reduce redundancy and since it won't aid in the analysis. 
- X is the part of the dataset where it contains the independent columns 
- y contains the target columns (Performance_category). This is a categoricla data, with 3 levels, 'Low', 'Medium', and 'High'.

## Null value handling
Null values were deleted. 

## Target Variable
Target variable is the "Performance Category". Which is divided into three tier list; low, medium and high.

## Encodings and Pipeline
 - Categorical columns are encoded using labelEncoder for the target and for X. However, we have used native encoding for X in XGBoost. Encoding mus tbe done after splitting of the data.
 - Pipelines where used for streamlined preprocessing. 


## NOTES
1. The macro F1-score was used as the primary evaluation metric due to class imbalance across performance categories. Accuracy alone can be misleading in such settings, whereas F1-score balances precision and recall, ensuring fair evaluation across all classes—particularly for the Low-performance category, which is the most critical from an intervention perspective.


## Objective
- To predict the performance category of student using columns which includes tradition studying methods and AI intergrated methods.
- Analysis is done using tree models - Decision trees, Random forest and XGBoost.

## Model - 1: Decision Treea

### Plain Decision Tree

#### Results
1. Classification Report 
   
              precision    recall  f1-score   support

        High       0.63      0.53      0.57       106
         Low       0.70      0.74      0.72       344
      Medium       0.79      0.78      0.78       654

    accuracy                           0.75      1104
   macro avg       0.71      0.68      0.69      1104
weighted avg       0.74      0.75      0.74      1104

2. Train Accuracy: 1.0 (Overfit)
   Test Accuracy: 0.74

3. Confusion Matrix
[[ 56   0  50]  --- High
 [  0 255  89]  --- Low
 [ 33 109 512]] --- Medium

### Decsion Tree with tuning

#### Results

1. Classification Report:
              precision    recall  f1-score   support

        High       0.56      0.85      0.67       106
         Low       0.70      0.85      0.77       344
      Medium       0.87      0.70      0.78       654

    accuracy                           0.76      1104
   macro avg       0.71      0.80      0.74      1104
weighted avg       0.79      0.76      0.76      1104

2. Train Accuracy: 0.83
   Test Accuracy: 0.76

3. Confusion Matrix
[[ 90   0  16]  ---High
 [  0 291  53]  ---Low 
 [ 71 123 460]] ---Medium

4. Best Parameters: {'classifier__criterion': 'gini', 'classifier__max_depth': 7, 'classifier__min_samples_leaf': 4, 'classifier__min_samples_split': 10}

5. Best Features (Top 5)
remainder__last_exam_score                    0.466771
remainder__assignment_scores_avg              0.255060
remainder__concept_understanding_score        0.190299
remainder__study_consistency_index            0.014127
remainder__improvement_rate                   0.010060

## Model - 2: Random Forest

### RF without tuning

#### Results

1. Classification Report:
              precision    recall  f1-score   support

        High       0.89      0.37      0.52       106
         Low       0.84      0.75      0.79       344
      Medium       0.80      0.92      0.85       654

    accuracy                           0.81      1104
   macro avg       0.84      0.68      0.72      1104
weighted avg       0.82      0.81      0.80      1104

2. Training score: 1.0
   Testing score: 0.81

3. Confusion Matrix:
[[ 39   0  67]  ---High
 [  0 258  86]  ---Low
 [  5  49 600]] ---Medium

4. Best Features (top 5)
remainder__last_exam_score                    0.292699
remainder__assignment_scores_avg              0.167220
remainder__concept_understanding_score        0.123215
remainder__ai_generated_content_percentage    0.036680
remainder__improvement_rate                   0.034343

### RF with tuning

#### Results

1. Classification Report (GridSearchCV):
              precision    recall  f1-score   support

        High       0.78      0.75      0.76       106
         Low       0.80      0.83      0.81       344
      Medium       0.87      0.85      0.86       654

    accuracy                           0.84      1104
   macro avg       0.81      0.81      0.81      1104
weighted avg       0.84      0.84      0.84      1104

2. Training score: 0.958
   Testing score: 0.811

3. Confusion Matrix
[[ 79   0  27]  ---High
 [  0 284  60]  ---Low
 [ 22  73 559]] ---Medium

4. Best Paramters: {'classifier__max_depth': 15, 'classifier__min_samples_leaf': 4, 'classifier__min_samples_split': 10, 'classifier__n_estimators': 200}

5. Best Features (top 5)
remainder__last_exam_score                    0.350754
remainder__assignment_scores_avg              0.192932
remainder__concept_understanding_score        0.145539
remainder__ai_generated_content_percentage    0.031857
remainder__improvement_rate                   0.026056

## Model - 3: XGBoost 

### XGBoost without tuning

#### Result

1. Classification Report:
              precision    recall  f1-score   support

        High       0.84      0.70      0.76       106
         Low       0.79      0.80      0.80       344
      Medium       0.85      0.87      0.86       654

    accuracy                           0.83      1104
   macro avg       0.83      0.79      0.81      1104
weighted avg       0.83      0.83      0.83      1104

2. Training score: 1.0
   Testing score: 0.83

3. Confusion Matrix:
[[ 74   0  32] --- High
 [  0 275  69] --- Low
 [ 14  71 569]] -- Medium

4. Best Features:
remainder__last_exam_score                    0.180338
remainder__concept_understanding_score        0.152360
remainder__assignment_scores_avg              0.107527
cat__ai_usage_purpose_Coding                  0.036577
remainder__ai_generated_content_percentage    0.036065

### XGBoost with tuning

#### Result 

1. Classification Report after Hyperparameter Tuning:
              precision    recall  f1-score   support

        High       0.80      0.74      0.77       106
         Low       0.82      0.81      0.82       344
      Medium       0.86      0.88      0.87       654

    accuracy                           0.84      1104
   macro avg       0.83      0.81      0.82      1104
weighted avg       0.84      0.84      0.84      1104

2. Training score: 0.89
   Testing score: 0.817

3. Confusion Matrix after Hyperparameter Tuning:
[[ 78   0  28]  --- High
 [  0 279  65]  --- Low 
 [ 19  61 574]] --- Medium

4. Best parameter: {'classifier__colsample_bytree': 1.0, 'classifier__learning_rate': 0.1, 'classifier__max_depth': 3, 'classifier__n_estimators': 200, 'classifier__subsample': 0.8}

5. Best Features (top 5)
remainder__last_exam_score                    0.190118
remainder__concept_understanding_score        0.130726
remainder__assignment_scores_avg              0.111046
remainder__ai_generated_content_percentage    0.037672
cat__ai_tools_used_ChatGPT                    0.030836




## OBSERVATIONS
1. Model comparision metric used is the f1-score. XGBoost performed the BEST, followed by Random Forest and then Decision trees. Then tuned model of all three trees performed better than the untuned models. 

2. Predicting MEDIUM category is more effective than other LOW and HIGH, where LOW predicitve capabality is better than HIGH.

3. Predicting HIGH and MEDIUM to so important as predicting the LOW category becuase focusing on the LOW helps with decreasing the failure rate and help students.

4. From the best features section of each of the 6 models, tradition learning methods trump over AI tools in learning process. 

5. 

