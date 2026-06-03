import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ai_impact_student_performance_dataset.csv')
print(df.info())
print('='*80)

print(pd.crosstab(df['uses_ai'], df['ai_tools_used'].isna(), rownames=['Uses AI'], colnames=['Tools Is Null']))
print('='*80)

# ==============================================================================
# 🔥 FIX 1: CALCULATE COHORT MODES OVER THE WHOLE INDEX TO PREVENT MISALIGNMENT
# ==============================================================================
mode_tools = df.groupby('performance_category')['ai_tools_used'].transform(
    lambda x: x.mode()[0] if not x.mode().empty else 'ChatGPT'
)
mode_purpose = df.groupby('performance_category')['ai_usage_purpose'].transform(
    lambda x: x.mode()[0] if not x.mode().empty else 'Exam Prep'
)

# 1. Fill the 900 active user omissions with their respective cohort modes first
df['ai_tools_used'] = df['ai_tools_used'].fillna(mode_tools)
df['ai_usage_purpose'] = df['ai_usage_purpose'].fillna(mode_purpose)

# 2. Fill the remaining 462 structural non-user nulls with an explicit string
df['ai_tools_used'] = df['ai_tools_used'].fillna('None').replace('', 'None')
df['ai_usage_purpose'] = df['ai_usage_purpose'].fillna('None').replace('', 'None')

# Check data consistency for edge-case survey contradictions
df.loc[(df['uses_ai'] == 0) & (df['ai_tools_used'] != 'None'), 'uses_ai'] = 1

print('Missing values after complete imputation loop:')
print(df[['ai_tools_used', 'ai_usage_purpose']].isna().sum())
print('='*80)

# Aggregation and Summaries
summ_category = df.groupby('performance_category')['final_score'].agg(['count', 'mean', 'min', 'max'])
print(summ_category)
print('='*80)

summ_passed = df.groupby('passed')['final_score'].agg(['count', 'min', 'max'])
print(summ_passed)
print('='*80)

# ==============================================================================
# 🔥 FIX 2: CONVERT STRING TYPES BEFORE EXPORTING FOR PANDAS 3.0 COMPATIBILITY
# ==============================================================================
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].astype('object')

# Save cleanly to disk
df.to_csv('cleaned_ai_impact_dataset.csv', index=False)
print("Successfully saved clean data to 'cleaned_ai_impact_dataset.csv'")
print('='*80)

# Verification Phase (Will output 8000 non-null values everywhere!)
df2 = pd.read_csv('cleaned_ai_impact_dataset.csv')
print("Verification of the written CSV file:")
print(df2.info())

print('End of AA.py')

# ==============================================================================
# VISUALIZATIONS (Executed at the end so calculations and exports are safe)
# ==============================================================================
plt.figure(figsize=(10, 5))
plt.hist(df['final_score'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Final Scores')
plt.xlabel('Final Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

df_num = df.select_dtypes(include=[np.number])
corr = df_num.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='performance_category', y='final_score')
plt.title('Final Scores by Performance Category')
plt.xlabel('Performance Category')
plt.ylabel('Final Score')
plt.show()