import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ai_impact_student_performance_dataset.csv')
print(df)
print('='*80)
print(df.info())
###################################################################################
print('='*80)
print(df.describe())
######################################################################
print('='*80)
print(df.isnull().sum())
print('='*80)
###############################################################################
print(pd.crosstab(df['uses_ai'], df['ai_tools_used'].isna(), rownames=['Uses AI'], colnames=['Tools Is Null']))

# From the crosstab, 900 students uses AI but have no tools used, which is a significant number. This could indicate that many students are using AI in some capacity.
# Following will use the mode to fill the missing values for the 900 students 

mode_tools = df[df['uses_ai'] == 1].groupby('performance_category')['ai_tools_used'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'ChatGPT+Gemini')
mode_purpose = df[df['uses_ai'] == 1].groupby('performance_category')['ai_usage_purpose'].transform(
    lambda x: x.mode()[0] if not x.mode().empty else 'Exam Prep'
)

# 1. Fill the 900 active user omissions with their respective cohort modes first
df['ai_tools_used'] = df['ai_tools_used'].fillna(mode_tools)
df['ai_usage_purpose'] = df['ai_usage_purpose'].fillna(mode_purpose)

# 2. Fill the remaining 462 structural non-user nulls with an explicit 'None' category
df['ai_tools_used'] = df['ai_tools_used'].fillna('None')
df['ai_usage_purpose'] = df['ai_usage_purpose'].fillna('None')


# Verify that your missing values are officially at 0
print(df[['ai_tools_used', 'ai_usage_purpose']].isna().sum())
print('='*80)
print('Below is the 2nd mssing value check')
print(df.isnull().sum())


df.loc[(df['uses_ai'] == 0) & (df['ai_tools_used'] != 'None'), 'uses_ai'] = 1
#########################################################################################
print('='*80)
print('Below is the 3rd mssing value check after corrections')
print(df.isnull().sum())
print('='*80)


summ_category =df.groupby('performance_category')['final_score'].agg(['count', 'mean', 'min', 'max'])
print(summ_category)
print('='*80)


summ_passed = df.groupby('passed')['final_score'].agg(['count', 'min', 'max'])
print(summ_passed)
print('='*80)


# Visualization of final scores distribution

plt.figure(figsize=(20, 10))
plt.hist(df['final_score'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Final Scores')
plt.xlabel('Final Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Scatter plot of study hours vs final score
plt.figure(figsize=(10, 6))
plt.scatter(df['study_hours_per_day'], df['final_score'], alpha=0.5)
plt.title('Study Hours vs Final Score')
plt.xlabel('Study Hours')
plt.ylabel('Final Score')
plt.grid(True)
plt.show()

# Heat map for all  the numerical features

df_num = df.select_dtypes(include=[np.number])
corr = df_num.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


# Bar plot of pass rates by AI usage
df.groupby('uses_ai')['passed'].mean().plot(kind='bar')
plt.ylabel('Pass rate')
plt.title('Pass rate by AI usage')
plt.show()


# Scatter plot for social media hours vs final score


plt.figure(figsize=(10, 6))
plt.scatter(df['social_media_hours'], df['final_score'], alpha=0.5, color='green')
plt.title('Social Media Hours vs Final Score')
plt.xlabel('Social Media Hours per Day')
plt.ylabel('Final Score')
plt.grid(True)
plt.show()


# Box plot of final scores by performance category

plt.figure(figsize=(10, 6))
sns.boxplot(df, x='performance_category', y='final_score')
plt.title('Final Scores by Performance Category')
plt.xlabel('Performance Category')
plt.ylabel('Final Score')
plt.show()

df.to_csv('cleaned_ai_impact_dataset.csv', index=False)
print("Successfully saved clean data to 'cleaned_ai_impact_dataset.csv'")

print('End of AA.py')

