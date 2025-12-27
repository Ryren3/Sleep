import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ai_impact_student_performance_dataset.csv')
print(df)
print('='*80)
#print(df.info())
print('='*80)
#print(df.describe())
print('='*80)
df = df.dropna()
print('='*80)
'''print(df.isnull().sum())
print('='*80)'''

'''
summ_category =df.groupby('performance_category')['final_score'].agg(['count', 'mean', 'min', 'max'])
print(summ_category)
print('='*80)

summ_passed = df.groupby('passed')['final_score'].agg(['count', 'min', 'max'])
print(summ_passed)
print('='*80)

'''
# Visualization of final scores distribution

'''plt.figure(figsize=(20, 10))
plt.hist(df['final_score'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Final Scores')
plt.xlabel('Final Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()'''

# Scatter plot of study hours vs final score
'''plt.figure(figsize=(10, 6))
plt.scatter(df['study_hours_per_day'], df['final_score'], alpha=0.5)
plt.title('Study Hours vs Final Score')
plt.xlabel('Study Hours')
plt.ylabel('Final Score')
plt.grid(True)
plt.show()'''

# Heat map for all  the numerical features

'''df_num = df.select_dtypes(include=[np.number])
corr = df_num.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
'''
# Bar plot of pass rates by AI usage
'''df.groupby('uses_ai')['passed'].mean().plot(kind='bar')
plt.ylabel('Pass rate')
plt.title('Pass rate by AI usage')
plt.show()
'''

# Scatter plot for social media hours vs final score
'''print(df.info())
plt.figure(figsize=(10, 6))
plt.scatter(df['social_media_hours'], df['final_score'], alpha=0.5, color='green')
plt.title('Social Media Hours vs Final Score')
plt.xlabel('Social Media Hours per Day')
plt.ylabel('Final Score')
plt.grid(True)
plt.show()'''


# Box plot of final scores by performance category
'''plt.figure(figsize=(10, 6))
sns.boxplot(df, x='performance_category', y='final_score')
plt.title('Final Scores by Performance Category')
plt.xlabel('Performance Category')
plt.ylabel('Final Score')
plt.show()'''

df2 = df.groupby('performance_categoder')['gender'].value_counts(normalize=True).unstack()
df2.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Performance Category Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.show()


print('End of AA.py')

