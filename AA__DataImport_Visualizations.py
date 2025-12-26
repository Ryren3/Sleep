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

df_num = df.select_dtypes(include=[np.number])
corr = df_num.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


