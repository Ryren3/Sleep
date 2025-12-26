import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ai_impact_student_performance_dataset.csv')
print(df)
print('='*80)
print(df.info())
print('='*80)
print(df.describe())
print('='*80)
print(df.isnull().sum())




