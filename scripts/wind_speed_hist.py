#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

paths = [
    'data/raw/wind_turbine_detection.csv',
    'wind_turbine_detection.csv',
    'data/wind_turbine_detection.csv'
]
df = None
for p in paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        src = p
        break

if df is None:
    print('ERROR: dataset CSV not found in expected paths:', paths)
    sys.exit(2)

col = 'wind_speed_mps'
if col not in df.columns:
    print('ERROR: column', col, 'not found. Columns:', list(df.columns))
    sys.exit(3)

series = df[col].dropna()
count = series.shape[0]
mean = series.mean()
median = series.median()
std = series.std()
q1 = series.quantile(0.25)
q3 = series.quantile(0.75)
minv = series.min()
maxv = series.max()

outdir = 'outputs'
os.makedirs(outdir, exist_ok=True)

plt.figure(figsize=(8,4))
sns.histplot(series, bins=60, kde=True, color='C0')
plt.title('Histogram of wind_speed_mps')
plt.xlabel('wind_speed_mps (m/s)')
plt.ylabel('Count')
plt.tight_layout()
outpath = os.path.join(outdir, 'wind_speed_hist.png')
plt.savefig(outpath)

print('Data source:', src)
print('Observations for', col)
print('Count (non-null):', count)
print(f'Mean: {mean:.3f}, Median: {median:.3f}, Std: {std:.3f}')
print(f'Q1: {q1:.3f}, Q3: {q3:.3f}, IQR: {q3-q1:.3f}')
print('Min:', minv, 'Max:', maxv)
print('\nSaved histogram to', outpath)
