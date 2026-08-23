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

col = 'failure'
if col not in df.columns:
    # try case-insensitive match
    for c in df.columns:
        if c.lower() == col:
            df[col] = df[c]
            break

if col not in df.columns:
    print('ERROR: target column "failure" not found. Columns:', list(df.columns))
    sys.exit(3)

counts = df[col].value_counts(dropna=False)
total = len(df)
percent = (counts / total * 100).round(2)

outdir = 'outputs'
os.makedirs(outdir, exist_ok=True)

print('Data source:', src)
print('Total rows:', total)
print('\nCounts:')
print(counts.to_string())
print('\nPercentages:')
print(percent.to_string())

plt.figure(figsize=(6,4))
uniq = df[col].dropna().unique()
if df[col].dtype == 'bool' or len(uniq) <= 10:
    ax = sns.countplot(x=col, data=df, palette='Set2')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')
else:
    ax = sns.histplot(df[col].dropna(), bins=30)

ax.set_title('Distribution of target: failure')
ax.set_xlabel('failure')
ax.set_ylabel('count')
plt.tight_layout()
outpath = os.path.join(outdir, 'failure_distribution.png')
plt.savefig(outpath)
print('\nSaved plot to', outpath)

summary_path = os.path.join(outdir, 'failure_counts.txt')
with open(summary_path, 'w') as f:
    f.write(f'Data source: {src}\n')
    f.write(f'Total rows: {total}\n\n')
    f.write('Counts:\n')
    f.write(counts.to_string())
    f.write('\n\nPercentages:\n')
    f.write(percent.to_string())

print('Saved counts summary to', summary_path)
