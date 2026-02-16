# ==============================
# BUSINESS ANALYTICS INTERNSHIP — ANALYSIS SCRIPT
# ==============================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# SETTINGS
# ------------------------------
plt.style.use("dark_background")
sns.set_context("talk")

OUTPUT_DIR = "outputs/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "Downloads/dataset.csv"   # change if needed

#df = pd.read_csv("Downloads/Data_set 2 - Copy.csv")
# ------------------------------
# LOAD DATA
# ------------------------------
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

print("\nPreview:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe(include="all"))

# ------------------------------
# COLUMN TYPES
# ------------------------------
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

print("\nNumeric Columns:", list(num_cols))
print("Categorical Columns:", list(cat_cols))

# ==============================
# TASK 2 — GENDER DISTRIBUTION
# ==============================
if "Gender" in df.columns:
    plt.figure(figsize=(6,5))
    sns.countplot(x="Gender", data=df, palette="bright")
    plt.title("Gender Distribution")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gender_bar.png")
    plt.show()

    df["Gender"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%"
    )
    plt.title("Gender Pie")
    plt.ylabel("")
    plt.savefig(f"{OUTPUT_DIR}/gender_pie.png")
    plt.show()

# ==============================
# TASK 3 — DESCRIPTIVE STATS
# ==============================
print("\nNumeric Statistics:")
for col in num_cols:
    print(f"\n{col}")
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Std:", df[col].std())

    plt.figure(figsize=(7,4))
    sns.histplot(df[col], kde=True, color="orange")
    plt.title(f"Distribution — {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hist_{col}.png")
    plt.show()

# ==============================
# TASK 4 — PREFERRED INVESTMENT
# ==============================
pref_col = next((c for c in df.columns if "invest" in c.lower()), None)

if pref_col:
    vc = df[pref_col].value_counts()
    colors = ["lime" if v == vc.max() else "red" for v in vc]

    vc.plot(kind="bar", color=colors, figsize=(8,5))
    plt.title("Most Preferred Investment")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/preferred_investment.png")
    plt.show()

# ==============================
# TASK 5 — REASONS
# ==============================
reason_col = next((c for c in df.columns if "reason" in c.lower()), None)

if reason_col:
    plt.figure(figsize=(8,5))
    sns.countplot(
        y=reason_col,
        data=df,
        order=df[reason_col].value_counts().index,
        palette="cool"
    )
    plt.title("Reasons for Investment")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/reasons.png")
    plt.show()

# ==============================
# TASK 6 — SAVINGS OBJECTIVES
# ==============================
save_col = next((c for c in df.columns if "sav" in c.lower()), None)

if save_col:
    plt.figure(figsize=(8,5))
    sns.countplot(
        y=save_col,
        data=df,
        order=df[save_col].value_counts().index,
        palette="viridis"
    )
    plt.title("Savings Objectives")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/savings.png")
    plt.show()

# ==============================
# TASK 7 — INFO SOURCES
# ==============================
info_col = next((c for c in df.columns
                 if "source" in c.lower() or "info" in c.lower()), None)

if info_col:
    plt.figure(figsize=(8,5))
    sns.countplot(
        y=info_col,
        data=df,
        order=df[info_col].value_counts().index,
        palette="magma"
    )
    plt.title("Information Sources")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sources.png")
    plt.show()

# ==============================
# TASK 8 — INVESTMENT DURATION FIX
# ==============================
dur_col = next((c for c in df.columns if "duration" in c.lower()), None)

if dur_col:
    duration_map = {
        "Less than 1 year": 0.5,
        "1-3 years": 2,
        "3-5 years": 4,
        "More than 5 years": 6
    }

    df["Duration_numeric"] = df[dur_col].map(duration_map)

    print("\nAverage Duration (years):",
          df["Duration_numeric"].mean())

    sns.boxplot(x=df["Duration_numeric"], color="cyan")
    plt.title("Investment Duration Spread")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/duration_box.png")
    plt.show()

    sns.countplot(y=df[dur_col], palette="bright")
    plt.title("Duration Categories")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/duration_counts.png")
    plt.show()

# ==============================
# TASK 9 — EXPECTATIONS
# ==============================
exp_col = next((c for c in df.columns if "expect" in c.lower()), None)

if exp_col:
    sns.countplot(
        y=exp_col,
        data=df,
        order=df[exp_col].value_counts().index,
        palette="cubehelix"
    )
    plt.title("Investment Expectations")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/expectations.png")
    plt.show()

# ==============================
# TASK 10 — CORRELATION
# ==============================
if len(num_cols) >= 2:
    plt.figure(figsize=(10,8))
    corr = df[num_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        linewidths=1,
        linecolor="black"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.show()

# Scatter high/low coloring
if len(num_cols) >= 2:
    x = num_cols[0]
    y = num_cols[1]

    colors = ["red" if v < df[y].mean() else "lime" for v in df[y]]

    plt.scatter(df[x], df[y], c=colors)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Scatter High vs Low")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_high_low.png")
    plt.show()

print("\n✅ Analysis Complete — Charts saved in outputs/charts/")
