import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.data_loader import fetch_data


def run_eda(data_path: str = "data/raw/heart.csv") -> pd.DataFrame:
    """Load dataset, print statistical summary, and generate EDA visual plots."""
    os.makedirs("docs/screenshots", exist_ok=True)

    # 1. Fetch or Load Data
    if not os.path.exists(data_path):
        print(f"[INFO] Raw dataset not found at {data_path}. Fetching from UCI repository...")
        df = fetch_data(data_path)
    else:
        df = pd.read_csv(data_path)

    print("\n" + "=" * 50)
    print("        UCI HEART DISEASE DATASET OVERVIEW        ")
    print("=" * 50)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Data Types & Missing Values ---")
    info_df = pd.DataFrame(
        {
            "Data Type": df.dtypes,
            "Missing Values": df.isnull().sum(),
            "Null Percentage (%)": (df.isnull().sum() / len(df)) * 100,
        }
    )
    print(info_df)

    print("\n--- Summary Statistics (Numerical Features) ---")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])

    # Process target into binary if needed
    if "target" in df.columns:
        df["target_binary"] = (df["target"] > 0).astype(int)
        print("\n--- Target Distribution (0 = Absence, 1 = Presence) ---")
        print(df["target_binary"].value_counts(normalize=True).rename({0: "Absence (0)", 1: "Presence (1)"}) * 100)

    # 2. Generate Plots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Target Distribution
    sns.countplot(
        x="target_binary" if "target_binary" in df.columns else "target",
        data=df,
        ax=axes[0, 0],
        palette="Blues_d",
    )
    axes[0, 0].set_title("Heart Disease Class Balance", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Target (0 = No Disease, 1 = Heart Disease)")
    axes[0, 0].set_ylabel("Count")

    # Plot 2: Age Distribution by Disease Status
    target_col = "target_binary" if "target_binary" in df.columns else "target"
    sns.histplot(
        data=df,
        x="age",
        hue=target_col,
        kde=True,
        ax=axes[0, 1],
        palette="coolwarm",
        bins=15,
    )
    axes[0, 1].set_title("Age Distribution by Heart Disease Status", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Age (Years)")

    # Plot 3: Max Heart Rate (thalach) vs Age
    sns.scatterplot(
        data=df,
        x="age",
        y="thalach",
        hue=target_col,
        palette="coolwarm",
        ax=axes[1, 0],
        alpha=0.8,
    )
    axes[1, 0].set_title("Max Heart Rate (thalach) vs Age", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Age")
    axes[1, 0].set_ylabel("Max Heart Rate Achieved")

    # Plot 4: Feature Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=axes[1, 1], cbar=True, vmin=-1, vmax=1)
    axes[1, 1].set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")

    plt.tight_layout()
    output_plot_path = "docs/screenshots/eda_plots.png"
    plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] EDA plot figure generated and saved to: {output_plot_path}")
    return df


if __name__ == "__main__":
    run_eda()
