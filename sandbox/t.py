
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Excel file in the same folder as this Python script
folder = Path(__file__).parent
file_path = folder / "FIFA_Master_Dataset_X4_X6_Suraj.xlsx"

# Load master dataset
df = pd.read_excel(
    file_path,
    sheet_name="Master_104_Matches"
)

# Your assigned variables and target
x4 = "last5_avg_goals_conceded_diff"
x6 = "last5_clean_sheet_rate_diff"
target = "goal_difference"

# Check that columns exist
required = [x4, x6, target]
missing = [col for col in required if col not in df.columns]

if missing:
    print("Missing columns:", missing)
    print("Available columns:", df.columns.tolist())
    raise ValueError("Please check the Excel column names.")

# Convert values to numbers
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df[required].isna().sum())

print("\nSummary statistics:")
print(df[[x4, x6]].describe().round(3))

# Create folder for diagrams
output = folder / "Suraj_Diagrams"
output.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Diagram 1: Histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df[x4].dropna(), kde=True, ax=axes[0])
axes[0].set_title("X4 - Average Goals Conceded Difference")

sns.histplot(df[x6].dropna(), kde=True, ax=axes[1])
axes[1].set_title("X6 - Clean Sheet Rate Difference")

plt.tight_layout()
fig.savefig(output / "01_distributions.png", dpi=300)
plt.show()
plt.close(fig)

# Diagram 2: Boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(y=df[x4].dropna(), ax=axes[0])
axes[0].set_title("X4 - Boxplot")

sns.boxplot(y=df[x6].dropna(), ax=axes[1])
axes[1].set_title("X6 - Boxplot")

plt.tight_layout()
fig.savefig(output / "02_boxplots.png", dpi=300)
plt.show()
plt.close(fig)

# Diagram 3: Regression scatterplots
for variable, label, filename in [
    (x4, "Average Goals Conceded Difference", "03_X4_regression.png"),
    (x6, "Clean Sheet Rate Difference", "04_X6_regression.png")
]:
    plot_df = df[[variable, target]].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(plot_df) >= 2 and plot_df[variable].nunique() >= 2:
        sns.regplot(
            data=plot_df,
            x=variable,
            y=target,
            ax=ax,
            ci=None
        )
    else:
        ax.scatter(plot_df[variable], plot_df[target])

    ax.set_title(f"{label} vs Goal Difference")
    ax.set_xlabel(label)
    ax.set_ylabel("Team A Goals - Team B Goals")

    plt.tight_layout()
    fig.savefig(output / filename, dpi=300)
    plt.show()
    plt.close(fig)

# Correlation analysis
print("\nCorrelation matrix:")
print(df[[x4, x6, target]].corr().round(3))

print(f"\nDiagrams saved in: {output}")