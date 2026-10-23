import json
import os
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_random_data(num_categories, num_series):
    categories = [f"Cat_{i+1}" for i in range(num_categories)]
    data = {"category": categories}
    for i in range(num_series):
        data[f"series_{i+1}"] = [random.randint(10, 100) for _ in range(num_categories)]
    df = pd.DataFrame(data)
    return df


def save_metadata(directory, prefix, chart_type, df, dpi):
    num_categories = len(df)          # Number of rows
    num_series = len(df.columns) - 1  # Exclude the "category" column
    num_values = num_categories * num_series

    # Complexity classification aligning with Module B3 spec
    if num_values <= 6:
        complexity = "simple"
    elif num_values <= 12:
        complexity = "medium"
    else:
        complexity = "complex"

    metadata = {
        "id": prefix,
        "image": f"{prefix}.png",
        "gold_table": f"{prefix}.csv",
        "chart_type": chart_type,
        "n_categories": num_categories,
        "n_series": num_series,
        "n_values": num_values,
        "dpi": dpi,
        "complexity": complexity,
        "complexity_level": complexity,
        "complexity_rule": "value_count_v1",
        "label_source": "generator",
    }

    json_path = os.path.join(directory, f"{prefix}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main():
    # Fix seed for reproducible sanity generation across machines
    random.seed(42)

    # Relative paths to project root
    project_root = Path(__file__).resolve().parent.parent
    base_dir = str(project_root / "data" / "sanity" / "generated")
    bar_dir = os.path.join(base_dir, "bar")
    grouped_dir = os.path.join(base_dir, "grouped")
    line_dir = os.path.join(base_dir, "line")

    # Image resolution
    dpi_setting = 300

    # Create directories if they don't exist
    for d in [bar_dir, grouped_dir, line_dir]:
        os.makedirs(d, exist_ok=True)

    # Generate 15 sample charts for each type (total 45 charts)
    num_samples = 15

    for idx in range(1, num_samples + 1):
        # 1. Bar Chart (1 series, 3-6 categories per Module B3 spec)
        num_cat = random.randint(3, 6)
        df_bar = generate_random_data(num_cat, 1)
        prefix = f"bar_{idx:02d}"
        df_bar.to_csv(os.path.join(bar_dir, f"{prefix}.csv"), index=False)

        plt.figure(figsize=(6, 4))
        plt.bar(df_bar["category"], df_bar["series_1"], color="skyblue")
        plt.title(f"Bar Chart Sample {idx}")
        plt.xlabel("Category")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(bar_dir, f"{prefix}.png"), dpi=dpi_setting)
        save_metadata(bar_dir, prefix, "bar", df_bar, dpi_setting)
        plt.close()

        # 2. Grouped Bar Chart (2-3 series, 3-6 categories per Module B3 spec)
        num_cat = random.randint(3, 6)
        num_series = random.randint(2, 3)
        df_grouped = generate_random_data(num_cat, num_series)
        prefix = f"grouped_{idx:02d}"
        df_grouped.to_csv(os.path.join(grouped_dir, f"{prefix}.csv"), index=False)

        series_cols = [c for c in df_grouped.columns if c != "category"]
        ax = df_grouped.plot(x="category", y=series_cols, kind="bar", figsize=(7, 4))
        plt.title(f"Grouped Bar Chart Sample {idx}")
        plt.xlabel("Category")
        plt.ylabel("Value")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(grouped_dir, f"{prefix}.png"), dpi=dpi_setting)
        save_metadata(grouped_dir, prefix, "grouped_bar", df_grouped, dpi_setting)
        plt.close()

        # 3. Line Chart (1-3 series, 3-6 categories per Module B3 spec)
        num_cat = random.randint(3, 6)
        num_series = random.randint(1, 3)
        df_line = generate_random_data(num_cat, num_series)
        prefix = f"line_{idx:02d}"
        df_line.to_csv(os.path.join(line_dir, f"{prefix}.csv"), index=False)

        plt.figure(figsize=(6, 4))
        for col in [c for c in df_line.columns if c != "category"]:
            plt.plot(df_line["category"], df_line[col], marker="o", label=col)
        plt.title(f"Line Chart Sample {idx}")
        plt.xlabel("Category")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(line_dir, f"{prefix}.png"), dpi=dpi_setting)
        save_metadata(line_dir, prefix, "line", df_line, dpi_setting)
        plt.close()

    print(f"Successfully generated {num_samples} samples per chart type in: {base_dir}")


if __name__ == "__main__":
    main()
