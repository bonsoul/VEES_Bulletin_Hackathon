"""Create a dashboard of vaccination activity from the vf KABS export."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "kabs2.0_raw" / "vf_29102026.csv"
DEFAULT_OUTPUT = ROOT / "data" / "vf_29102026_visuals.png"
REQUIRED_COLUMNS = {
    "vaccination_date",
    "species",
    "disease",
    "total_number_vaccinated",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("vf_visuals")


def load_data(path: Path) -> pd.DataFrame:
    """Load the export and validate fields used by the dashboard."""
    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if df.empty:
        raise ValueError(f"No records found in {path}")

    df = df.copy()
    df["total_number_vaccinated"] = pd.to_numeric(
        df["total_number_vaccinated"], errors="coerce"
    )
    invalid_counts = df["total_number_vaccinated"].isna()
    if invalid_counts.any():
        rows = (df.index[invalid_counts] + 2).tolist()
        raise ValueError(
            "Invalid or missing vaccination totals on CSV rows: "
            + ", ".join(map(str, rows[:20]))
        )
    if (df["total_number_vaccinated"] < 0).any():
        raise ValueError("Vaccination totals must not be negative.")

    date_text = (
        df["vaccination_date"]
        .astype("string")
        .str.replace(r"\s+(?:EAT|East Africa Time)$", "", regex=True)
    )
    df["parsed_date"] = pd.to_datetime(date_text, errors="coerce")
    invalid_dates = int(df["parsed_date"].isna().sum())
    if invalid_dates:
        log.warning(
            "%d of %d dates could not be parsed and will be omitted from the time chart.",
            invalid_dates,
            len(df),
        )

    for column in ("county", "species", "disease"):
        if column in df:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .fillna("Unknown")
            )
    return df


def plot_top_categories(
    axis: plt.Axes, df: pd.DataFrame, column: str, title: str, limit: int = 10
) -> None:
    totals = (
        df.groupby(column, dropna=False)["total_number_vaccinated"]
        .sum()
        .nlargest(limit)
        .sort_values()
    )
    axis.barh(totals.index.astype(str), totals.values, color="#3182bd")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Animals vaccinated")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)


def create_dashboard(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("KABS Vaccination Activity", fontsize=18, fontweight="bold")

    dated = df.dropna(subset=["parsed_date"])
    if dated.empty:
        axes[0, 0].text(
            0.5, 0.5, "No valid dates available", ha="center", va="center"
        )
        axes[0, 0].set_title("Monthly vaccination activity", loc="left")
        axes[0, 0].set_axis_off()
    else:
        monthly = (
            dated.set_index("parsed_date")["total_number_vaccinated"]
            .resample("MS")
            .sum()
        )
        axes[0, 0].plot(
            monthly.index, monthly.values, marker="o", color="#238b45", linewidth=2
        )
        axes[0, 0].set_title(
            "Monthly vaccination activity", loc="left", fontweight="bold"
        )
        axes[0, 0].set_ylabel("Animals vaccinated")
        axes[0, 0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
        )
        axes[0, 0].grid(alpha=0.2)

    plot_top_categories(
        axes[0, 1], df, "county", "Top 10 counties by doses"
    )
    plot_top_categories(
        axes[1, 0], df, "species", "Doses by species"
    )
    plot_top_categories(
        axes[1, 1], df, "disease", "Top 10 diseases by doses"
    )

    total_doses = df["total_number_vaccinated"].sum()
    fig.text(
        0.99,
        0.015,
        f"{len(df):,} records  |  {total_doses:,.0f} reported doses",
        ha="right",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved dashboard to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    create_dashboard(load_data(args.input), args.output)


if __name__ == "__main__":
    main()
