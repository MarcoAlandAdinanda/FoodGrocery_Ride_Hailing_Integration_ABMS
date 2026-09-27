"""Collect food demand distribution time and create hourly demand per day.

Reads the Meituan waybill dataset, computes hourly order demand using
``platform_order_time`` (when the customer placed the order), saves the
result as a CSV, and generates a line chart showing demand curves for
each day.

Usage::

    python src/food_demand_distribution.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "dataset" / "all_waybill_info_meituan_0322.csv"
OUT_CSV = ROOT / "dataset" / "hourly_demand_distribution.csv"
OUT_PNG = ROOT / "dataset" / "hourly_demand_distribution.png"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(f"Loading {DATA_CSV} ...")
df = pd.read_csv(DATA_CSV, usecols=["dt", "platform_order_time"])
print(f"  Rows loaded: {len(df):,}")

# ---------------------------------------------------------------------------
# 2. Convert Unix timestamps → datetime, extract date & hour
# ---------------------------------------------------------------------------
df["platform_order_time"] = (
    pd.to_datetime(df["platform_order_time"], unit="s", utc=True)
    .dt.tz_convert("Asia/Shanghai")
)
df["date"] = df["platform_order_time"].dt.date
df["day_name"] = df["platform_order_time"].dt.day_name()
df["hour"] = df["platform_order_time"].dt.hour

# ---------------------------------------------------------------------------
# 3. Per-date hourly demand (for CSV detail)
# ---------------------------------------------------------------------------
hourly = (
    df.groupby(["date", "day_name", "hour"])
    .size()
    .reset_index(name="demand_count")
    .sort_values(["date", "hour"])
)

# ---------------------------------------------------------------------------
# 4. Average hourly demand per day_name
# ---------------------------------------------------------------------------
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
avg_hourly = (
    hourly.groupby(["day_name", "hour"])["demand_count"]
    .mean()
    .reset_index(name="avg_demand")
)
avg_hourly["day_name"] = pd.Categorical(avg_hourly["day_name"], categories=day_order, ordered=True)
avg_hourly = avg_hourly.sort_values(["day_name", "hour"])

print(f"\nAverage hourly demand by day_name shape: {avg_hourly.shape}")
print(avg_hourly.head(15))

# ---------------------------------------------------------------------------
# 5. Save CSV
# ---------------------------------------------------------------------------
hourly.to_csv(OUT_CSV, index=False)
avg_csv = OUT_CSV.with_name("food_hourly_demand_profile.csv")
avg_hourly.to_csv(avg_csv, index=False)
print(f"\nSaved per-date hourly demand to {OUT_CSV}")
print(f"Saved average hourly demand to {avg_csv}")

# ---------------------------------------------------------------------------
# 6. Pivot for plotting (hours as rows, day_name as columns)
# ---------------------------------------------------------------------------
pivot = avg_hourly.pivot(index="hour", columns="day_name", values="avg_demand").fillna(0)
pivot = pivot[day_order]  # reorder columns Mon→Sun

# ---------------------------------------------------------------------------
# 7. Plot: 7 stacked bar charts (one per day), full width
# ---------------------------------------------------------------------------
# Shared Y-axis limit (global max across all days)
y_max = pivot.max().max()
y_limit = y_max * 1.12  # 12% headroom for top labels

# Colors – one per day
colors = [
    "#4C72B0",  # Monday    – muted blue
    "#DD8452",  # Tuesday   – orange
    "#55A868",  # Wednesday – green
    "#C44E52",  # Thursday  – red
    "#8172B3",  # Friday    – purple
    "#937860",  # Saturday  – brown
    "#DA8BC3",  # Sunday    – pink
]

fig, axes = plt.subplots(7, 1, figsize=(16, 28), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.42, top=0.955, bottom=0.04, left=0.07, right=0.97)

# Threshold for placing label inside vs outside the bar (fraction of y_max)
INSIDE_THRESHOLD = 0.40

for i, day in enumerate(day_order):
    ax = axes[i]
    values = pivot[day].values
    bars = ax.bar(range(24), values, color=colors[i], alpha=0.88, width=0.72,
                  edgecolor="white", linewidth=0.4)

    # --- Smart label placement ---
    for bar, val in zip(bars, values):
        if val <= 0:
            continue
        x_center = bar.get_x() + bar.get_width() / 2
        bar_top = bar.get_height()

        if bar_top / y_limit >= INSIDE_THRESHOLD:
            # Inside the bar – white text, near the top
            ax.text(
                x_center,
                bar_top - y_limit * 0.012,
                f"{int(val):,}",
                ha="center",
                va="top",
                fontsize=7.5,
                fontweight="bold",
                color="white",
                clip_on=False,
            )
        else:
            # Outside the bar – black text above
            ax.text(
                x_center,
                bar_top + y_limit * 0.008,
                f"{int(val):,}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
                clip_on=False,
            )

    # Subplot styling
    ax.set_title(day, fontsize=14, fontweight="bold", loc="left", pad=6)
    ax.set_ylim(0, y_limit)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Only bottom subplot gets x-axis label
axes[-1].set_xlabel("Hour of Day", fontsize=12, labelpad=8)
# Only leftmost subplots get y-axis label (shared via sharey)
axes[3].set_ylabel("Average Number of Orders", fontsize=12, labelpad=10)

fig.suptitle(
    "Average Hourly Food Demand by Day of Week\n(platform_order_time)",
    fontsize=18,
    fontweight="bold",
    y=0.99,
)

fig.savefig(OUT_PNG, dpi=150)
print(f"Saved plot to {OUT_PNG}")
plt.show()
