"""Central values for the food-grocery simulation and experiment design.

Edit this file before starting a run. Set P_INTEGRATION to a value in [0, 1]
to use it as the CLI default; leave it as None to require an explicit CLI value.
The hourly demand shape is read from DEFAULT_DEMAND_CSV, not defined here.
"""

from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEMAND_CSV = ROOT / "dataset" / "food_hourly_demand_profile.csv"

# Environment, population, and time (one tick is one minute).
GRID_WIDTH = 50
GRID_HEIGHT = 50
# Reference scenario: 300 daily food orders; fleet and restaurant counts are
# provisional ratios scaled from Meituan order, courier, and merchant records.
DAILY_CUSTOMERS = 300
NUM_DRIVERS = 12
NUM_RESTAURANTS = 50
# No grocery-store count is available from the food-delivery records.
NUM_STORES = 15
HORIZON = 1440
TICKS_PER_HOUR = 60
ASSIGNMENT_TIMEOUT = 15
HANDOVER_TIMEOUT = 30
DEFAULT_SEED = 42
# Reference integration probability: average obtained from the primary survey.
P_INTEGRATION: float | None = 0.44

# Food: Normal(mean, standard deviation) fitted to positive values of
# (estimate_meal_prepare_time - order_push_time) / 60 in the Meituan dataset.
# This is an estimated meal-ready offset, used as a proxy for preparation time,
# not an observed cooking duration. The engine clamps negatives to zero and
# rounds up to whole-minute ticks. Fit: output/meituan_prep_analysis/summary.json.
FOOD_PREPARATION = (12.902867262801337, 4.008371882051349)

# Other Normal(mean, standard deviation) draws retain their prior assumptions.
# Negative draws are clamped to zero and rounded up by the engine.
FOOD_ITEM_VALUE = (33760, 4418)
GROCERY_PREPARATION = (31, 19)
GROCERY_ITEM_VALUE = (42491, 29634)

# Distance, operating cost, emissions, and billing.
KM_PER_STEP = Decimal("0.5")
COST_PER_STEP = Decimal("200")
# Scenario assumption for a low-carbon motorcycle: 60 g CO2/km. Because one
# grid movement represents 0.5 km, each actual movement emits 30 g CO2.
EMISSION_PER_STEP = 30
# Generic Indonesian Zone I delivery-tariff proxy. KP 667/2022 publishes a
# Rp8,000--10,000 minimum and Rp2,000--2,500/km range for app-based motorcycle
# services; the model uses each midpoint rather than claiming a platform tariff.
DELIVERY_MINIMUM_FEE = Decimal("9000")
DELIVERY_FEE_PER_KM = Decimal("2250")
TAX_RATE = Decimal("0.10")
# Grab announced a maximum 8% Driver revenue share for the platform from
# 1 July 2026. The model applies that percentage uniformly to every order.
DRIVER_PLATFORM_SHARE = Decimal("0.08")
DRIVER_DELIVERY_FEE_SHARE = Decimal("1") - DRIVER_PLATFORM_SHARE

# Grab does not publish one universal GrabFood/GrabMart merchant commission;
# commercial terms can differ by merchant and delivery model. The simulation
# therefore uses one explicit baseline assumption for both Restaurants and
# Stores. This effective share preserves the prior merchant settlement level
# while expressing it as a single, generalizable proportion of gross merchant
# proceeds (item value plus modeled tax). It is not an official Grab tariff.
MERCHANT_PLATFORM_SHARE = Decimal("0.1781025272727272727272727273")

# Paired experiment defaults. These baseline values and sensitivity grids were
# confirmed for the final experiment design. For paired comparisons,
# P_INTEGRATION must be > 0.
DEFAULT_REPLICATIONS = 30
DEFAULT_PROBABILITY_VALUES = "0.22,0.44,0.66"
DEFAULT_STORE_COUNT_VALUES = "10,15,20"
DEFAULT_STORE_SELECTION_VALUES = "uniform,nearest_restaurant,nearest_customer"
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_QUANTILES = (0.025, 0.975)
BOOTSTRAP_SEED_BASE = 9_170_000
SIGNIFICANCE_ALPHA = 0.05
DEFAULT_VISUALIZER_INTERVAL_MS = 50
