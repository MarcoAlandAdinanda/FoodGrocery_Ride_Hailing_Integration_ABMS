"""Paired full-factorial experiment of integration mechanisms.

Example:
    python -B src/simulation_experiments.py --p-integrated 0.5 --replications 30 --sensitivity
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy import stats

if __package__:
    from .simulation_engine import (DAILY_CUSTOMERS, DEFAULT_DEMAND_CSV, NUM_DRIVERS, NUM_STORES, ROOT,
                                    IntegratedDeliveryModel, SimulationConfig, build_experiment_scenario)
    from .simulation_parameters import (
        DEFAULT_REPLICATIONS, DEFAULT_SEED, P_INTEGRATION, DEFAULT_PROBABILITY_VALUES,
        DEFAULT_STORE_COUNT_VALUES, DEFAULT_STORE_SELECTION_VALUES, BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_QUANTILES, BOOTSTRAP_SEED_BASE, DEFAULT_VISUALIZER_INTERVAL_MS,
        SIGNIFICANCE_ALPHA,
    )
else:
    from simulation_engine import (DAILY_CUSTOMERS, DEFAULT_DEMAND_CSV, NUM_DRIVERS, NUM_STORES, ROOT,
                                   IntegratedDeliveryModel, SimulationConfig, build_experiment_scenario)
    from simulation_parameters import (
        DEFAULT_REPLICATIONS, DEFAULT_SEED, P_INTEGRATION, DEFAULT_PROBABILITY_VALUES,
        DEFAULT_STORE_COUNT_VALUES, DEFAULT_STORE_SELECTION_VALUES, BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_QUANTILES, BOOTSTRAP_SEED_BASE, DEFAULT_VISUALIZER_INTERVAL_MS,
        SIGNIFICANCE_ALPHA,
    )


METRICS = (
    "service_units_per_driver_hour", "service_units_per_busy_driver_hour", "service_units_per_km",
    "food_completion_rate", "food_cancellation_rate", "food_unfinished_rate",
    "mean_food_completion_ticks", "mean_food_pickup_wait_ticks",
    "mean_food_preparation_queue_ticks",
    "mean_food_ready_to_handover_ticks",
    "mean_food_post_pickup_delivery_ticks",
    "platform_revenue_per_order", "driver_revenue_per_order",
    "merchant_revenue_per_order", "driver_profit_per_order",
    "emission_intensity_per_service_unit", "total_emission_units",
    "emission_units_per_order", "emission_units_per_completed_order",
    "driver_utilization", "driver_profit_per_driver",
)
PRIMARY_METRICS = (
    "service_units_per_driver_hour", "food_completion_rate", "mean_food_completion_ticks",
    "platform_revenue_per_order", "emission_intensity_per_service_unit",
)
EMISSION_PAIR_METRICS = (
    "marginal_emission_per_completed_grocery",
    "marginal_emission_per_additional_service_unit",
    "service_output_change_pct", "total_emission_change_pct", "emission_output_elasticity",
)
DRIVER_PAIR_METRICS = (
    "driver_food_pickup_wait_saved_ticks", "driver_food_pickup_wait_saved_pct",
)
INTEGRATED_EXPERIENCE_METRICS = (
    "integrated_completion_rate", "mean_integrated_completion_ticks",
    "integrated_cancellation_rate", "integrated_unfinished_rate",
    "mean_integrated_food_pickup_wait_ticks",
    "mean_integrated_grocery_pickup_wait_ticks",
    "mean_integrated_grocery_preparation_queue_ticks",
    "mean_integrated_grocery_ready_to_handover_ticks",
)

# KPI ownership is organized by the subject whose outcome or resource is measured.
# Research questions consume metrics across these subjects but do not define the taxonomy.
KPI_SUBJECTS = {
    "driver": (
        "service_units_per_driver_hour", "service_units_per_busy_driver_hour",
        "service_units_per_km", "mean_food_pickup_wait_ticks",
        "mean_integrated_food_pickup_wait_ticks", "mean_integrated_grocery_pickup_wait_ticks",
        "driver_food_pickup_wait_saved_ticks", "driver_food_pickup_wait_saved_pct",
        "driver_revenue_per_order",
        "driver_profit_per_order", "driver_utilization", "driver_profit_per_driver",
    ),
    "customer": (
        "food_completion_rate", "food_cancellation_rate", "food_unfinished_rate",
        "mean_food_completion_ticks", "mean_food_post_pickup_delivery_ticks",
        "integrated_completion_rate", "mean_integrated_completion_ticks",
        "integrated_cancellation_rate", "integrated_unfinished_rate",
    ),
    "merchant": (
        "merchant_revenue_per_order", "mean_food_preparation_queue_ticks",
        "mean_integrated_grocery_preparation_queue_ticks", "mean_food_ready_to_handover_ticks",
        "mean_integrated_grocery_ready_to_handover_ticks",
    ),
    "platform": ("platform_revenue_per_order",),
    "environment": (
        "emission_intensity_per_service_unit", "total_emission_units",
        "emission_units_per_order", "emission_units_per_completed_order",
        "marginal_emission_per_completed_grocery",
        "marginal_emission_per_additional_service_unit", "total_emission_change_pct",
        "emission_output_elasticity",
    ),
    "system": ("service_output_change_pct",),
}
KPI_SUBJECT = {
    metric: subject
    for subject, metrics in KPI_SUBJECTS.items()
    for metric in metrics
}


def _kpi_subject(metric: str) -> str:
    """Return the canonical subject owner for an experiment KPI."""
    try:
        return KPI_SUBJECT[metric]
    except KeyError as exc:
        raise ValueError(f"KPI subject is not registered: {metric}") from exc


class _ImmediateFileHandler(logging.Handler):
    """Append one formatted record at a time without retaining a Windows file lock."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def _build_run_logger(output_dir: Path) -> logging.Logger:
    """Create an experiment-local, immediately flushed audit log."""
    logger = logging.getLogger(f"simulation_experiment.{output_dir.name}.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _ImmediateFileHandler(output_dir / "experiment.log")
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = __import__("time").gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _metric_values(summary: dict, num_drivers: int) -> dict[str, float | None]:
    generated = summary["generated_orders"]
    completed = summary["completed_orders"]
    if generated <= 0:
        raise ValueError("A complete experiment run must generate orders.")
    return {
        "service_units_per_driver_hour": summary["service_units_per_driver_hour"],
        "service_units_per_busy_driver_hour": summary["service_units_per_busy_driver_hour"],
        "service_units_per_km": summary["service_units_per_km"],
        "food_completion_rate": summary["food_completion_rate"],
        "food_cancellation_rate": summary["food_cancellation_rate"],
        "food_unfinished_rate": summary["food_unfinished_rate"],
        "mean_food_completion_ticks": summary["mean_food_completion_ticks"],
        "mean_food_post_pickup_delivery_ticks": summary["mean_food_post_pickup_delivery_ticks"],
        **{metric: summary[metric] for metric in INTEGRATED_EXPERIENCE_METRICS},
        "mean_food_pickup_wait_ticks": summary["mean_food_pickup_wait_ticks"],
        "mean_food_preparation_queue_ticks": summary["mean_food_preparation_queue_ticks"],
        "mean_food_ready_to_handover_ticks": summary["mean_food_ready_to_handover_ticks"],
        "platform_revenue_per_order": float(Decimal(summary["platform_revenue"]) / generated),
        "driver_revenue_per_order": float(Decimal(summary["driver_revenue"]) / generated),
        "merchant_revenue_per_order": float(Decimal(summary["merchant_revenue"]) / generated),
        "driver_profit_per_order": float(Decimal(summary["driver_profit"]) / generated),
        "emission_intensity_per_service_unit": summary["emission_intensity_per_service_unit"],
        "total_emission_units": summary["emission_units"],
        "emission_units_per_order": summary["emission_units"] / generated,
        "emission_units_per_completed_order": (summary["emission_units"] / completed
                                                if completed else None),
        "driver_utilization": summary["driver_utilization"],
        "driver_profit_per_driver": float(Decimal(summary["driver_profit"]) / num_drivers),
    }


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_probabilities(args: argparse.Namespace) -> list[float]:
    if not math.isfinite(args.p_integrated) or not 0 <= args.p_integrated <= 1:
        raise ValueError("--p-integrated must be a finite number between 0 and 1.")
    probabilities = {args.p_integrated}
    if args.sensitivity:
        for token in args.p_values.split(","):
            try:
                value = float(token.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid sensitivity probability: {token!r}") from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Sensitivity probabilities must be finite values in [0,1].")
            if value > 0:
                probabilities.add(value)
    return sorted(probabilities)


def _read_store_selections(values: str) -> list[str]:
    result = {"uniform"}
    allowed = {"uniform", "nearest_restaurant", "nearest_customer"}
    for token in values.split(","):
        value = token.strip()
        if value not in allowed:
            raise ValueError(f"Invalid store selection: {value!r}")
        result.add(value)
    return [value for value in ("uniform", "nearest_restaurant", "nearest_customer") if value in result]


def _read_store_counts(values: str, reference: int) -> list[int]:
    result = {reference}
    for token in values.split(","):
        try:
            value = int(token.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid Store count: {token!r}") from exc
        if value <= 0:
            raise ValueError("Store counts must be positive integers.")
        result.add(value)
    return sorted(result)


def _conditions(args: argparse.Namespace) -> list[dict]:
    probabilities = _read_probabilities(args)
    store_values = _read_store_selections(args.store_selection_values) if args.sensitivity else ["uniform"]
    store_counts = (_read_store_counts(args.store_count_values, args.num_stores)
                    if args.sensitivity else [args.num_stores])
    conditions = [
        {"condition_id": "baseline", "factor": "baseline", "p_integration": 0.0,
         "food_preparation_multiplier": 1.0, "grocery_preparation_multiplier": 1.0,
         "num_stores": args.num_stores, "store_selection": "uniform"}
    ]
    for probability in probabilities:
        for num_stores in store_counts:
            for store_selection in store_values:
                conditions.append({
                    "condition_id": f"p_{probability:g}_stores_{num_stores}_{store_selection}",
                    "factor": "factorial_cell",
                    "p_integration": probability,
                    "food_preparation_multiplier": 1.0,
                    "grocery_preparation_multiplier": 1.0,
                    "num_stores": num_stores,
                    "store_selection": store_selection,
                })
    return conditions


def _bootstrap_interval(differences: list[float], seed: int) -> tuple[float | None, float | None]:
    """Percentile interval for the mean, resampling complete replication pairs."""
    if len(differences) < 2:
        return None, None
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, BOOTSTRAP_QUANTILES)
    return float(lower), float(upper)


def _paired_significance(differences: list[float]) -> dict[str, float | str | bool | None]:
    """Run two-sided inference on replication-level paired differences."""
    if len(differences) < 3:
        return {
            "normality_test": None, "normality_p_value": None,
            "significance_test": None, "test_statistic": None, "raw_p_value": None,
            "cohen_dz": None, "statistically_significant_raw": None,
        }
    values = np.asarray(differences, dtype=float)
    if np.all(values == 0):
        return {
            "normality_test": "not_applicable_all_zero", "normality_p_value": None,
            "significance_test": "all_zero_differences", "test_statistic": 0.0,
            "raw_p_value": 1.0, "cohen_dz": 0.0,
            "statistically_significant_raw": False,
        }
    normality = stats.shapiro(values)
    normality_p = float(normality.pvalue)
    if normality_p >= SIGNIFICANCE_ALPHA:
        result = stats.ttest_1samp(values, popmean=0.0, alternative="two-sided")
        test_name = "paired_t_test"
    else:
        result = stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox")
        test_name = "wilcoxon_signed_rank"
    standard_deviation = float(np.std(values, ddof=1))
    effect_size = (float(np.mean(values)) / standard_deviation
                   if standard_deviation > 0 else None)
    p_value = float(result.pvalue)
    return {
        "normality_test": "shapiro_wilk", "normality_p_value": normality_p,
        "significance_test": test_name, "test_statistic": float(result.statistic),
        "raw_p_value": p_value, "cohen_dz": effect_size,
        "statistically_significant_raw": p_value < SIGNIFICANCE_ALPHA,
    }


def _apply_holm_correction(rows: list[dict]) -> None:
    """Adjust primary-metric p-values within each integrated condition."""
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        row["holm_family"] = "primary_metrics_within_condition" if row["primary_metric"] else None
        row["adjusted_p_value_holm"] = None
        row["statistically_significant_holm"] = None
        if row["primary_metric"] and row["raw_p_value"] is not None:
            by_condition.setdefault(row["condition_id"], []).append(row)
    for family in by_condition.values():
        ordered = sorted(family, key=lambda row: row["raw_p_value"])
        running_max = 0.0
        family_size = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (family_size - rank) * row["raw_p_value"])
            running_max = max(running_max, adjusted)
            row["adjusted_p_value_holm"] = running_max
            row["statistically_significant_holm"] = running_max < SIGNIFICANCE_ALPHA


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(args: argparse.Namespace) -> Path:
    if args.replications < 2:
        raise ValueError("At least two replications are required for a paired interval.")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be nonnegative.")
    conditions = _conditions(args)
    demand_csv = Path(args.demand_csv).expanduser().resolve()
    if not demand_csv.is_file():
        raise ValueError(f"Demand CSV does not exist: {demand_csv}")
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
                  ROOT / "output" / "experiments" /
                  f"experiment_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    if args.visualize_replication is not None and not 1 <= args.visualize_replication <= args.replications:
        raise ValueError("--visualize-replication must identify one requested replication.")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios_dir = output_dir / "scenarios"
    runs_dir = output_dir / "runs"
    scenarios_dir.mkdir()
    runs_dir.mkdir()
    logger = _build_run_logger(output_dir)
    logger.info(
        "experiment_started output_dir=%s replications=%d conditions=%d model_runs=%d "
        "seed_start=%d daily_customers=%d num_drivers=%d p_integrated=%g sensitivity=%s "
        "summary_only=%s demand_csv=%s demand_sha256=%s",
        output_dir, args.replications, len(conditions), args.replications * len(conditions),
        args.seed_start, args.daily_customers, args.num_drivers, args.p_integrated,
        args.sensitivity, args.summary_only, demand_csv, _file_sha256(demand_csv),
    )

    raw_rows: list[dict] = []
    pair_rows: list[dict] = []
    scenario_records: list[dict] = []
    visualize_scenario = None
    for replication in range(1, args.replications + 1):
        seed = args.seed_start + replication - 1
        logger.info("replication_started replication=%d seed=%d", replication, seed)
        try:
            scenario = build_experiment_scenario(
                seed, demand_csv, args.daily_customers, args.num_drivers,
                max(condition["num_stores"] for condition in conditions),
            )
        except Exception:
            logger.exception("scenario_failed replication=%d seed=%d", replication, seed)
            raise
        scenario_name = f"rep_{replication:03d}"
        scenario_path = scenarios_dir / f"{scenario_name}.json"
        scenario_path.write_text(json.dumps(asdict(scenario), indent=2) + "\n", encoding="utf-8")
        scenario_records.append({"replication": replication, "seed": seed,
                                 "sha256": scenario.sha256,
                                 "path": str(scenario_path.relative_to(output_dir))})
        if replication == args.visualize_replication:
            visualize_scenario = scenario
        by_condition = {}
        for condition in conditions:
            logger.info(
                "condition_started replication=%d seed=%d condition_id=%s factor=%s "
                "p_integration=%g food_multiplier=%g grocery_multiplier=%g store_selection=%s",
                replication, seed, condition["condition_id"], condition["factor"],
                condition["p_integration"], condition["food_preparation_multiplier"],
                condition["grocery_preparation_multiplier"], condition["store_selection"],
            )
            config = SimulationConfig(
                p_integration=condition["p_integration"], seed=seed, demand_csv=demand_csv,
                daily_customers=args.daily_customers, num_drivers=args.num_drivers,
                num_stores=condition["num_stores"],
                food_preparation_multiplier=condition["food_preparation_multiplier"],
                grocery_preparation_multiplier=condition["grocery_preparation_multiplier"],
                store_selection=condition["store_selection"],
            )
            try:
                model = IntegratedDeliveryModel(config, scenario=scenario)
                summary = model.run()
                if args.summary_only:
                    run_path = None
                else:
                    run_path = model.export_results(runs_dir / scenario_name / condition["condition_id"])
            except Exception:
                logger.exception(
                    "condition_failed replication=%d seed=%d condition_id=%s",
                    replication, seed, condition["condition_id"],
                )
                raise
            metrics = _metric_values(summary, args.num_drivers)
            row = {"replication": replication, "seed": seed, "scenario_sha256": scenario.sha256,
                   "daily_customers": args.daily_customers, "num_drivers": args.num_drivers, **condition,
                   "run_dir": str(run_path.relative_to(output_dir)) if run_path else "",
                   "generated_orders": summary["generated_orders"],
                   "integrated_orders": summary["integrated_orders"],
                   "completed_orders": summary["completed_orders"],
                   "completed_food_service_units": summary["completed_food_service_units"],
                   "completed_grocery_service_units": summary["completed_grocery_service_units"],
                   "completed_service_units": summary["completed_service_units"],
                   "cancelled_orders": summary["cancelled_orders"],
                   "active_orders": summary["active_orders"], **metrics}
            row["observed_completion_ticks_count"] = summary["observed_completion_total_ticks_count"]
            row["observed_food_pickup_wait_ticks_count"] = summary["observed_food_pickup_wait_ticks_count"]
            raw_rows.append(row)
            by_condition[condition["condition_id"]] = row
            logger.info(
                "condition_completed replication=%d seed=%d condition_id=%s generated=%d "
                "integrated=%d completed=%d cancelled=%d unfinished=%d service_units=%d "
                "productivity=%s emission_g_co2=%s emission_intensity=%s run_dir=%s",
                replication, seed, condition["condition_id"], summary["generated_orders"],
                summary["integrated_orders"], summary["completed_orders"],
                summary["cancelled_orders"], summary["active_orders"],
                summary["completed_service_units"], summary["service_units_per_driver_hour"],
                summary["emission_units"], summary["emission_intensity_per_service_unit"],
                str(run_path.relative_to(output_dir)) if run_path else "summary-only",
            )
        for condition in conditions:
            if condition["p_integration"] == 0:
                continue
            treatment = by_condition[condition["condition_id"]]
            baseline_id = "baseline"
            baseline = by_condition[baseline_id]
            pair_row = {"replication": replication, "seed": seed,
                        "scenario_sha256": scenario.sha256,
                        "daily_customers": args.daily_customers, "num_drivers": args.num_drivers,
                        **condition, "baseline_condition_id": baseline_id,
                        "primary_comparison": condition["condition_id"] ==
                        f"p_{args.p_integrated:g}_stores_{args.num_stores}_uniform"}
            for metric in METRICS:
                before, after = baseline[metric], treatment[metric]
                pair_row[f"baseline_{metric}"] = before
                pair_row[f"integrated_{metric}"] = after
                pair_row[f"delta_{metric}"] = (after - before
                                                if before is not None and after is not None else None)
            delta_emission = treatment["total_emission_units"] - baseline["total_emission_units"]
            completed_grocery = treatment["completed_grocery_service_units"]
            additional_service_units = (treatment["completed_service_units"] -
                                        baseline["completed_service_units"])
            service_change_pct = (additional_service_units / baseline["completed_service_units"] * 100
                                  if baseline["completed_service_units"] else None)
            emission_change_pct = (delta_emission / baseline["total_emission_units"] * 100
                                   if baseline["total_emission_units"] else None)
            pair_row.update({
                "driver_food_pickup_wait_saved_ticks": (
                    baseline["mean_food_pickup_wait_ticks"] -
                    treatment["mean_food_pickup_wait_ticks"]
                    if baseline["mean_food_pickup_wait_ticks"] is not None and
                    treatment["mean_food_pickup_wait_ticks"] is not None else None),
                "driver_food_pickup_wait_saved_pct": (
                    (baseline["mean_food_pickup_wait_ticks"] -
                     treatment["mean_food_pickup_wait_ticks"]) /
                    baseline["mean_food_pickup_wait_ticks"] * 100
                    if baseline["mean_food_pickup_wait_ticks"] not in (None, 0) and
                    treatment["mean_food_pickup_wait_ticks"] is not None else None),
                "additional_service_units": additional_service_units,
                "marginal_emission_per_completed_grocery": (
                    delta_emission / completed_grocery if completed_grocery else None),
                "marginal_emission_per_additional_service_unit": (
                    delta_emission / additional_service_units if additional_service_units > 0 else None),
                "service_output_change_pct": service_change_pct,
                "total_emission_change_pct": emission_change_pct,
                "emission_output_elasticity": (
                    emission_change_pct / service_change_pct
                    if emission_change_pct is not None and service_change_pct not in (None, 0) else None),
            })
            pair_rows.append(pair_row)
        print(f"Replication {replication}/{args.replications}: seed {seed}, "
              f"p=0 completed {by_condition['baseline']['completed_orders']}, "
              f"reference completed "
              f"{by_condition[f'p_{args.p_integrated:g}_stores_{args.num_stores}_uniform']['completed_orders']}")
        logger.info(
            "replication_completed replication=%d seed=%d baseline_completed=%d "
            "reference_integrated_completed=%d",
            replication, seed, by_condition["baseline"]["completed_orders"],
            by_condition[f"p_{args.p_integrated:g}_stores_{args.num_stores}_uniform"]["completed_orders"],
        )

    raw_columns = list(raw_rows[0])
    pair_columns = list(pair_rows[0])
    _write_csv(output_dir / "run_summaries.csv", raw_rows, raw_columns)
    _write_csv(output_dir / "paired_differences.csv", pair_rows, pair_columns)
    statistical_rows = []
    for condition in conditions:
        if condition["p_integration"] == 0:
            continue
        condition_pairs = [row for row in pair_rows if row["condition_id"] == condition["condition_id"]]
        for metric_index, metric in enumerate(METRICS):
            differences = [row[f"delta_{metric}"] for row in condition_pairs
                           if row[f"delta_{metric}"] is not None]
            lower, upper = _bootstrap_interval(differences, BOOTSTRAP_SEED_BASE + metric_index +
                                               sum(ord(ch) for ch in condition["condition_id"]))
            statistical_rows.append({
                "daily_customers": args.daily_customers, "num_drivers": args.num_drivers,
                **condition, "primary_comparison": condition["condition_id"] ==
                f"p_{args.p_integrated:g}_stores_{args.num_stores}_uniform",
                "metric": metric,
                "subject": _kpi_subject(metric),
                "primary_metric": metric in PRIMARY_METRICS,
                "observed_pairs": len(differences),
                "mean_delta": float(np.mean(differences)) if differences else None,
                "ci95_lower": lower, "ci95_upper": upper,
                **_paired_significance(differences),
            })
    _apply_holm_correction(statistical_rows)
    _write_csv(output_dir / "paired_statistics.csv", statistical_rows, list(statistical_rows[0]))

    emission_statistical_rows = []
    for condition in conditions:
        if condition["p_integration"] == 0:
            continue
        condition_pairs = [row for row in pair_rows if row["condition_id"] == condition["condition_id"]]
        for metric_index, metric in enumerate(EMISSION_PAIR_METRICS):
            values = [row[metric] for row in condition_pairs if row[metric] is not None]
            lower, upper = _bootstrap_interval(values, BOOTSTRAP_SEED_BASE + 200_000 + metric_index +
                                               sum(ord(ch) for ch in condition["condition_id"]))
            emission_statistical_rows.append({
                "daily_customers": args.daily_customers, "num_drivers": args.num_drivers,
                **condition, "primary_comparison": condition["condition_id"] ==
                f"p_{args.p_integrated:g}_stores_{args.num_stores}_uniform",
                "metric": metric, "subject": _kpi_subject(metric), "observed_pairs": len(values),
                "mean_value": float(np.mean(values)) if values else None,
                "ci95_lower": lower, "ci95_upper": upper,
            })
    _write_csv(output_dir / "emission_statistics.csv", emission_statistical_rows,
               list(emission_statistical_rows[0]))

    driver_statistical_rows = []
    for condition in conditions:
        if condition["p_integration"] == 0:
            continue
        condition_pairs = [row for row in pair_rows if row["condition_id"] == condition["condition_id"]]
        for metric_index, metric in enumerate(DRIVER_PAIR_METRICS):
            values = [row[metric] for row in condition_pairs if row[metric] is not None]
            lower, upper = _bootstrap_interval(values, BOOTSTRAP_SEED_BASE + 250_000 + metric_index +
                                               sum(ord(ch) for ch in condition["condition_id"]))
            driver_statistical_rows.append({
                "daily_customers": args.daily_customers, "num_drivers": args.num_drivers,
                **condition, "primary_comparison": condition["condition_id"] ==
                f"p_{args.p_integrated:g}_stores_{args.num_stores}_uniform",
                "metric": metric, "subject": _kpi_subject(metric),
                "primary_driver_metric": metric == "driver_food_pickup_wait_saved_ticks",
                "observed_pairs": len(values),
                "mean_value": float(np.mean(values)) if values else None,
                "ci95_lower": lower, "ci95_upper": upper,
                **_paired_significance(values),
            })
    _write_csv(output_dir / "driver_statistics.csv", driver_statistical_rows,
               list(driver_statistical_rows[0]))

    service_level_rows = []
    for condition in conditions:
        if condition["p_integration"] == 0:
            continue
        condition_runs = [row for row in raw_rows if row["condition_id"] == condition["condition_id"]]
        for metric_index, metric in enumerate(INTEGRATED_EXPERIENCE_METRICS):
            values = [row[metric] for row in condition_runs if row[metric] is not None]
            lower, upper = _bootstrap_interval(values, BOOTSTRAP_SEED_BASE + 300_000 + metric_index +
                                               sum(ord(ch) for ch in condition["condition_id"]))
            service_level_rows.append({
                "daily_customers": args.daily_customers, "num_drivers": args.num_drivers,
                **condition, "primary_comparison": condition["condition_id"] ==
                f"p_{args.p_integrated:g}_stores_{args.num_stores}_uniform",
                "metric": metric,
                "subject": _kpi_subject(metric),
                "primary_integrated_experience_metric": metric in (
                    "integrated_completion_rate", "mean_integrated_completion_ticks"),
                "observed_replications": len(values),
                "mean_value": float(np.mean(values)) if values else None,
                "ci95_lower": lower, "ci95_upper": upper,
            })
    _write_csv(output_dir / "service_level_statistics.csv", service_level_rows,
               list(service_level_rows[0]))

    reference_id = f"p_{args.p_integrated:g}_stores_{args.num_stores}_uniform"
    pairs_by_key = {(row["replication"], row["condition_id"]): row for row in pair_rows}
    sensitivity_rows = []
    for condition in conditions:
        if condition["p_integration"] == 0 or condition["condition_id"] == reference_id:
            continue
        for metric_index, metric in enumerate(METRICS):
            changes = []
            for replication in range(1, args.replications + 1):
                ref_key = (replication, reference_id)
                alt_key = (replication, condition["condition_id"])
                if ref_key not in pairs_by_key or alt_key not in pairs_by_key:
                    logger.warning("Missing pair for sensitivity: replication=%d ref=%s alt=%s",
                                   replication, reference_id, condition["condition_id"])
                    continue
                reference = pairs_by_key[ref_key].get(f"delta_{metric}")
                alternative = pairs_by_key[alt_key].get(f"delta_{metric}")
                if reference is not None and alternative is not None:
                    changes.append(alternative - reference)
            lower, upper = _bootstrap_interval(changes, BOOTSTRAP_SEED_BASE + 100_000 + metric_index +
                                               sum(ord(ch) for ch in condition["condition_id"]))
            sensitivity_rows.append({"condition_id": condition["condition_id"],
                                     "reference_condition_id": reference_id, "factor": condition["factor"],
                                     "p_integration": condition["p_integration"],
                                     "food_preparation_multiplier": condition["food_preparation_multiplier"],
                                     "grocery_preparation_multiplier": condition["grocery_preparation_multiplier"],
                                     "num_stores": condition["num_stores"],
                                     "store_selection": condition["store_selection"],
                                     "metric": metric, "subject": _kpi_subject(metric),
                                     "primary_metric": metric in PRIMARY_METRICS,
                                     "observed_pairs": len(changes),
                                     "mean_change_in_delta": float(np.mean(changes)) if changes else None,
                                     "ci95_lower": lower, "ci95_upper": upper})
    if sensitivity_rows:
        _write_csv(output_dir / "sensitivity_effects.csv", sensitivity_rows, list(sensitivity_rows[0]))

    manifest = {
        "experiment_schema_version": 17,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_p": 0.0, "integrated_p": args.p_integrated,
        "probabilities": sorted({c["p_integration"] for c in conditions if c["p_integration"] > 0}),
        "store_selections": list(dict.fromkeys(
            c["store_selection"] for c in conditions if c["p_integration"] > 0
        )),
        "store_counts": sorted({c["num_stores"] for c in conditions if c["p_integration"] > 0}),
        "sensitivity": args.sensitivity, "conditions": conditions,
        "reference_daily_customers": args.daily_customers, "reference_num_drivers": args.num_drivers,
        "reference_num_stores": args.num_stores,
        "design": ("full factorial of positive integration probabilities, Store counts, and Store-selection policies, "
                   "plus one common p=0 food-only control; paired within replication"),
        "model_runs": len(raw_rows), "scenario_count": len(scenario_records),
        "scenario_generator": "factorial-v1; separate position streams for drivers, restaurants and stores",
        "grid_interpretation": ("Every positive integration-probability level is crossed with every "
                                "Store count and Store-selection policy; Store positions are nested prefixes; "
                                "preparation multipliers remain fixed at 1."),
        "replications": args.replications, "seed_start": args.seed_start,
        "demand_csv": str(demand_csv), "demand_sha256": _file_sha256(demand_csv),
        "engine_sha256": _file_sha256(Path(__file__).with_name("simulation_engine.py")),
        "controller_sha256": _file_sha256(Path(__file__)),
        "parameters_sha256": _file_sha256(Path(__file__).with_name("simulation_parameters.py")),
        "scenario_records": scenario_records,
        "experiment_log": "experiment.log",
        "primary_metrics": list(PRIMARY_METRICS),
        "kpi_taxonomy": "subject-centric; research questions select metrics across subjects",
        "kpi_subjects": {subject: list(metrics) for subject, metrics in KPI_SUBJECTS.items()},
        "emission_pair_metrics": list(EMISSION_PAIR_METRICS),
        "driver_pair_metrics": list(DRIVER_PAIR_METRICS),
        "integrated_experience_metrics": list(INTEGRATED_EXPERIENCE_METRICS),
        "statistics": ("integrated minus matching p=0 within each condition and replication; "
                       f"{BOOTSTRAP_RESAMPLES} paired bootstrap resamples, "
                       f"pointwise percentile {100 * (BOOTSTRAP_QUANTILES[1] - BOOTSTRAP_QUANTILES[0]):g}% "
                       f"interval; two-sided paired significance at alpha={SIGNIFICANCE_ALPHA:g}; "
                       "Shapiro-Wilk selects paired t-test when normality is not rejected and Wilcoxon "
                       "signed-rank otherwise; Holm correction across primary metrics within condition"),
        "sensitivity_effects": ("alternative paired delta minus reference paired delta within replication; "
                                "pointwise paired bootstrap intervals"),
        "emission_statistics": ("paired total-emission change divided by completed grocery or positive net "
                                "additional service units; percentage changes and emission-output elasticity; "
                                "pointwise paired bootstrap intervals"),
        "driver_statistics": ("paired food pickup-wait saved time, defined as matching food-only baseline "
                              "minus integrated condition; positive values mean time saved; percentage is "
                              "undefined when baseline pickup wait is zero or unobserved"),
        "service_level_statistics": ("condition-level integrated-customer experience across independently "
                                     "seeded day blocks; pointwise bootstrap intervals; food-service "
                                     "protection remains paired against food-only"),
        "replication_unit": "independently seeded day block; conditions sharing a replication seed are dependent",
        "visualization_condition": "baseline or reference integration probability",
        "interpretation": ("Food-only is the baseline. One completed food component and one completed grocery "
                           "component are equal unweighted service units; integrated route is "
                           "store-restaurant-customer."),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "experiment_completed model_runs=%d scenarios=%d output_dir=%s",
        len(raw_rows), len(scenario_records), output_dir,
    )
    for handler in logger.handlers:
        handler.flush()

    if visualize_scenario is not None:
        if __package__:
            from .simulation_visualizer import GridVisualizer
        else:
            from simulation_visualizer import GridVisualizer
        probability = 0.0 if args.visualize_condition == "baseline" else args.p_integrated
        model = IntegratedDeliveryModel(
            SimulationConfig(probability, visualize_scenario.seed, demand_csv, args.daily_customers,
                             args.num_drivers, args.num_stores),
            scenario=visualize_scenario,
        )
        GridVisualizer(model, interval_ms=args.interval_ms,
                       output_dir=output_dir / "visualized_run").run()
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run paired ABM replications and a full-factorial sensitivity design.")
    parser.add_argument("--p-integrated", type=float, default=P_INTEGRATION,
                        required=P_INTEGRATION is None,
                        help="Primary integration probability; 0 < p <= 1. Required when P_INTEGRATION is None.")
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED)
    parser.add_argument("--demand-csv", type=Path, default=DEFAULT_DEMAND_CSV)
    parser.add_argument("--daily-customers", type=int, default=DAILY_CUSTOMERS,
                        help="Fixed daily food-order count for every condition.")
    parser.add_argument("--num-drivers", type=int, default=NUM_DRIVERS,
                        help="Fixed driver count for every condition.")
    parser.add_argument("--num-stores", type=int, default=NUM_STORES,
                        help="Reference Grocery Store count.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Cross integration-probability and grocery Store-selection levels as a full factorial.")
    parser.add_argument("--p-values", default=DEFAULT_PROBABILITY_VALUES,
                        help="Comma-separated positive sensitivity probabilities; reference p is always included.")
    parser.add_argument("--store-selection-values", default=DEFAULT_STORE_SELECTION_VALUES,
                        help="Comma-separated store selection rules; uniform is always included.")
    parser.add_argument("--store-count-values", default=DEFAULT_STORE_COUNT_VALUES,
                        help="Comma-separated positive Store counts; reference count is always included.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip detailed seven-file exports for each run.")
    parser.add_argument("--visualize-replication", type=int,
                        help="After the batch, display this one-based replication at the reference demand and fleet.")
    parser.add_argument("--visualize-condition", choices=("baseline", "integrated"),
                        default="integrated")
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_VISUALIZER_INTERVAL_MS)
    args = parser.parse_args(argv)
    try:
        destination = run_experiment(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Experiment results: {destination}")


if __name__ == "__main__":
    main()
