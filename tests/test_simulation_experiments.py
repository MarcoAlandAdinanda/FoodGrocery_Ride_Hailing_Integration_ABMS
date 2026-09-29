"""Behavioral checks for shared exogenous days and paired output."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation_engine import (IntegratedDeliveryModel, SimulationConfig,
                                   build_experiment_scenario)
from src.simulation_experiments import (DRIVER_PAIR_METRICS, EMISSION_PAIR_METRICS,
                                        INTEGRATED_EXPERIENCE_METRICS,
                                        KPI_SUBJECT, KPI_SUBJECTS, METRICS,
                                        _apply_holm_correction, _paired_significance,
                                        _read_store_counts, _read_store_selections, main)
from src.simulation_visualizer import GridVisualizer


class SharedScenarioTests(unittest.TestCase):
    def test_every_experiment_kpi_has_exactly_one_subject(self):
        expected = (set(METRICS) | set(EMISSION_PAIR_METRICS) |
                    set(DRIVER_PAIR_METRICS) | set(INTEGRATED_EXPERIENCE_METRICS))
        self.assertEqual(len(INTEGRATED_EXPERIENCE_METRICS),
                         len(set(INTEGRATED_EXPERIENCE_METRICS)))
        registered = [metric for metrics in KPI_SUBJECTS.values() for metric in metrics]
        self.assertEqual(set(registered), expected)
        self.assertEqual(len(registered), len(set(registered)))
        self.assertEqual(KPI_SUBJECT["food_completion_rate"], "customer")
        self.assertEqual(KPI_SUBJECT["service_units_per_driver_hour"], "driver")
        self.assertEqual(KPI_SUBJECT["driver_food_pickup_wait_saved_ticks"], "driver")
        self.assertEqual(KPI_SUBJECT["merchant_revenue_per_order"], "merchant")
        self.assertEqual(KPI_SUBJECT["mean_food_preparation_queue_ticks"], "merchant")
        self.assertEqual(
            KPI_SUBJECT["mean_integrated_grocery_preparation_queue_ticks"], "merchant")
        self.assertEqual(KPI_SUBJECT["mean_food_ready_to_handover_ticks"], "merchant")
        self.assertEqual(KPI_SUBJECT["platform_revenue_per_order"], "platform")
        self.assertEqual(KPI_SUBJECT["emission_intensity_per_service_unit"], "environment")

    def test_paired_significance_and_holm_correction(self):
        normal = _paired_significance([-2, -1, 0, 1, 2, 3, 4, 5])
        self.assertEqual(normal["normality_test"], "shapiro_wilk")
        self.assertEqual(normal["significance_test"], "paired_t_test")
        nonnormal = _paired_significance([0] * 19 + [10])
        self.assertEqual(nonnormal["significance_test"], "wilcoxon_signed_rank")
        self.assertEqual(_paired_significance([0, 0, 0])["raw_p_value"], 1.0)
        self.assertIsNone(_paired_significance([1, 2])["raw_p_value"])

        rows = [
            {"condition_id": "p_0.5", "primary_metric": True, "raw_p_value": 0.01},
            {"condition_id": "p_0.5", "primary_metric": True, "raw_p_value": 0.04},
            {"condition_id": "p_0.5", "primary_metric": False, "raw_p_value": 0.001},
        ]
        _apply_holm_correction(rows)
        self.assertAlmostEqual(rows[0]["adjusted_p_value_holm"], 0.02)
        self.assertAlmostEqual(rows[1]["adjusted_p_value_holm"], 0.04)
        self.assertIsNone(rows[2]["adjusted_p_value_holm"])

    def test_factorial_scenarios_preserve_merchants_and_validate_dimensions(self):
        small = build_experiment_scenario(42, daily_customers=24, num_drivers=2)
        large = build_experiment_scenario(42, daily_customers=48, num_drivers=4)
        same_demand = build_experiment_scenario(42, daily_customers=24, num_drivers=4)
        self.assertEqual(small.restaurant_positions, large.restaurant_positions)
        self.assertEqual(small.store_positions, large.store_positions)
        self.assertEqual(small.driver_positions, large.driver_positions[:2])
        self.assertEqual(small.orders, same_demand.orders)
        self.assertEqual(small.demand_schedule, same_demand.demand_schedule)
        self.assertEqual(sum(large.demand_schedule), 48)
        self.assertEqual(len(large.orders), 48)
        self.assertTrue(all(sorted(p) == list(range(4)) for p in large.driver_permutations))
        config = SimulationConfig(0.5, seed=42, daily_customers=24, num_drivers=2)
        for mismatch in (replace(config, daily_customers=48), replace(config, num_drivers=4)):
            with self.assertRaises(ValueError):
                IntegratedDeliveryModel(mismatch, scenario=small)
        model = IntegratedDeliveryModel(config, scenario=small)
        self.assertEqual(model.run()["generated_orders"], 24)
        self.assertEqual(len(model.drivers), 2)
        self.assertEqual(model._metadata()["baseline"]["daily_customers"], 24)
        self.assertEqual(model._metadata()["config"]["num_drivers"], 2)

    def test_parameter_validation_and_reference_inclusion(self):
        self.assertEqual(_read_store_selections("nearest_customer"), ["uniform", "nearest_customer"])
        self.assertEqual(_read_store_counts("10,20", 15), [10, 15, 20])
        for value in (0, -1, 1.5, True):
            for field in ("daily_customers", "num_drivers", "num_stores"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    SimulationConfig(0.5, **{field: value})
        for kwargs in ({"food_preparation_multiplier": 0},
                       {"grocery_preparation_multiplier": float("nan")},
                       {"store_selection": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SimulationConfig(0.44, **kwargs)

    def test_sensitivity_controller_builds_full_factorial_with_fixed_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "sensitivity"
            main(["--p-integrated", "0.44", "--replications", "2", "--sensitivity",
                  "--daily-customers", "12", "--num-drivers", "1", "--summary-only",
                  "--output-dir", str(destination)])
            def rows(name):
                with (destination / name).open(newline="", encoding="utf-8") as handle:
                    return list(csv.DictReader(handle))
            runs = rows("run_summaries.csv")
            pairs = rows("paired_differences.csv")
            statistics = rows("paired_statistics.csv")
            effects = rows("sensitivity_effects.csv")
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(runs), 56)
            self.assertEqual(len(pairs), 54)
            self.assertEqual(len(statistics), 27 * len(METRICS))
            self.assertEqual(len(effects), 26 * len(METRICS))
            self.assertEqual(manifest["model_runs"], 56)
            self.assertEqual(manifest["scenario_count"], 2)
            self.assertEqual(len({r["path"] for r in manifest["scenario_records"]}), 2)
            indexed = {(r["replication"], r["condition_id"]): r for r in runs}
            self.assertTrue(all(r["generated_orders"] == r["daily_customers"] for r in runs))
            self.assertTrue(all(int(r["observed_pairs"]) <= 2 for r in statistics))
            for pair in pairs:
                baseline = indexed[(pair["replication"], pair["baseline_condition_id"])]
                treatment = indexed[(pair["replication"], pair["condition_id"])]
                self.assertEqual(baseline["scenario_sha256"], treatment["scenario_sha256"])
                self.assertAlmostEqual(float(pair["delta_food_completion_rate"]),
                                       float(treatment["food_completion_rate"]) -
                                       float(baseline["food_completion_rate"]))
            self.assertEqual(sum(r["primary_comparison"] == "True" for r in pairs), 2)
            self.assertEqual({r["factor"] for r in runs}, {"baseline", "factorial_cell"})
            factorial_cells = {(r["p_integration"], r["num_stores"], r["store_selection"])
                               for r in runs if r["factor"] == "factorial_cell"}
            self.assertEqual(factorial_cells, {
                (probability, count, store)
                for probability in ("0.22", "0.44", "0.66")
                for count in ("10", "15", "20")
                for store in ("uniform", "nearest_restaurant", "nearest_customer")
            })
            self.assertTrue(all(r["food_preparation_multiplier"] == "1.0" for r in runs))
            self.assertTrue(all(r["grocery_preparation_multiplier"] == "1.0" for r in runs))
            effect = next(r for r in effects if r["condition_id"] == "p_0.22_stores_15_uniform"
                          and r["metric"] == "food_completion_rate")
            changes = []
            for replication in ("1", "2"):
                reference = next(r for r in pairs if r["replication"] == replication
                                 and r["condition_id"] == "p_0.44_stores_15_uniform")
                alternative = next(r for r in pairs if r["replication"] == replication
                                   and r["condition_id"] == "p_0.22_stores_15_uniform")
                changes.append(float(alternative["delta_food_completion_rate"]) -
                               float(reference["delta_food_completion_rate"]))
            self.assertAlmostEqual(float(effect["mean_change_in_delta"]), sum(changes) / 2)

    def test_food_inputs_and_driver_order_are_shared_across_policies(self):
        scenario = build_experiment_scenario(42)
        self.assertEqual(scenario.sha256, build_experiment_scenario(42).sha256)
        baseline = IntegratedDeliveryModel(SimulationConfig(0, seed=42), scenario=scenario)
        integrated = IntegratedDeliveryModel(SimulationConfig(1, seed=42), scenario=scenario)
        self.assertEqual(baseline.demand_schedule, integrated.demand_schedule)
        self.assertEqual([a.pos for a in baseline.drivers], [a.pos for a in integrated.drivers])
        for _ in range(120):
            baseline.step()
            integrated.step()
            self.assertEqual(baseline.last_driver_activation_order,
                             integrated.last_driver_activation_order)
        self.assertGreater(len(baseline.orders), 0)
        for order_id in baseline.orders:
            before = baseline.orders[order_id]
            after = integrated.orders[order_id]
            self.assertEqual(before.created_tick, after.created_tick)
            self.assertEqual(before.customer_pos, after.customer_pos)
            self.assertEqual(before.food.merchant_id, after.food.merchant_id)
            self.assertEqual(before.food.preparation_duration, after.food.preparation_duration)
            self.assertEqual(before.food.item_value, after.food.item_value)
            self.assertIsNone(before.grocery)
            self.assertIsNotNone(after.grocery)
        self.assertEqual(baseline._metadata()["experiment_scenario"]["sha256"], scenario.sha256)
        self.assertEqual(baseline._metadata()["schema_version"], 16)

    def test_preparation_multipliers_and_nearest_store_rules(self):
        scenario = build_experiment_scenario(42, daily_customers=24, num_drivers=2)
        for rule in ("nearest_restaurant", "nearest_customer"):
            config = SimulationConfig(1, seed=42, daily_customers=24, num_drivers=2,
                                      food_preparation_multiplier=1.25,
                                      grocery_preparation_multiplier=0.75, store_selection=rule)
            model = IntegratedDeliveryModel(config, scenario=scenario)
            model.run()
            for index, order in enumerate(model.orders.values()):
                drawn = scenario.orders[index]
                self.assertEqual(order.food.preparation_duration, math.ceil(drawn.food_preparation * 1.25))
                self.assertEqual(order.grocery.preparation_duration,
                                 math.ceil(drawn.grocery_preparation * 0.75))
                restaurant = model.merchants_by_id[order.food.merchant_id]
                selected = model.merchants_by_id[order.grocery.merchant_id]
                target = restaurant.pos if rule == "nearest_restaurant" else order.customer_pos
                distance = lambda store: abs(store.pos[0] - target[0]) + abs(store.pos[1] - target[1])
                self.assertEqual(distance(selected), min(map(distance, model.stores)))

    def test_store_rule_uses_requested_target_and_stable_tie_break(self):
        positions = [(1, 0), (0, 1), (48, 49)] + [(20, 20)] * 12
        restaurant_pos, customer_pos = (0, 0), (49, 49)
        # Stores 0 and 1 are equally near the Restaurant. random_index=1 must
        # therefore select the second tied Store to preserve common randomness.
        for rule, expected_index in (("uniform", 1), ("nearest_restaurant", 1),
                                     ("nearest_customer", 2)):
            model = IntegratedDeliveryModel(SimulationConfig(1, store_selection=rule))
            for store, position in zip(model.stores, positions):
                model.grid.move_agent(store, position)
            selected = model._select_store(1, restaurant_pos, customer_pos)
            self.assertIs(selected, model.stores[expected_index])

    def test_uniform_rule_uses_shared_uniform_for_variable_store_counts(self):
        scenario = build_experiment_scenario(42, daily_customers=24, num_drivers=2, num_stores=20)
        model = IntegratedDeliveryModel(SimulationConfig(1, seed=42, daily_customers=24,
                                                         num_drivers=2, num_stores=10), scenario=scenario)
        model.run()
        for index, order in enumerate(model.orders.values()):
            expected = min(int(scenario.orders[index].store_uniform * 10), 9)
            self.assertEqual(order.grocery.merchant_id,
                             model.stores[expected].unique_id)

    def test_controller_exports_paired_differences_and_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "experiment"
            main(["--p-integrated", "0.5", "--replications", "2", "--seed-start", "42",
                  "--summary-only", "--output-dir", str(output_dir)])
            with (output_dir / "run_summaries.csv").open(newline="", encoding="utf-8") as handle:
                runs = list(csv.DictReader(handle))
            with (output_dir / "paired_differences.csv").open(newline="", encoding="utf-8") as handle:
                pairs = list(csv.DictReader(handle))
            with (output_dir / "paired_statistics.csv").open(newline="", encoding="utf-8") as handle:
                statistics = list(csv.DictReader(handle))
            with (output_dir / "emission_statistics.csv").open(newline="", encoding="utf-8") as handle:
                emission_statistics = list(csv.DictReader(handle))
            with (output_dir / "driver_statistics.csv").open(newline="", encoding="utf-8") as handle:
                driver_statistics = list(csv.DictReader(handle))
            with (output_dir / "service_level_statistics.csv").open(newline="", encoding="utf-8") as handle:
                service_statistics = list(csv.DictReader(handle))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            experiment_log = (output_dir / "experiment.log").read_text(encoding="utf-8")
            self.assertEqual(len(runs), 4)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(len(manifest["scenario_records"]), 2)
            self.assertEqual(manifest["experiment_schema_version"], 17)
            self.assertEqual(manifest["kpi_taxonomy"],
                             "subject-centric; research questions select metrics across subjects")
            self.assertEqual(manifest["kpi_subjects"]["customer"][0], "food_completion_rate")
            self.assertIn("mean_food_preparation_queue_ticks", runs[0])
            self.assertIn("mean_integrated_grocery_preparation_queue_ticks", runs[0])
            self.assertIn(
                "mean_food_preparation_queue_ticks", manifest["kpi_subjects"]["merchant"])
            self.assertIn(
                "mean_integrated_grocery_preparation_queue_ticks",
                manifest["kpi_subjects"]["merchant"],
            )
            self.assertTrue(all(row["subject"] for row in statistics))
            self.assertTrue(all(row["subject"] for row in emission_statistics))
            self.assertTrue(all(row["subject"] == "driver" for row in driver_statistics))
            self.assertTrue(all(row["subject"] for row in service_statistics))
            self.assertEqual(manifest["experiment_log"], "experiment.log")
            self.assertIn("experiment_started", experiment_log)
            self.assertIn("condition_completed", experiment_log)
            self.assertIn("replication_completed", experiment_log)
            self.assertIn("experiment_completed", experiment_log)
            self.assertTrue(all(row["generated_orders"] == "300" for row in runs))
            self.assertTrue(all(row["integrated_orders"] == "0" for row in runs
                                if row["p_integration"] == "0.0"))
            self.assertTrue(all(int(row["completed_service_units"]) ==
                                int(row["completed_food_service_units"]) +
                                int(row["completed_grocery_service_units"])
                                for row in runs))
            self.assertTrue(any(row["metric"] == "service_units_per_driver_hour" and
                                row["primary_metric"] == "True" for row in statistics))
            self.assertTrue(any(row["metric"] == "emission_intensity_per_service_unit" and
                                row["primary_metric"] == "True" for row in statistics))
            self.assertEqual({row["metric"] for row in emission_statistics}, {
                "marginal_emission_per_completed_grocery",
                "marginal_emission_per_additional_service_unit",
                "service_output_change_pct", "total_emission_change_pct",
                "emission_output_elasticity",
            })
            self.assertEqual({row["metric"] for row in driver_statistics}, set(DRIVER_PAIR_METRICS))
            self.assertTrue(any(row["metric"] == "driver_food_pickup_wait_saved_ticks" and
                                row["primary_driver_metric"] == "True"
                                for row in driver_statistics))
            self.assertIn("integrated_completion_rate", {row["metric"] for row in service_statistics})
            self.assertIn("mean_integrated_completion_ticks", {row["metric"] for row in service_statistics})
            self.assertFalse(any(row["metric"].startswith("p90_") for row in service_statistics))
            self.assertTrue(all(row["observed_replications"] == "2" for row in service_statistics))
            self.assertTrue(any(row["metric"] == "food_completion_rate" and
                                row["primary_metric"] == "True" for row in statistics))
            self.assertTrue(any(row["metric"] == "mean_food_completion_ticks" and
                                row["primary_metric"] == "True" for row in statistics))
            for pair in pairs:
                if pair["baseline_mean_food_pickup_wait_ticks"] and pair["integrated_mean_food_pickup_wait_ticks"]:
                    expected_saved = (float(pair["baseline_mean_food_pickup_wait_ticks"]) -
                                      float(pair["integrated_mean_food_pickup_wait_ticks"]))
                    self.assertAlmostEqual(float(pair["driver_food_pickup_wait_saved_ticks"]),
                                           expected_saved)
                delta_emission = (float(pair["integrated_total_emission_units"]) -
                                  float(pair["baseline_total_emission_units"]))
                completed_grocery = next(
                    int(row["completed_grocery_service_units"]) for row in runs
                    if row["replication"] == pair["replication"] and
                    row["condition_id"] == pair["condition_id"]
                )
                self.assertAlmostEqual(float(pair["marginal_emission_per_completed_grocery"]),
                                       delta_emission / completed_grocery)
            self.assertTrue(all(abs(float(row["delta_food_completion_rate"]) -
                                    (float(row["integrated_food_completion_rate"]) -
                                     float(row["baseline_food_completion_rate"]))) < 1e-12
                                for row in pairs))
            completion = next(row for row in statistics if row["metric"] == "food_completion_rate")
            self.assertEqual(completion["observed_pairs"], "2")
            self.assertNotEqual(completion["ci95_lower"], "")
            self.assertEqual(completion["raw_p_value"], "")
            paired_metric_names = {row["metric"] for row in statistics}
            self.assertTrue({"food_completion_rate", "food_cancellation_rate",
                             "food_unfinished_rate", "mean_food_completion_ticks"}
                            <= paired_metric_names)
            self.assertTrue({"completion_rate", "cancellation_rate", "active_rate",
                             "mean_completion_ticks"}.isdisjoint(paired_metric_names))

    def test_visualizer_reset_reuses_shared_scenario(self):
        scenario = build_experiment_scenario(42, daily_customers=24, num_drivers=2)
        with tempfile.TemporaryDirectory() as temporary:
            model = IntegratedDeliveryModel(SimulationConfig(0.5, seed=42, daily_customers=24, num_drivers=2),
                                            scenario=scenario)
            visualizer = GridVisualizer(model, output_dir=Path(temporary) / "view")
            try:
                visualizer._step_once()
                visualizer._reset()
                self.assertIs(visualizer.model.experiment_scenario, scenario)
                self.assertEqual(visualizer.model.tick_counter, 0)
                self.assertEqual(len(visualizer.model.drivers), 2)
                self.assertEqual(sum(visualizer.model.demand_schedule), 24)
                self.assertEqual(visualizer.model._metadata()["experiment_scenario"]["sha256"],
                                 scenario.sha256)
            finally:
                plt.close(visualizer.figure)


if __name__ == "__main__":
    unittest.main()
