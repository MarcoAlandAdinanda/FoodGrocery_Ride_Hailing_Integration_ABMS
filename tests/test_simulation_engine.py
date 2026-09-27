"""Behavior checks against the PDF and the approved implementation decisions."""

import csv
import hashlib
import io
import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.simulation_engine import (
    ComponentStatus, CustomerState, DriverState, HORIZON, IntegratedDeliveryModel,
    MerchantState, OrderStatus, OrderType, SimulationConfig, build_demand_schedule,
    manhattan, load_hourly_weights, merchant_revenue, SPECIFICATION_PDF,
)
from src.simulation_parameters import (
    FOOD_PREPARATION, FOOD_ITEM_VALUE, GROCERY_PREPARATION, GROCERY_ITEM_VALUE,
    TIMEOUT,
)


def fixture_model(probability=0):
    model = IntegratedDeliveryModel(SimulationConfig(probability))
    model.demand_schedule = [0] * HORIZON
    return model


def colocate(model, position=(0, 0)):
    for agent in model.agents:
        model.grid.move_agent(agent, position)


def fixture_order(model, duration=0, customer_pos=(0, 0)):
    with patch.object(model, "_random_position", return_value=customer_pos):
        order = model._create_order()
    for component in order.components:
        component.preparation_duration = duration
    return order


def csv_bytes(rows=None):
    rows = rows if rows is not None else [(h, 1) for h in range(24)]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["hour", "avg_demand"])
    writer.writerows(rows)
    return buffer.getvalue().encode()


class DemandTests(unittest.TestCase):
    def test_probability_required_and_validated(self):
        with self.assertRaises(TypeError):
            SimulationConfig()
        for probability in (-1, 1.1, float("nan"), float("inf"), True, "0.5", None):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                SimulationConfig(probability)
        for seed in (-1, 1.2, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                SimulationConfig(0.5, seed=seed)

    def test_hourly_average_and_largest_remainder_ties(self):
        weights = load_hourly_weights(csv_bytes([(h, v) for h in range(24) for v in (1, 3)]))
        np.testing.assert_allclose(weights, [1 / 24] * 24)
        hourly, minutes = build_demand_schedule(weights, np.random.default_rng(42))
        self.assertEqual(hourly, [13] * 12 + [12] * 12)
        self.assertEqual(sum(minutes), 300)
        self.assertEqual(len(minutes), 1440)
        for hour in range(24):
            self.assertEqual(sum(minutes[hour * 60:(hour + 1) * 60]), hourly[hour])
        self.assertNotEqual(minutes[:60], [1] * 13 + [0] * 47)

    def test_divmod_can_spawn_multiple_customers_at_same_tick(self):
        weights = np.array([1.0] + [0.0] * 23)
        hourly, minutes = build_demand_schedule(weights, np.random.default_rng(42))
        self.assertEqual(hourly[0], 300)
        self.assertEqual(minutes[:60], [5] * 60)
        self.assertEqual(sum(minutes[60:]), 0)

    def test_invalid_csv(self):
        examples = [b"wrong,columns\n1,2\n", b"", csv_bytes([(h, 1) for h in range(23)]),
                    csv_bytes([(h, 0) for h in range(24)]),
                    csv_bytes([(h, -1 if h == 0 else 1) for h in range(24)]),
                    csv_bytes([(0.5 if h == 0 else h, 1) for h in range(24)]),
                    csv_bytes([(h, "nan" if h == 0 else 1) for h in range(24)]),
                    csv_bytes([(h, "inf" if h == 0 else 1) for h in range(24)]),
                    csv_bytes([(h, 1) for h in range(25)])]
        for content in examples:
            with self.subTest(content=content[:30]), self.assertRaises(ValueError):
                load_hourly_weights(content)

    def test_large_finite_weights_do_not_overflow(self):
        weights = load_hourly_weights(csv_bytes([(h, 1e308) for h in range(24)]))
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(sum(weights)), 1)

    def test_baseline_populations_and_no_order_agents(self):
        model = fixture_model(1)
        self.assertEqual((model.grid.width, model.grid.height, model.grid.torus), (50, 50, False))
        self.assertEqual((len(model.drivers), len(model.restaurants), len(model.stores)), (12, 50, 15))
        initial = len(model.agents)
        order = fixture_order(model)
        self.assertEqual(len(model.agents), initial + 1)
        self.assertEqual(len(order.components), 2)
        self.assertEqual(order.order_type, OrderType.INTEGRATED)
        self.assertFalse(hasattr(model, "dispatcher"))

    def test_normal_parameters_clamping_and_ceiling(self):
        model = fixture_model(1)
        with patch.object(model, "_normal_nonnegative_integer", side_effect=[15, 30000, 20, 50000]) as draw:
            order = fixture_order(model)
        self.assertEqual([c.args for c in draw.call_args_list],
                         [FOOD_PREPARATION, FOOD_ITEM_VALUE, GROCERY_PREPARATION,
                          GROCERY_ITEM_VALUE])
        self.assertEqual([c.item_value for c in order.components], [30000, 50000])
        model.rng = Mock(normal=Mock(side_effect=[-2.5, 0, 1.01]))
        self.assertEqual([model._normal_nonnegative_integer(0, 1) for _ in range(3)], [0, 0, 2])


class LifecycleTests(unittest.TestCase):
    def test_integrated_colocated_timeline_and_no_same_tick_state_chaining(self):
        model = fixture_model(1)
        colocate(model)
        order = fixture_order(model)
        model.step()  # t=0: READY, then claim only.
        driver = model.drivers_by_id[order.assigned_driver_id]
        self.assertEqual(driver.state, DriverState.GROCERY_TRAVEL)
        self.assertEqual(order.grocery.ready_tick, 0)
        self.assertIsNone(order.grocery.handed_over_tick)
        expected = [DriverState.GROCERY_PICKUP, DriverState.FOOD_TRAVEL,
                    DriverState.FOOD_PICKUP, DriverState.DELIVER, DriverState.IDLE]
        for state in expected:
            model.step()
            self.assertEqual(driver.state, state)
        self.assertEqual((order.grocery_arrival_tick, order.grocery.handed_over_tick), (1, 2))
        self.assertEqual((order.food_arrival_tick, order.food.handed_over_tick), (3, 4))
        self.assertEqual(order.delivered_tick, 5)
        self.assertEqual(order.status, OrderStatus.DELIVERED)
        self.assertFalse(order.settled)
        self.assertEqual(driver.movement_count, 0)
        self.assertEqual(driver.busy_ticks, 6)
        model.step()
        self.assertEqual(order.status, OrderStatus.COMPLETED)
        self.assertEqual(order.completed_tick, 6)
        self.assertEqual(order.time_metrics()["acknowledgement_delay_ticks"], 1)
        self.assertEqual(order.time_metrics()["delivery_service_ticks"], 5)
        self.assertEqual(order.time_metrics()["completion_total_ticks"], 6)
        self.assertNotIn(order.customer_id, model.customers)
        self.assertNotIn(order.order_id, model.active_orders)
        self.assertEqual(len(model.agents), len(model.drivers) + len(model.restaurants) + len(model.stores))
        self.assertTrue(order.settled)

    def test_preparation_zero_one_and_parallel_records(self):
        model = fixture_model()
        first, second = fixture_order(model, duration=0), fixture_order(model, duration=1)
        merchant = model.merchants_by_id[first.food.merchant_id]
        original = model.merchants_by_id[second.food.merchant_id]
        del original.components[second.order_id]
        second.food.merchant_id = merchant.unique_id
        merchant.components[second.order_id] = second.food
        self.assertEqual(merchant.state, MerchantState.ACTIVE)
        merchant.step()
        self.assertEqual(first.food.status, ComponentStatus.READY)
        self.assertEqual(second.food.status, ComponentStatus.PREPARING)
        model.tick_counter = 1
        merchant.step()
        self.assertEqual(second.food.ready_tick, 1)
        self.assertEqual(second.food.status, ComponentStatus.READY)

    def test_timeout_precedes_matching_at_exact_threshold(self):
        model = fixture_model()
        order = fixture_order(model, duration=100)
        model.tick_counter = TIMEOUT - 1
        customer = model.customers[order.customer_id]
        customer.step()
        self.assertEqual(order.status, OrderStatus.AVAILABLE)
        model.tick_counter = TIMEOUT
        model.step()
        self.assertEqual(order.status, OrderStatus.CANCELLED)
        self.assertIsNone(order.assigned_driver_id)
        self.assertEqual(order.cancelled_tick, TIMEOUT)
        self.assertEqual(order.food.cancellation_event, "PREPARATION_STOPPED")
        merchant = model.merchants_by_id[order.food.merchant_id]
        self.assertEqual(merchant.cancellation_kpis, {
            "preparing_cancelled_units": 1,
            "preparing_cancelled_product_value": Decimal(order.food.item_value),
        })
        self.assertEqual(model.summary()["store_preparing_cancelled_units"], 0)
        self.assertEqual(order.driver_revenue, 0)
        self.assertEqual(customer.state, CustomerState.EXIT)

    def test_assigned_orders_never_timeout(self):
        model = fixture_model()
        order = fixture_order(model, duration=1000)
        model.drivers[0].step()
        model.tick_counter = 100
        model.customers[order.customer_id].step()
        self.assertEqual(order.status, OrderStatus.ASSIGNED)
        self.assertEqual(model.customers[order.customer_id].state, CustomerState.IN_SERVICE)
        self.assertIsNone(order.cancelled_tick)

    def test_customer_cancellation_wins_over_becoming_ready_same_tick(self):
        model = fixture_model()
        order = fixture_order(model, duration=TIMEOUT)
        model.tick_counter = TIMEOUT
        model.step()
        self.assertEqual(order.food.cancellation_event, "PREPARATION_STOPPED")
        self.assertIsNone(order.food.ready_tick)
        self.assertEqual(model.summary()["restaurant_preparing_cancelled_units"], 1)
        self.assertEqual(model.summary()["restaurant_preparing_cancelled_product_value"],
                         Decimal(order.food.item_value))

    def test_ready_cancellation_disposal_vs_return_recorded_once(self):
        model = fixture_model(1)
        order = fixture_order(model)
        merchants = [model.merchants_by_id[c.merchant_id] for c in order.components]
        for merchant in merchants:
            merchant.step()
        model.tick_counter = 60
        model.customers[order.customer_id].step()
        for _ in range(2):
            for merchant in merchants:
                merchant.step()
        self.assertEqual(order.food.cancellation_event, "PREPARED_FOOD_DISPOSAL")
        self.assertEqual(order.grocery.cancellation_event, "GROCERY_CANCELLATION_RETURN")
        self.assertTrue(all(m.state == MerchantState.IDLE for m in merchants))
        self.assertEqual(model.summary()["food_disposal_events"], 1)
        self.assertEqual(model.summary()["grocery_return_events"], 1)
        self.assertTrue(all(m.revenue == 0 for m in merchants))

    def test_every_handover_guard_and_grocery_precedence(self):
        model = fixture_model(1)
        colocate(model)
        order = fixture_order(model)
        driver = model.drivers[0]
        driver.step()
        restaurant = model.merchants_by_id[order.food.merchant_id]
        order.food.status = ComponentStatus.READY
        driver.state = DriverState.FOOD_PICKUP
        self.assertFalse(restaurant._can_handover(order, order.food))
        order.grocery.status = ComponentStatus.HANDED_OVER
        self.assertTrue(restaurant._can_handover(order, order.food))
        with patch.object(order.food, "status", ComponentStatus.PREPARING):
            self.assertFalse(restaurant._can_handover(order, order.food))
        with patch.object(order, "status", OrderStatus.AVAILABLE):
            self.assertFalse(restaurant._can_handover(order, order.food))
        with patch.object(order, "assigned_driver_id", None):
            self.assertFalse(restaurant._can_handover(order, order.food))
        with patch.object(driver, "current_order_id", order.order_id + 1):
            self.assertFalse(restaurant._can_handover(order, order.food))
        with patch.object(driver, "state", DriverState.FOOD_TRAVEL):
            self.assertFalse(restaurant._can_handover(order, order.food))
        model.grid.move_agent(driver, (1, 1))
        self.assertFalse(restaurant._can_handover(order, order.food))

    def test_pickup_wait_counts_only_unsatisfied_pickup_activations(self):
        model = fixture_model()
        colocate(model)
        order = fixture_order(model, duration=5)
        for _ in range(6):
            model.step()
        # t0 claim, t1 arrival, t2/t3/t4 wait, t5 handover.
        self.assertEqual(order.food_pickup_wait_ticks, 3)
        self.assertEqual(order.food.handed_over_tick, 5)
        self.assertEqual(order.time_metrics()["food_pickup_wait_ticks"], 3)
        self.assertEqual(order.movement_count, 0)

    def test_resolving_previous_order_preserves_new_driver_assignment(self):
        model = fixture_model()
        colocate(model)
        previous = fixture_order(model)
        for _ in range(4):
            model.step()
        driver = model.drivers_by_id[previous.assigned_driver_id]
        self.assertEqual(previous.status, OrderStatus.DELIVERED)
        new = fixture_order(model)
        model.customers[previous.customer_id].step()
        driver.step()
        self.assertEqual(driver.current_order_id, new.order_id)
        model._resolve_orders()
        self.assertEqual(driver.current_order_id, new.order_id)
        self.assertEqual(driver.state, DriverState.FOOD_TRAVEL)
        self.assertEqual(new.status, OrderStatus.ASSIGNED)
        self.assertTrue(previous.settled)
        revenue = driver.revenue
        model._settle_order(previous)
        self.assertEqual(driver.revenue, revenue)

    def test_last_tick_delivery_is_not_acknowledged_or_settled(self):
        model = fixture_model()
        colocate(model)
        model.tick_counter = HORIZON - 4
        order = fixture_order(model)
        for _ in range(4):
            model.step()
        self.assertTrue(model.finished)
        self.assertEqual(order.delivered_tick, 1439)
        self.assertEqual(order.status, OrderStatus.DELIVERED)
        self.assertIsNone(order.completed_tick)
        self.assertFalse(order.settled)
        self.assertEqual(order.driver_revenue, 0)
        self.assertIn(order.customer_id, model.customers)
        before = model.snapshot()
        model.step()
        self.assertEqual(model.snapshot(), before)


class MatchingMovementFinanceTests(unittest.TestCase):
    def test_matching_distance_then_fifo_then_id(self):
        model = fixture_model()
        colocate(model)
        oldest, younger = fixture_order(model), fixture_order(model)
        oldest.created_tick, younger.created_tick = -2, -1
        driver = model.drivers[0]
        driver.step()
        self.assertEqual(driver.current_order_id, oldest.order_id)
        model.drivers[1].step()
        self.assertEqual(model.drivers[1].current_order_id, younger.order_id)
        first, second = fixture_order(model), fixture_order(model)
        model.drivers[2].step()
        self.assertEqual(model.drivers[2].current_order_id, first.order_id)
        far = fixture_order(model)
        # Assign a different restaurant to far, so shared merchant positions do not affect second.
        far_merchant = next(m for m in model.restaurants if m.unique_id != second.food.merchant_id)
        del model.merchants_by_id[far.food.merchant_id].components[far.order_id]
        far.food.merchant_id = far_merchant.unique_id
        far_merchant.components[far.order_id] = far.food
        model.grid.move_agent(far_merchant, (10, 10))
        far.created_tick = -100
        model.drivers[3].step()
        self.assertEqual(model.drivers[3].current_order_id, second.order_id)

    def test_integrated_matching_uses_store_not_restaurant(self):
        model = fixture_model(1)
        colocate(model)
        first = fixture_order(model)
        first_store = model.merchants_by_id[first.grocery.merchant_id]
        model.grid.move_agent(first_store, (20, 20))
        second = fixture_order(model)
        different_store = next(s for s in model.stores if s != first_store)
        del model.merchants_by_id[second.grocery.merchant_id].components[second.order_id]
        second.grocery.merchant_id = different_store.unique_id
        different_store.components[second.order_id] = second.grocery
        model.drivers[0].step()
        self.assertEqual(model.drivers[0].current_order_id, second.order_id)

    def test_manhattan_distance(self):
        for start, target, expected in (
            ((0, 0), (0, 0), 0), ((0, 0), (5, 0), 5),
            ((0, 0), (0, 5), 5), ((0, 0), (3, 4), 7),
            ((0, 0), (49, 49), 98), ((3, 7), (6, 2), 8),
        ):
            with self.subTest(start=start, target=target):
                self.assertEqual(manhattan(start, target), expected)
                self.assertEqual(manhattan(target, start), expected)

    def test_matching_uses_manhattan_not_straight_line_or_diagonal_steps(self):
        for probability in (0, 1):
            with self.subTest(probability=probability):
                model = fixture_model(probability)
                colocate(model)
                first, second = fixture_order(model), fixture_order(model)
                merchants = model.stores if probability else model.restaurants
                for order, merchant, position in zip(
                    (first, second), merchants[:2], ((3, 3), (5, 0))
                ):
                    component = order.grocery if probability else order.food
                    del model.merchants_by_id[component.merchant_id].components[order.order_id]
                    component.merchant_id = merchant.unique_id
                    merchant.components[order.order_id] = component
                    model.grid.move_agent(merchant, position)
                model.drivers[0].step()
                self.assertEqual(model.drivers[0].current_order_id, second.order_id)

    def test_movement_steps_orthogonal_cost_and_boundary(self):
        model = fixture_model()
        colocate(model)
        order = fixture_order(model)
        merchant = model.merchants_by_id[order.food.merchant_id]
        model.grid.move_agent(merchant, (3, 4))
        driver = model.drivers[0]
        driver.step()
        self.assertEqual(driver.movement_count, 0)
        for step in range(7):
            before = driver.pos
            driver.step()
            self.assertEqual(manhattan(before, driver.pos), 1)
            if step == 0:
                # Euclidean score favors the larger remaining coordinate gap.
                self.assertEqual(driver.pos, (0, 1))
        self.assertEqual(driver.pos, (3, 4))
        self.assertEqual(driver.movement_count, 7)
        self.assertEqual(order.first_pickup_moves, 7)
        self.assertEqual(model._driver_row(driver)["travel_distance_km"], Decimal("3.5"))
        self.assertEqual(model._driver_row(driver)["operating_cost"], Decimal(1400))
        self.assertEqual(model._driver_row(driver)["emission_units"], 210)
        # Unlike Manhattan score plus tuple tie-breaking, Euclidean favors x here.
        model.grid.move_agent(driver, (0, 0))
        model.grid.move_agent(merchant, (4, 3))
        driver.state = DriverState.FOOD_TRAVEL
        driver.step()
        self.assertEqual(driver.pos, (1, 0))
        model.grid.move_agent(driver, (49, 49))
        model.grid.move_agent(merchant, (0, 0))
        driver.state = DriverState.FOOD_TRAVEL
        driver.step()
        self.assertEqual(driver.pos, (48, 49))
        for _ in range(97):
            before = driver.pos
            driver.step()
            self.assertEqual(manhattan(before, driver.pos), 1)
            self.assertTrue(all(0 <= coordinate < 50 for coordinate in driver.pos))
        self.assertEqual(driver.pos, (0, 0))
        moves = driver.movement_count
        driver.state = DriverState.FOOD_TRAVEL
        driver.step()  # Zero-distance arrival still enters pickup without moving.
        self.assertEqual(driver.state, DriverState.FOOD_PICKUP)
        driver.step()  # Waiting for handover also adds no movement.
        self.assertEqual(driver.movement_count, moves)

    def test_actual_legs_partial_export_and_billable_distance(self):
        for probability in (0, 1):
            with self.subTest(probability=probability):
                model = fixture_model(probability)
                colocate(model)
                for restaurant in model.restaurants:
                    model.grid.move_agent(restaurant, (1, 1))
                for store in model.stores:
                    model.grid.move_agent(store, (3, 1))
                order = fixture_order(model, customer_pos=(4, 3))
                quoted = order.billable_distance_km
                model.step()  # Claim.
                driver = model.drivers_by_id[order.assigned_driver_id]
                model.step()  # One actual step toward first pickup.
                self.assertEqual(order.first_pickup_moves, 1)
                self.assertEqual(order.billable_distance_km, quoted)
                with TemporaryDirectory() as temporary:
                    destination = model.export_results(Path(temporary) / "partial")
                    with (destination / "orders.csv").open(newline="", encoding="utf-8") as handle:
                        row = next(csv.DictReader(handle))
                    self.assertEqual(Decimal(row["travel_distance_km"]), Decimal("0.5"))
                    self.assertEqual(Decimal(row["billable_distance_km"]), quoted)
                    self.assertEqual(Decimal(row["driver_operating_cost"]), Decimal(200))
                    self.assertEqual(int(row["emission_units"]), 30)
                    metadata = json.loads((destination / "metadata.json").read_text())
                    self.assertEqual(metadata["schema_version"], 7)
                    self.assertEqual(metadata["financial_parameters"]["delivery_minimum_fee"], "9000")
                    self.assertEqual(metadata["financial_parameters"]["delivery_fee_per_km"], "2250")
                    self.assertEqual(metadata["financial_parameters"]["driver_platform_share"], "0.08")
                    self.assertEqual(metadata["financial_parameters"]["driver_delivery_fee_share"], "0.92")
                    self.assertEqual(
                        metadata["financial_parameters"]["merchant_platform_share"],
                        "0.1781025272727272727272727273",
                    )
                    self.assertEqual(metadata["policies"]["matching"],
                                     "Manhattan to first merchant, creation tick, order ID")
                    self.assertIn("Manhattan", metadata["policies"]["billable_distance"])
                    self.assertIn("Von Neumann", metadata["policies"]["movement"])
                    self.assertIn("squared Euclidean", metadata["policies"]["movement"])
                    self.assertIn("(x, y)", metadata["policies"]["movement"])
                    self.assertEqual(metadata["specification"]["sha256"],
                                     hashlib.sha256(SPECIFICATION_PDF.read_bytes()).hexdigest())
                for _ in range(30):
                    model.step()
                    if order.status == OrderStatus.COMPLETED:
                        break
                self.assertEqual(order.status, OrderStatus.COMPLETED)
                self.assertEqual(order.first_pickup_moves, 4 if probability else 2)
                self.assertEqual(order.store_to_restaurant_moves, 2 if probability else 0)
                self.assertEqual(order.restaurant_to_customer_moves, 5)
                self.assertEqual(driver.movement_count, order.movement_count)
                self.assertEqual(model._order_row(order)["travel_distance_km"],
                                 quoted + Decimal(order.first_pickup_moves) * Decimal("0.5"))

    def test_billable_legs_quotes_and_exact_settlement(self):
        for probability, expected_steps, expected_fee in ((0, 10, 11250), (1, 12, 13500)):
            with self.subTest(probability=probability):
                model = fixture_model(probability)
                for restaurant in model.restaurants:
                    model.grid.move_agent(restaurant, (1, 1))
                for store in model.stores:
                    model.grid.move_agent(store, (3, 1))
                with patch.object(model, "_normal_nonnegative_integer", side_effect=[0, 30000, 0, 50000]):
                    order = fixture_order(model, customer_pos=(11, 1))
                self.assertEqual(order.billable_distance_km, Decimal(expected_steps) * Decimal("0.5"))
                self.assertEqual(order.delivery_fee, Decimal(expected_fee))
                item_total = 30000 if probability == 0 else 80000
                self.assertEqual(order.customer_payment, Decimal(item_total) * Decimal("1.1") + order.delivery_fee)
                driver = model.drivers[0]
                driver.step()
                model._settle_order(order)
                self.assertEqual(driver.revenue, 0)
                order.status = OrderStatus.COMPLETED
                model._settle_order(order)
                self.assertEqual(order.driver_revenue, order.delivery_fee * Decimal("0.92"))
                expected_merchant = sum((Decimal(c.item_value) * Decimal("0.90408722") for c in order.components), Decimal(0))
                self.assertEqual(sum(c.settled_revenue for c in order.components), expected_merchant)
                self.assertEqual(order.platform_revenue, order.customer_payment - order.driver_revenue - expected_merchant)
                self.assertEqual(merchant_revenue(30000), Decimal("27122.61660000"))


class CancellationKpiTests(unittest.TestCase):
    def test_integrated_components_evaluated_independently(self):
        for food_preparing, grocery_preparing in ((True, True), (True, False),
                                                  (False, True), (False, False)):
            with self.subTest(food=food_preparing, grocery=grocery_preparing):
                model = fixture_model(1)
                order = fixture_order(model)
                for component, preparing, value in ((order.food, food_preparing, 30000),
                                                     (order.grocery, grocery_preparing, 50000)):
                    component.preparation_duration = 100 if preparing else 0
                    component.item_value = value
                    model.merchants_by_id[component.merchant_id].step()
                model.tick_counter = 60
                model.step()
                summary = model.summary()
                for label, component, preparing in (("restaurant", order.food, food_preparing),
                                                     ("store", order.grocery, grocery_preparing)):
                    expected_units = int(preparing)
                    expected_value = Decimal(component.item_value if preparing else 0)
                    merchant = model.merchants_by_id[component.merchant_id]
                    self.assertEqual(merchant.cancellation_kpis["preparing_cancelled_units"], expected_units)
                    self.assertEqual(merchant.cancellation_kpis["preparing_cancelled_product_value"], expected_value)
                    self.assertEqual(summary[f"{label}_preparing_cancelled_units"], expected_units)
                    self.assertEqual(summary[f"{label}_preparing_cancelled_product_value"], expected_value)
                    merchant.step()
                    merchant.step()
                self.assertEqual(summary["food_disposal_events"], int(not food_preparing))
                self.assertEqual(summary["grocery_return_events"], int(not grocery_preparing))
                for key in ("restaurant_revenue", "store_revenue", "platform_revenue", "driver_revenue",
                            "driver_profit", "driver_operating_cost", "settled_customer_payment"):
                    self.assertEqual(summary[key], 0)
                self.assertFalse(order.settled)
                self.assertEqual(model.summary(), summary)

    def test_active_and_completed_orders_do_not_count(self):
        model = fixture_model(1)
        colocate(model)
        order = fixture_order(model)
        for tick in range(8):
            for merchant in model.merchants_by_id.values():
                self.assertEqual(merchant.cancellation_kpis, {
                    "preparing_cancelled_units": 0,
                    "preparing_cancelled_product_value": Decimal(0),
                })
            if tick < 7:
                model.step()
        self.assertEqual(order.status, OrderStatus.COMPLETED)
        self.assertTrue(order.settled)

    def test_cumulative_zero_value_and_repeated_partial_exports(self):
        model = fixture_model(1)
        first = fixture_order(model, duration=100)
        second = fixture_order(model, duration=100)
        for a, b, first_value, second_value in ((first.food, second.food, 0, 30000),
                                               (first.grocery, second.grocery, 50000, 20000)):
            a.item_value, b.item_value = first_value, second_value
            del model.merchants_by_id[b.merchant_id].components[b.order_id]
            b.merchant_id = a.merchant_id
            model.merchants_by_id[a.merchant_id].components[b.order_id] = b
        model.tick_counter = 60
        model.step()
        before = model.snapshot()
        with TemporaryDirectory() as temporary:
            for run_index in range(2):
                destination = model.export_results(Path(temporary) / f"run_{run_index}")
                tables = {}
                for name in ("merchants", "components", "orders", "kpi_ticks"):
                    with (destination / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        self.assertFalse(any("compensation" in key for key in reader.fieldnames))
                        tables[name] = list(reader)
                summary = json.loads((destination / "summary.json").read_text())
                metadata = json.loads((destination / "metadata.json").read_text())
                self.assertEqual(metadata["schema_version"], 7)
                self.assertNotIn("UNRESOLVED", json.dumps([summary, metadata]))
                self.assertEqual(summary["run_status"], "PARTIAL")
                for kind, label, expected_value in (("FOOD", "restaurant", 30000),
                                                     ("GROCERY", "store", 70000)):
                    rows = [m for m in tables["merchants"] if m["kind"] == kind]
                    self.assertEqual(sum(int(m["preparing_cancelled_units"]) for m in rows), 2)
                    self.assertEqual(sum(Decimal(m["preparing_cancelled_product_value"]) for m in rows), expected_value)
                    affected = [m for m in rows if int(m["preparing_cancelled_units"])]
                    self.assertEqual(len(affected), 1)
                    for suffix, expected in (("units", 2), ("product_value", expected_value)):
                        key = f"{label}_preparing_cancelled_{suffix}"
                        self.assertEqual(Decimal(summary[key]), expected)
                        self.assertEqual(Decimal(tables["kpi_ticks"][-1][key]), expected)
                    source = [c for c in tables["components"] if c["kind"] == kind
                              and c["cancellation_event"] == "PREPARATION_STOPPED"]
                    self.assertEqual(len(source), 2)
                    self.assertEqual(sum(Decimal(c["item_value"]) for c in source), expected_value)
                    self.assertTrue(all(c["merchant_id"] == affected[0]["merchant_id"] for c in source))
                self.assertEqual(model.snapshot(), before)
                exported_merchants = {int(m["merchant_id"]): m for m in tables["merchants"]}
                for merchant in before["merchants"]:
                    row = exported_merchants[merchant["merchant_id"]]
                    for key in ("preparing_cancelled_units", "preparing_cancelled_product_value"):
                        self.assertEqual(Decimal(row[key]), merchant[key])


class FullRunAndExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = IntegratedDeliveryModel(SimulationConfig(0.5))
        cls.model.run()

    def test_daily_reconciliation_and_permanent_merchants(self):
        model = self.model
        summary = model.summary()
        self.assertEqual(len(model.kpi_records), 1440)
        self.assertEqual(len(model.orders), 300)
        self.assertEqual(sum(summary[f"{s.value.lower()}_orders"] for s in OrderStatus), 300)
        self.assertEqual(summary["delivered_cumulative"], summary["delivered_orders"] + summary["completed_orders"])
        self.assertEqual(summary["active_orders"], summary["available_orders"] + summary["assigned_orders"] + summary["delivered_orders"])
        self.assertEqual(summary["movement_count"], sum(o.movement_count for o in model.orders.values()))
        self.assertEqual(summary["driver_revenue"], sum(o.driver_revenue for o in model.orders.values()))
        self.assertEqual(summary["merchant_revenue"], sum(c.settled_revenue for o in model.orders.values() for c in o.components))
        self.assertEqual(summary["settled_customer_payment"], summary["driver_revenue"] + summary["merchant_revenue"] + summary["platform_revenue"])
        self.assertEqual(summary["driver_profit"], summary["driver_revenue"] - summary["driver_operating_cost"])
        self.assertEqual((len(model.restaurants), len(model.stores)), (50, 15))
        self.assertTrue(all(m.pos is not None for m in model.merchants_by_id.values()))
        self.assertTrue(0 <= summary["driver_utilization"] <= 1)
        completed_integrated = sum(
            o.status == OrderStatus.COMPLETED and o.order_type == OrderType.INTEGRATED
            for o in model.orders.values()
        )
        self.assertEqual(summary["completed_food_service_units"], summary["completed_orders"])
        self.assertEqual(summary["completed_grocery_service_units"], completed_integrated)
        self.assertEqual(summary["completed_service_units"],
                         summary["completed_orders"] + completed_integrated)
        expected_productivity = summary["completed_service_units"] / (len(model.drivers) * 24)
        self.assertAlmostEqual(summary["service_units_per_driver_hour"], expected_productivity)
        self.assertAlmostEqual(summary["emission_intensity_per_service_unit"],
                               summary["emission_units"] / summary["completed_service_units"])
        self.assertEqual(summary["food_completion_rate"],
                         summary["completed_orders"] / summary["generated_orders"])
        self.assertEqual(summary["food_cancellation_rate"],
                         summary["cancelled_orders"] / summary["generated_orders"])
        self.assertEqual(summary["food_unfinished_rate"],
                         summary["active_orders"] / summary["generated_orders"])
        completed_ticks = [o.completed_tick - o.created_tick for o in model.orders.values()
                           if o.status == OrderStatus.COMPLETED]
        self.assertAlmostEqual(summary["mean_food_completion_ticks"],
                               float(np.mean(completed_ticks)))
        self.assertFalse(any(key.startswith("p90_") for key in summary))
        completed_integrated_orders = [o for o in model.orders.values()
                                       if o.order_type == OrderType.INTEGRATED and
                                       o.status == OrderStatus.COMPLETED]
        self.assertEqual(summary["integrated_completion_rate"],
                         len(completed_integrated_orders) / summary["integrated_orders"])
        for order in model.orders.values():
            self.assertEqual(order.settled, order.status == OrderStatus.COMPLETED)
            if order.status == OrderStatus.CANCELLED:
                self.assertEqual(order.cancelled_tick - order.created_tick, TIMEOUT)
                self.assertIsNone(order.assigned_tick)
            if order.grocery is not None and order.food.handed_over_tick is not None:
                self.assertLess(order.grocery.handed_over_tick, order.food.handed_over_tick)

    def test_seed_reproduces_complete_history_and_shuffle(self):
        other = IntegratedDeliveryModel(SimulationConfig(0.5))
        other.run()
        self.assertEqual(self.model.kpi_records, other.kpi_records)
        self.assertEqual(self.model.snapshot(), other.snapshot())
        self.assertEqual([self.model._order_row(o) for o in self.model.orders.values()],
                         [other._order_row(o) for o in other.orders.values()])
        sequences = {row["driver_activation_order"] for row in other.kpi_records}
        self.assertGreater(len(sequences), 1)
        self.assertEqual(other.last_driver_activation_order, self.model.last_driver_activation_order)

    def test_probability_endpoints(self):
        for probability, order_type in ((0, OrderType.FOOD_ONLY), (1, OrderType.INTEGRATED)):
            with self.subTest(probability=probability):
                model = IntegratedDeliveryModel(SimulationConfig(probability))
                model.run()
                self.assertEqual(len(model.orders), 300)
                self.assertTrue(all(o.order_type == order_type for o in model.orders.values()))

    def test_export_reconciliation_and_no_overwrite(self):
        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result"
            self.model.export_results(destination)
            self.assertEqual(len(list(destination.iterdir())), 7)
            with (destination / "orders.csv").open(newline="", encoding="utf-8") as handle:
                orders = list(csv.DictReader(handle))
            with (destination / "kpi_ticks.csv").open(newline="", encoding="utf-8") as handle:
                ticks = list(csv.DictReader(handle))
            self.assertEqual((len(orders), len(ticks)), (300, 1440))
            self.assertEqual((ticks[0]["tick"], ticks[-1]["tick"]), ("0", "1439"))
            summary = json.loads((destination / "summary.json").read_text())
            metadata = json.loads((destination / "metadata.json").read_text())
            self.assertTrue(summary["finished"])
            self.assertEqual(sum(Decimal(o["driver_revenue"]) for o in orders), Decimal(summary["driver_revenue"]))
            self.assertEqual(metadata["config"], {"p_integration": 0.5, "seed": 42,
                                                   "daily_customers": 300, "num_drivers": self.model.config.num_drivers,
                                                   "num_stores": self.model.config.num_stores,
                                                   "food_preparation_multiplier": 1.0,
                                                   "grocery_preparation_multiplier": 1.0,
                                                   "store_selection": "uniform"})
            self.assertEqual(len(metadata["demand_input"]["sha256"]), 64)
            self.assertEqual(metadata["policies"]["replications"], 1)
            with self.assertRaises(FileExistsError):
                self.model.export_results(destination)

    def test_empty_partial_export_and_detached_snapshot(self):
        model = fixture_model()
        before = model.snapshot()
        snapshot = model.snapshot()
        snapshot["drivers"][0]["x"] = -99
        snapshot["kpis"]["generated_orders"] = -1
        self.assertEqual(model.snapshot(), before)
        self.assertIsNone(before["kpis"]["mean_assignment_wait_ticks"])
        with TemporaryDirectory() as temporary:
            destination = model.export_results(Path(temporary) / "empty")
            summary = json.loads((destination / "summary.json").read_text())
            self.assertEqual(summary["run_status"], "PARTIAL")
            for label in ("restaurant", "store"):
                self.assertEqual(summary[f"{label}_preparing_cancelled_units"], 0)
                self.assertEqual(Decimal(summary[f"{label}_preparing_cancelled_product_value"]), 0)
            with (destination / "merchants.csv").open(newline="", encoding="utf-8") as handle:
                for merchant in csv.DictReader(handle):
                    self.assertEqual(int(merchant["preparing_cancelled_units"]), 0)
                    self.assertEqual(Decimal(merchant["preparing_cancelled_product_value"]), 0)
            for name in ("orders.csv", "components.csv", "kpi_ticks.csv"):
                with (destination / name).open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertTrue(reader.fieldnames)
                    if name == "kpi_ticks.csv":
                        self.assertIn("restaurant_preparing_cancelled_units", reader.fieldnames)
                        self.assertIn("store_preparing_cancelled_product_value", reader.fieldnames)
                    self.assertEqual(list(reader), [])
        self.assertEqual(model.tick_counter, 0)


if __name__ == "__main__":
    unittest.main()
