"""Food-grocery ABM: the PDF baseline plus the user's approved open choices.

Source: source/Integrated_Food_Grocery_Delivery_Model_Documentation.pdf.
No mechanisms are inherited from the legacy dispatcher model.
Run: python src/simulation_engine.py --p-integration 0.5
The probability in this example is an explicit experiment input, not a default.
Dependencies: Mesa 3.5.1, NumPy, pandas; Python >= 3.11.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from uuid import uuid4

import mesa
import numpy as np
import pandas as pd
from mesa.space import MultiGrid

if __package__:
    from .simulation_parameters import (
        ROOT, DEFAULT_DEMAND_CSV, GRID_WIDTH, GRID_HEIGHT, NUM_DRIVERS,
        NUM_RESTAURANTS, NUM_STORES, DAILY_CUSTOMERS, HORIZON, TICKS_PER_HOUR,
        ASSIGNMENT_TIMEOUT, HANDOVER_TIMEOUT,
        DEFAULT_SEED, P_INTEGRATION, KM_PER_STEP, COST_PER_STEP, EMISSION_PER_STEP,
        FOOD_PREPARATION, FOOD_ITEM_VALUE, GROCERY_PREPARATION,
        GROCERY_ITEM_VALUE, DELIVERY_MINIMUM_FEE, DELIVERY_FEE_PER_KM,
        TAX_RATE, DRIVER_PLATFORM_SHARE, DRIVER_DELIVERY_FEE_SHARE,
        MERCHANT_PLATFORM_SHARE,
    )
else:
    from simulation_parameters import (
        ROOT, DEFAULT_DEMAND_CSV, GRID_WIDTH, GRID_HEIGHT, NUM_DRIVERS,
        NUM_RESTAURANTS, NUM_STORES, DAILY_CUSTOMERS, HORIZON, TICKS_PER_HOUR,
        ASSIGNMENT_TIMEOUT, HANDOVER_TIMEOUT,
        DEFAULT_SEED, P_INTEGRATION, KM_PER_STEP, COST_PER_STEP, EMISSION_PER_STEP,
        FOOD_PREPARATION, FOOD_ITEM_VALUE, GROCERY_PREPARATION,
        GROCERY_ITEM_VALUE, DELIVERY_MINIMUM_FEE, DELIVERY_FEE_PER_KM,
        TAX_RATE, DRIVER_PLATFORM_SHARE, DRIVER_DELIVERY_FEE_SHARE,
        MERCHANT_PLATFORM_SHARE,
    )

SPECIFICATION_PDF = ROOT / "source" / "Integrated_Food_Grocery_Delivery_Model_Documentation.pdf"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "simulations"
ZERO = Decimal("0")
FINANCIAL_NOTE = (
    "PREPARING cancellations are recorded as merchant unit and product-value KPIs only. "
    "One food/grocery component is one unit; product value is item_value before tax/fees. "
    "These KPIs do not post payments or change revenue/profit. "
    "Tax and revenue equations are model assumptions from the specification."
)


class OrderStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    ASSIGNED = "ASSIGNED"
    DELIVERED = "DELIVERED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ComponentStatus(StrEnum):
    PREPARING = "PREPARING"
    READY = "READY"
    HANDED_OVER = "HANDED_OVER"
    CANCELLED = "CANCELLED"


class DriverState(StrEnum):
    IDLE = "IDLE"
    GROCERY_TRAVEL = "GROCERY_TRAVEL"
    GROCERY_PICKUP = "GROCERY_PICKUP"
    FOOD_TRAVEL = "FOOD_TRAVEL"
    FOOD_PICKUP = "FOOD_PICKUP"
    DELIVER = "DELIVER"


class MerchantState(StrEnum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class SimulationConfig:
    p_integration: float
    seed: int = DEFAULT_SEED
    demand_csv: Path = DEFAULT_DEMAND_CSV
    daily_customers: int = DAILY_CUSTOMERS
    num_drivers: int = NUM_DRIVERS
    num_stores: int = NUM_STORES
    food_preparation_multiplier: float = 1.0
    grocery_preparation_multiplier: float = 1.0
    store_selection: str = "uniform"

    def __post_init__(self):
        if (isinstance(self.p_integration, bool)
                or not isinstance(self.p_integration, Real)
                or not math.isfinite(self.p_integration)
                or not 0 <= self.p_integration <= 1):
            raise ValueError("p_integration must be a finite number between 0 and 1.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        for name in ("daily_customers", "num_drivers", "num_stores"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        for name in ("food_preparation_multiplier", "grocery_preparation_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
            object.__setattr__(self, name, float(value))
        if self.store_selection not in ("uniform", "nearest_restaurant", "nearest_customer"):
            raise ValueError("store_selection must be uniform, nearest_restaurant or nearest_customer.")
        object.__setattr__(self, "p_integration", float(self.p_integration))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "demand_csv", Path(self.demand_csv).expanduser().resolve())


@dataclass(frozen=True)
class ScenarioOrder:
    created_tick: int
    customer_pos: tuple[int, int]
    integration_uniform: float
    restaurant_index: int
    store_uniform: float
    food_preparation: int
    food_item_value: int
    grocery_preparation: int
    grocery_item_value: int


@dataclass(frozen=True)
class ExperimentScenario:
    """Exogenous day shared by all policies in one paired replication."""
    seed: int
    demand_sha256: str
    driver_positions: tuple[tuple[int, int], ...]
    restaurant_positions: tuple[tuple[int, int], ...]
    store_positions: tuple[tuple[int, int], ...]
    demand_schedule: tuple[int, ...]
    orders: tuple[ScenarioOrder, ...]
    driver_permutations: tuple[tuple[int, ...], ...]

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_experiment_scenario(seed: int, demand_csv: str | Path = DEFAULT_DEMAND_CSV,
                              daily_customers: int = DAILY_CUSTOMERS,
                              num_drivers: int = NUM_DRIVERS,
                              num_stores: int = NUM_STORES) -> ExperimentScenario:
    """Pre-draw all exogenous values, including potential grocery for food-only orders."""
    config = SimulationConfig(p_integration=0, seed=seed, demand_csv=demand_csv,
                              daily_customers=daily_customers, num_drivers=num_drivers,
                              num_stores=num_stores)
    csv_bytes = config.demand_csv.read_bytes()
    weights = load_hourly_weights(csv_bytes)
    position_seed, demand_seed, order_seed, activation_seed = np.random.SeedSequence(seed).spawn(4)
    # Changing fleet size must not relocate Restaurants or Stores in the same day block.
    driver_seed, restaurant_seed, store_seed = position_seed.spawn(3)
    demand_rng = np.random.default_rng(demand_seed)
    order_rng = np.random.default_rng(order_seed)
    activation_rng = np.random.default_rng(activation_seed)

    def positions(count: int, child_seed) -> tuple[tuple[int, int], ...]:
        rng = np.random.default_rng(child_seed)
        return tuple((int(rng.integers(GRID_WIDTH)),
                      int(rng.integers(GRID_HEIGHT))) for _ in range(count))

    def nonnegative_normal(mean: float, std: float) -> int:
        return math.ceil(max(0, order_rng.normal(mean, std)))

    _, schedule = build_demand_schedule(weights, demand_rng, config.daily_customers)
    orders = []
    for tick, count in enumerate(schedule):
        for _ in range(count):
            orders.append(ScenarioOrder(
                created_tick=tick,
                customer_pos=(int(order_rng.integers(GRID_WIDTH)),
                              int(order_rng.integers(GRID_HEIGHT))),
                integration_uniform=float(order_rng.random()),
                restaurant_index=int(order_rng.integers(NUM_RESTAURANTS)),
                store_uniform=float(order_rng.random()),
                food_preparation=nonnegative_normal(*FOOD_PREPARATION),
                food_item_value=nonnegative_normal(*FOOD_ITEM_VALUE),
                grocery_preparation=nonnegative_normal(*GROCERY_PREPARATION),
                grocery_item_value=nonnegative_normal(*GROCERY_ITEM_VALUE),
            ))
    return ExperimentScenario(
        seed=seed,
        demand_sha256=hashlib.sha256(csv_bytes).hexdigest(),
        driver_positions=positions(config.num_drivers, driver_seed),
        restaurant_positions=positions(NUM_RESTAURANTS, restaurant_seed),
        store_positions=positions(config.num_stores, store_seed),
        demand_schedule=tuple(schedule),
        orders=tuple(orders),
        driver_permutations=tuple(tuple(int(i) for i in activation_rng.permutation(config.num_drivers))
                                  for _ in range(HORIZON)),
    )


def load_hourly_weights(csv_bytes: bytes) -> np.ndarray:
    """PDF §1.4; approved mapping: average all daily avg_demand rows by hour."""
    try:
        frame = pd.read_csv(io.BytesIO(csv_bytes))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Cannot read demand CSV: {exc}") from exc
    required = {"hour", "avg_demand"}
    if not required.issubset(frame.columns):
        raise ValueError("Demand CSV requires columns: hour, avg_demand.")
    hours = pd.to_numeric(frame["hour"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(frame["avg_demand"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(hours).all() or not np.equal(hours, np.floor(hours)).all():
        raise ValueError("Demand hours must be finite integers from 0 to 23.")
    if set(hours) != set(range(24)):
        raise ValueError("Demand CSV must contain every hour from 0 to 23, with no other hours.")
    if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError("Demand weights must be finite, nonnegative, with a positive total.")
    # Scale before aggregation to avoid overflow for large but finite input weights.
    frame = pd.DataFrame({"hour": hours.astype(int), "weight": weights / weights.max()})
    hourly = frame.groupby("hour")["weight"].mean().reindex(range(24)).to_numpy()
    return hourly / hourly.sum()


def build_demand_schedule(weights: np.ndarray, rng: np.random.Generator,
                          daily_customers: int = DAILY_CUSTOMERS) -> tuple[list[int], list[int]]:
    """PDF §1.4: exact daily total, stable hour ties, seeded minute remainder."""
    if isinstance(daily_customers, bool) or not isinstance(daily_customers, Integral) or daily_customers <= 0:
        raise ValueError("daily_customers must be a positive integer.")
    raw = daily_customers * weights
    hourly = np.floor(raw).astype(int)
    priority = sorted(range(24), key=lambda hour: (-(raw[hour] - hourly[hour]), hour))
    hourly[priority[:daily_customers - int(hourly.sum())]] += 1
    schedule = np.zeros(HORIZON, dtype=int)
    for hour, count in enumerate(hourly):
        base, remainder = divmod(int(count), 60)
        start = hour * 60
        schedule[start:start + 60] = base
        if remainder:
            schedule[start + rng.choice(60, size=remainder, replace=False)] += 1
    return hourly.tolist(), schedule.tolist()


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Minimum orthogonal steps on the unobstructed grid (PDF §1.7–1.8)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def merchant_revenue(item_value: int) -> Decimal:
    """Merchant gross proceeds less the common modeled platform share."""
    value = Decimal(item_value)
    gross_proceeds = value * (1 + TAX_RATE)
    return gross_proceeds * (1 - MERCHANT_PLATFORM_SHARE)


def elapsed(end: int | None, start: int | None) -> int | None:
    return None if end is None or start is None else end - start


def mean_observed(values) -> float | None:
    observed = [value for value in values if value is not None]
    return float(sum(observed) / len(observed)) if observed else None


@dataclass
class MerchantOrderComponent:
    order_id: int
    merchant_id: int
    kind: str
    created_tick: int
    preparation_duration: int
    item_value: int
    status: ComponentStatus = ComponentStatus.PREPARING
    ready_tick: int | None = None
    handed_over_tick: int | None = None
    cancelled_tick: int | None = None
    cancellation_event: str | None = None
    settled_revenue: Decimal = ZERO

    @property
    def planned_ready_tick(self) -> int:
        return self.created_tick + self.preparation_duration


@dataclass
class Order:
    order_id: int
    customer_id: int
    customer_pos: tuple[int, int]
    order_type: str  # "FOOD_ONLY" or "INTEGRATED"
    created_tick: int
    food: MerchantOrderComponent
    grocery: MerchantOrderComponent | None = None
    status: OrderStatus = OrderStatus.AVAILABLE
    assigned_driver_id: int | None = None  # Historical identity retained after delivery.
    assigned_tick: int | None = None
    delivered_tick: int | None = None
    completed_tick: int | None = None
    cancelled_tick: int | None = None
    grocery_arrival_tick: int | None = None
    food_arrival_tick: int | None = None
    grocery_pickup_wait_ticks: int = 0
    food_pickup_wait_ticks: int = 0
    first_pickup_moves: int = 0
    store_to_restaurant_moves: int = 0
    restaurant_to_customer_moves: int = 0
    billable_distance_km: Decimal = ZERO
    tax: Decimal = ZERO
    delivery_fee: Decimal = ZERO
    customer_payment: Decimal = ZERO
    settled: bool = False
    settlement_tick: int | None = None
    driver_revenue: Decimal = ZERO
    platform_revenue: Decimal = ZERO

    @property
    def components(self) -> tuple[MerchantOrderComponent, ...]:
        return (self.food,) if self.grocery is None else (self.food, self.grocery)

    @property
    def is_integrated(self) -> bool:
        """True if this order contains a grocery component."""
        return self.grocery is not None

    @property
    def movement_count(self) -> int:
        return self.first_pickup_moves + self.store_to_restaurant_moves + self.restaurant_to_customer_moves

    def time_metrics(self) -> dict:
        return {
            "assignment_wait_ticks": elapsed(self.assigned_tick, self.created_tick),
            "delivery_service_ticks": elapsed(self.delivered_tick, self.created_tick),
            "acknowledgement_delay_ticks": elapsed(self.completed_tick, self.delivered_tick),
            "completion_total_ticks": elapsed(self.completed_tick, self.created_tick),
            "cancellation_wait_ticks": elapsed(self.cancelled_tick, self.created_tick),
            # In-progress waits are retained separately; averages use finished pickups.
            "food_pickup_wait_ticks": (self.food_pickup_wait_ticks
                                       if self.food.handed_over_tick is not None else None),
            "grocery_pickup_wait_ticks": (self.grocery_pickup_wait_ticks if self.grocery is not None
                                          and self.grocery.handed_over_tick is not None else None),
        }


class Customer(mesa.Agent):
    def __init__(self, model: IntegratedDeliveryModel):
        super().__init__(model)
        self.state = "WAITING"  # WAITING, IN_SERVICE, EXIT
        self.order_id: int | None = None

    def step(self):
        """PDF §1.13-1.14: observe only shared order status and assignment."""
        order = self.model.orders[self.order_id]
        tick = self.model.tick_counter
        assignment_expired = (
            order.status == OrderStatus.AVAILABLE
            and tick - order.created_tick >= ASSIGNMENT_TIMEOUT
        )
        handover_expired = (
            order.status == OrderStatus.ASSIGNED
            and order.assigned_tick is not None
            and tick - order.assigned_tick >= HANDOVER_TIMEOUT
            and any(component.status != ComponentStatus.HANDED_OVER
                    for component in order.components)
        )
        if assignment_expired or handover_expired:
            order.status = OrderStatus.CANCELLED
            order.cancelled_tick = tick
            self.state = "EXIT"
        elif order.assigned_driver_id is not None:
            self.state = "IN_SERVICE"
        if order.status == OrderStatus.DELIVERED:
            order.status = OrderStatus.COMPLETED
            order.completed_tick = tick
            self.state = "EXIT"


class Merchant(mesa.Agent):
    kind: str
    pickup_state: DriverState

    def __init__(self, model: IntegratedDeliveryModel):
        super().__init__(model)
        self.components: dict[int, MerchantOrderComponent] = {}
        self.revenue = ZERO

    @property
    def state(self) -> MerchantState:
        active = any(c.status in (ComponentStatus.PREPARING, ComponentStatus.READY)
                     for c in self.components.values())
        return MerchantState.ACTIVE if active else MerchantState.IDLE

    @property
    def cancellation_kpis(self) -> dict:
        """Terminal component events make PREPARING-only counts idempotent."""
        cancelled = [c for c in self.components.values()
                     if c.cancellation_event == "PREPARATION_STOPPED"]
        return {
            "preparing_cancelled_units": len(cancelled),
            "preparing_cancelled_product_value": sum((Decimal(c.item_value) for c in cancelled), ZERO),
        }

    def step(self):
        """PDF §1.10-1.12,1.14: parallel component processing, no capacity limit."""
        for component in self.components.values():
            if component.status in (ComponentStatus.HANDED_OVER, ComponentStatus.CANCELLED):
                continue
            order = self.model.orders[component.order_id]
            if order.status == OrderStatus.CANCELLED:
                self._handle_timeout(component)
                continue
            if (component.status == ComponentStatus.PREPARING
                    and self.model.tick_counter >= component.planned_ready_tick):
                component.status = ComponentStatus.READY
                component.ready_tick = self.model.tick_counter
            if self._can_handover(order, component):
                component.status = ComponentStatus.HANDED_OVER
                component.handed_over_tick = self.model.tick_counter

    def _can_handover(self, order: Order, component: MerchantOrderComponent) -> bool:
        driver = self.model.drivers_by_id.get(order.assigned_driver_id)
        return (
            component.status == ComponentStatus.READY
            and order.status == OrderStatus.ASSIGNED
            and driver is not None
            and driver.current_order_id == order.order_id
            and driver.pos == self.pos
            and driver.state == self.pickup_state
            and (self.kind == "GROCERY" or order.grocery is None
                 or order.grocery.status == ComponentStatus.HANDED_OVER)
        )

    def _handle_timeout(self, component: MerchantOrderComponent):
        if component.status == ComponentStatus.READY:
            component.cancellation_event = (
                "PREPARED_FOOD_DISPOSAL" if self.kind == "FOOD" else "GROCERY_CANCELLATION_RETURN"
            )
        else:
            component.cancellation_event = "PREPARATION_STOPPED"
        component.status = ComponentStatus.CANCELLED
        component.cancelled_tick = self.model.tick_counter


class Restaurant(Merchant):
    kind = "FOOD"
    pickup_state = DriverState.FOOD_PICKUP


class Store(Merchant):
    kind = "GROCERY"
    pickup_state = DriverState.GROCERY_PICKUP


class Driver(mesa.Agent):
    def __init__(self, model: IntegratedDeliveryModel):
        super().__init__(model)
        self.state = DriverState.IDLE
        self.current_order_id: int | None = None
        self.movement_count = 0
        self.busy_ticks = 0
        self.revenue = ZERO

    @property
    def target_pos(self) -> tuple[int, int] | None:
        route = self.remaining_route()
        return route[0]["pos"] if route else None

    def remaining_route(self) -> list[dict]:
        if self.current_order_id is None:
            return []
        order = self.model.orders[self.current_order_id]
        stops = []
        if self.state in (DriverState.GROCERY_TRAVEL, DriverState.GROCERY_PICKUP):
            stops.append({"action": "GROCERY_PICKUP",
                          "pos": tuple(self.model.merchants_by_id[order.grocery.merchant_id].pos)})
        if self.state != DriverState.DELIVER:
            stops.append({"action": "FOOD_PICKUP",
                          "pos": tuple(self.model.merchants_by_id[order.food.merchant_id].pos)})
        stops.append({"action": "DELIVER", "pos": order.customer_pos})
        return stops

    def _claim_order(self):
        """PDF §1.7: driver-owned matching, distance/FIFO/order-ID ranking."""
        if self.current_order_id is not None:
            raise RuntimeError("An idle driver cannot hold an operational order.")
        candidates = [o for o in self.model.active_orders.values() if o.status == OrderStatus.AVAILABLE]
        if not candidates:
            return
        def rank(order):
            first = order.grocery if order.grocery is not None else order.food
            pos = self.model.merchants_by_id[first.merchant_id].pos
            return manhattan(self.pos, pos), order.created_tick, order.order_id
        order = min(candidates, key=rank)
        order.status = OrderStatus.ASSIGNED
        order.assigned_driver_id = self.unique_id
        order.assigned_tick = self.model.tick_counter
        self.current_order_id = order.order_id
        self.state = DriverState.GROCERY_TRAVEL if order.grocery is not None else DriverState.FOOD_TRAVEL

    def _move(self, order: Order, state: DriverState):
        """PDF §1.8,1.17: choose an orthogonal step by squared Euclidean score."""
        target = self.target_pos
        if target is None:
            raise RuntimeError("A travelling driver must have a target.")
        x, y = self.pos
        candidates = [(px, py) for px, py in ((x - 1, y), (x + 1, y), (x, y - 1),
                                             (x, y + 1), (x, y))
                      if 0 <= px < GRID_WIDTH and 0 <= py < GRID_HEIGHT]
        best = min(candidates, key=lambda p: ((target[0] - p[0]) ** 2 + (target[1] - p[1]) ** 2, p))
        if best != self.pos:
            self.model.grid.move_agent(self, best)
            self.movement_count += 1
            if state == DriverState.DELIVER:
                order.restaurant_to_customer_moves += 1
            elif state == DriverState.FOOD_TRAVEL and order.grocery is not None:
                order.store_to_restaurant_moves += 1
            else:
                order.first_pickup_moves += 1

    def step(self):
        """PDF Algorithm 1: execute exactly the initial state branch."""
        initial_state = self.state
        if initial_state == DriverState.IDLE:
            self._claim_order()
            if self.current_order_id is not None:
                self.busy_ticks += 1
            return
        order = self.model.orders[self.current_order_id]
        # Customer deadlines run before Driver activation. Do not move or accrue
        # another busy tick for an order cancelled earlier in the same tick.
        if order.status == OrderStatus.CANCELLED:
            return
        self.busy_ticks += 1
        if initial_state in (DriverState.GROCERY_TRAVEL, DriverState.FOOD_TRAVEL):
            self._move(order, initial_state)
            if self.pos == self.target_pos:
                if initial_state == DriverState.GROCERY_TRAVEL:
                    self.state = DriverState.GROCERY_PICKUP
                    order.grocery_arrival_tick = self.model.tick_counter
                else:
                    self.state = DriverState.FOOD_PICKUP
                    order.food_arrival_tick = self.model.tick_counter
        elif initial_state == DriverState.GROCERY_PICKUP:
            if order.grocery.status == ComponentStatus.HANDED_OVER:
                self.state = DriverState.FOOD_TRAVEL
            else:
                order.grocery_pickup_wait_ticks += 1
        elif initial_state == DriverState.FOOD_PICKUP:
            if order.food.status == ComponentStatus.HANDED_OVER:
                self.state = DriverState.DELIVER
            else:
                order.food_pickup_wait_ticks += 1
        elif initial_state == DriverState.DELIVER:
            self._move(order, initial_state)
            if self.pos == order.customer_pos:
                if any(c.status != ComponentStatus.HANDED_OVER for c in order.components):
                    raise RuntimeError("Delivery requires every component to be handed over.")
                order.status = OrderStatus.DELIVERED
                order.delivered_tick = self.model.tick_counter
                self.state = DriverState.IDLE
                # The order retains assigned_driver_id for next-tick settlement.
                self.current_order_id = None


class IntegratedDeliveryModel(mesa.Model):
    def __init__(self, config: SimulationConfig, scenario: ExperimentScenario | None = None):
        super().__init__(rng=config.seed)
        self.config = config
        self.experiment_scenario = scenario
        self.grid = MultiGrid(GRID_WIDTH, GRID_HEIGHT, torus=False)
        self.tick_counter = 0  # Next domain tick; never use Mesa's internal clock here.
        self.orders: dict[int, Order] = {}
        self.active_orders: dict[int, Order] = {}
        self.customers: dict[int, Customer] = {}
        self.kpi_records: list[dict] = []
        self.last_driver_activation_order: tuple[int, ...] = ()
        self.platform_revenue = ZERO
        csv_bytes = config.demand_csv.read_bytes()
        self.demand_sha256 = hashlib.sha256(csv_bytes).hexdigest()
        # Hash of the parameter file for reproducibility checks (e.g., visualizer reset).
        param_path = Path(__file__).with_name("simulation_parameters.py")
        self.parameters_sha256 = hashlib.sha256(param_path.read_bytes()).hexdigest()
        self.hourly_weights = load_hourly_weights(csv_bytes)
        if scenario is not None:
            self._validate_scenario(scenario)
        self.drivers = self._initialize_agents(
            Driver, config.num_drivers, scenario.driver_positions if scenario else None)
        self.restaurants = self._initialize_agents(
            Restaurant, NUM_RESTAURANTS, scenario.restaurant_positions if scenario else None)
        self.stores = self._initialize_agents(
            Store, config.num_stores,
            scenario.store_positions[:config.num_stores] if scenario else None)
        self.drivers_by_id = {a.unique_id: a for a in self.drivers}
        self.merchants_by_id = {a.unique_id: a for a in (*self.restaurants, *self.stores)}
        if scenario is None:
            self.hourly_demand, self.demand_schedule = build_demand_schedule(
                self.hourly_weights, self.rng, config.daily_customers)
        else:
            self.demand_schedule = list(scenario.demand_schedule)
            self.hourly_demand = [sum(self.demand_schedule[h * 60:(h + 1) * 60]) for h in range(24)]
        self.specification_sha256 = (hashlib.sha256(SPECIFICATION_PDF.read_bytes()).hexdigest()
                                     if SPECIFICATION_PDF.is_file() else None)

    def _validate_scenario(self, scenario: ExperimentScenario) -> None:
        if scenario.seed != self.config.seed or scenario.demand_sha256 != self.demand_sha256:
            raise ValueError("Scenario seed or demand CSV hash does not match configuration.")
        if (len(scenario.driver_positions) != self.config.num_drivers
                or len(scenario.restaurant_positions) != NUM_RESTAURANTS
                or len(scenario.store_positions) < self.config.num_stores
                or len(scenario.demand_schedule) != HORIZON
                or sum(scenario.demand_schedule) != self.config.daily_customers
                or len(scenario.orders) != self.config.daily_customers
                or len(scenario.driver_permutations) != HORIZON):
            raise ValueError("Scenario dimensions do not match the model configuration.")
        ticks = [t for t, n in enumerate(scenario.demand_schedule) for _ in range(n)]
        if any(o.created_tick != t for o, t in zip(scenario.orders, ticks)):
            raise ValueError("Scenario order creation ticks do not match demand schedule.")
        if any(sorted(p) != list(range(self.config.num_drivers)) for p in scenario.driver_permutations):
            raise ValueError("Scenario contains an invalid driver permutation.")

    @property
    def finished(self) -> bool:
        return self.tick_counter >= HORIZON

    def _random_position(self) -> tuple[int, int]:
        return int(self.rng.integers(GRID_WIDTH)), int(self.rng.integers(GRID_HEIGHT))

    def _initialize_agents(self, agent_class, count, positions=None):
        agents = []
        for index in range(count):
            agent = agent_class(self)
            self.grid.place_agent(agent, positions[index] if positions is not None else self._random_position())
            agents.append(agent)
        return agents

    def _normal_nonnegative_integer(self, mean: float, std: float) -> int:
        return math.ceil(max(0, self.rng.normal(mean, std)))

    def _food_preparation_duration(self) -> int:
        """Draw a whole-minute food-ready offset from the fitted Meituan proxy."""
        return self._normal_nonnegative_integer(*FOOD_PREPARATION)

    def _select_store(self, random_index: int, restaurant_pos: tuple[int, int],
                      customer_pos: tuple[int, int]):
        """Select a store using the configured policy.

        The exogenous `random_index` (derived from `store_uniform`) is used as:
        - Direct index for "uniform" policy.
        - Tiebreaker for "nearest_restaurant" and "nearest_customer" policies
          to preserve common random numbers across conditions.
        """
        if self.config.store_selection == "uniform":
            return self.stores[random_index]

        target = (restaurant_pos
                  if self.config.store_selection == "nearest_restaurant"
                  else customer_pos)

        # Compute distances to all stores.
        distances = [(manhattan(store.pos, target), idx) for idx, store in enumerate(self.stores)]
        min_dist = min(d for d, _ in distances)

        # Among stores at minimum distance, use random_index as tiebreaker.
        # Map random_index to a store index within the tied set.
        tied_indices = [idx for d, idx in distances if d == min_dist]
        chosen_idx = tied_indices[random_index % len(tied_indices)]
        return self.stores[chosen_idx]

    def _create_order(self) -> Order:
        """PDF §1.5,1.10,1.15: one customer, one order, independently drawn components."""
        customer = Customer(self)
        exogenous = (self.experiment_scenario.orders[len(self.orders)]
                     if self.experiment_scenario is not None else None)
        self.grid.place_agent(customer, exogenous.customer_pos if exogenous else self._random_position())
        if exogenous is None:
            integrated = bool(self.rng.binomial(1, self.config.p_integration))
            restaurant = self.restaurants[int(self.rng.integers(len(self.restaurants)))]
            store = self._select_store(int(self.rng.integers(len(self.stores))),
                                       restaurant.pos, customer.pos) if integrated else None
            food_preparation = self._food_preparation_duration()
            food_item_value = self._normal_nonnegative_integer(*FOOD_ITEM_VALUE)
        else:
            integrated = exogenous.integration_uniform < self.config.p_integration
            restaurant = self.restaurants[exogenous.restaurant_index]
            store_index = min(int(exogenous.store_uniform * len(self.stores)), len(self.stores) - 1)
            store = self._select_store(store_index, restaurant.pos,
                                       customer.pos) if integrated else None
            food_preparation = exogenous.food_preparation
            food_item_value = exogenous.food_item_value
        order_id = len(self.orders) + 1
        food = MerchantOrderComponent(order_id, restaurant.unique_id, "FOOD", self.tick_counter,
                                      math.ceil(food_preparation * self.config.food_preparation_multiplier),
                                      food_item_value)
        grocery = None
        if store is not None:
            grocery = MerchantOrderComponent(order_id, store.unique_id, "GROCERY", self.tick_counter,
                                             math.ceil((exogenous.grocery_preparation if exogenous else
                                                        self._normal_nonnegative_integer(*GROCERY_PREPARATION))
                                                       * self.config.grocery_preparation_multiplier),
                                             (exogenous.grocery_item_value if exogenous else
                                              self._normal_nonnegative_integer(*GROCERY_ITEM_VALUE)))
            store.components[order_id] = grocery
        restaurant.components[order_id] = food
        order = Order(order_id, customer.unique_id, tuple(customer.pos),
                      "INTEGRATED" if integrated else "FOOD_ONLY",
                      self.tick_counter, food, grocery)
        billable_steps = manhattan(restaurant.pos, customer.pos)
        if store is not None:
            billable_steps += manhattan(store.pos, restaurant.pos)
        order.billable_distance_km = billable_steps * KM_PER_STEP
        order.delivery_fee = max(
            DELIVERY_MINIMUM_FEE,
            DELIVERY_FEE_PER_KM * order.billable_distance_km,
        )
        item_total = sum(c.item_value for c in order.components)
        order.tax = TAX_RATE * item_total
        order.customer_payment = item_total + order.tax + order.delivery_fee
        customer.order_id = order_id
        self.customers[customer.unique_id] = customer
        self.orders[order_id] = order
        self.active_orders[order_id] = order
        return order

    def _settle_order(self, order: Order):
        """PDF §1.18: post revenue once, exclusively on COMPLETED."""
        if order.status != OrderStatus.COMPLETED or order.settled:
            return
        driver = self.drivers_by_id[order.assigned_driver_id]
        order.driver_revenue = order.delivery_fee * DRIVER_DELIVERY_FEE_SHARE
        driver.revenue += order.driver_revenue
        merchant_total = ZERO
        for component in order.components:
            component.settled_revenue = merchant_revenue(component.item_value)
            self.merchants_by_id[component.merchant_id].revenue += component.settled_revenue
            merchant_total += component.settled_revenue
        order.platform_revenue = order.customer_payment - order.driver_revenue - merchant_total
        self.platform_revenue += order.platform_revenue
        order.settled = True
        order.settlement_tick = self.tick_counter

    def _resolve_orders(self):
        for order in list(self.active_orders.values()):
            if order.status not in (OrderStatus.COMPLETED, OrderStatus.CANCELLED):
                continue
            self._settle_order(order)
            driver = self.drivers_by_id.get(order.assigned_driver_id)
            # Never clear a new assignment while resolving a previous delivery.
            if driver is not None and driver.current_order_id == order.order_id:
                driver.current_order_id = None
                driver.state = DriverState.IDLE
            customer = self.customers.pop(order.customer_id)
            self.grid.remove_agent(customer)
            customer.remove()
            del self.active_orders[order.order_id]

    def step(self):
        """Algorithm 1 / §1.1: domain ticks 0..1439, no post-horizon drain."""
        if self.finished:
            return
        tick = self.tick_counter
        for _ in range(self.demand_schedule[tick]):
            self._create_order()
        for customer in list(self.customers.values()):
            customer.step()
        for merchant in (*self.stores, *self.restaurants):
            merchant.step()
        permutation = (self.experiment_scenario.driver_permutations[tick]
                       if self.experiment_scenario is not None
                       else self.rng.permutation(len(self.drivers)))
        activation = [self.drivers[int(i)] for i in permutation]
        self.last_driver_activation_order = tuple(driver.unique_id for driver in activation)
        for driver in activation:
            driver.step()
        self._resolve_orders()
        self.kpi_records.append(self._calculate_kpis(tick, tick + 1))
        self.tick_counter += 1
        self.running = not self.finished

    def run(self) -> dict:
        while not self.finished:
            self.step()
        return self.summary()

    def _calculate_kpis(self, tick: int | None, elapsed_ticks: int) -> dict:
        """PDF §1.19 and approved timestamp/utilization definitions."""
        orders = list(self.orders.values())
        components = [c for order in orders for c in order.components]
        statuses = Counter(order.status for order in orders)
        types = Counter(order.order_type for order in orders)
        completed_orders = [order for order in orders if order.status == OrderStatus.COMPLETED]
        integrated_orders = [order for order in orders if order.is_integrated]
        completed_integrated_orders = [order for order in integrated_orders
                                       if order.status == OrderStatus.COMPLETED]
        completed_food_service_units = len(completed_orders)
        completed_grocery_service_units = sum(
            order.is_integrated for order in completed_orders
        )
        completed_service_units = completed_food_service_units + completed_grocery_service_units
        moves = sum(d.movement_count for d in self.drivers)
        elapsed_driver_hours = len(self.drivers) * elapsed_ticks / TICKS_PER_HOUR
        busy_driver_hours = sum(d.busy_ticks for d in self.drivers) / TICKS_PER_HOUR
        travel_distance_km = moves * KM_PER_STEP
        total_emission_units = moves * EMISSION_PER_STEP
        driver_revenue_total = sum((d.revenue for d in self.drivers), ZERO)
        restaurant_revenue = sum((m.revenue for m in self.restaurants), ZERO)
        store_revenue = sum((m.revenue for m in self.stores), ZERO)
        metrics = {
            "tick": tick,
            "elapsed_ticks": elapsed_ticks,
            "generated_orders": len(orders),
            "food_only_orders": types.get("FOOD_ONLY", 0),
            "integrated_orders": types.get("INTEGRATED", 0),
            **{f"{status.value.lower()}_orders": statuses[status] for status in OrderStatus},
            "delivered_cumulative": sum(o.delivered_tick is not None for o in orders),
            "active_orders": len(self.active_orders),
            "active_customers": len(self.customers),
            "completed_food_service_units": completed_food_service_units,
            "completed_grocery_service_units": completed_grocery_service_units,
            "completed_service_units": completed_service_units,
            "service_units_per_driver_hour": (completed_service_units / elapsed_driver_hours
                                               if elapsed_driver_hours else None),
            "service_units_per_busy_driver_hour": (completed_service_units / busy_driver_hours
                                                    if busy_driver_hours else None),
            "service_units_per_km": (completed_service_units / float(travel_distance_km)
                                     if travel_distance_km else None),
            "emission_intensity_per_service_unit": (total_emission_units / completed_service_units
                                                     if completed_service_units else None),
            "movement_count": moves,
            "travel_distance_km": travel_distance_km,
            "driver_operating_cost": moves * COST_PER_STEP,
            "emission_units": total_emission_units,
            "driver_revenue": driver_revenue_total,
            "restaurant_revenue": restaurant_revenue,
            "store_revenue": store_revenue,
            "merchant_revenue": restaurant_revenue + store_revenue,
            "platform_revenue": self.platform_revenue,
            "driver_profit": driver_revenue_total - moves * COST_PER_STEP,
            "settled_customer_payment": sum((o.customer_payment for o in orders if o.settled), ZERO),
            "driver_busy_ticks": sum(d.busy_ticks for d in self.drivers),
            "driver_utilization": (sum(d.busy_ticks for d in self.drivers) / (len(self.drivers) * elapsed_ticks)
                                   if elapsed_ticks and self.drivers else None),
            "food_pickup_wait_ticks_so_far": sum(o.food_pickup_wait_ticks for o in orders),
            "grocery_pickup_wait_ticks_so_far": sum(o.grocery_pickup_wait_ticks for o in orders),
            "food_disposal_events": sum(c.cancellation_event == "PREPARED_FOOD_DISPOSAL" for c in components),
            "grocery_return_events": sum(c.cancellation_event == "GROCERY_CANCELLATION_RETURN" for c in components),
            "preparation_stopped_events": sum(c.cancellation_event == "PREPARATION_STOPPED" for c in components),
            "driver_activation_order": ",".join(map(str, self.last_driver_activation_order)),
        }
        food_completion_ticks = [elapsed(o.completed_tick, o.created_tick) for o in completed_orders]
        food_pickup_waits = [o.food_pickup_wait_ticks for o in orders
                             if o.food.handed_over_tick is not None]
        food_post_pickup_ticks = [elapsed(o.delivered_tick, o.food.handed_over_tick)
                                  for o in completed_orders]
        integrated_completion_ticks = [elapsed(o.completed_tick, o.created_tick)
                                        for o in completed_integrated_orders]
        integrated_food_pickup_waits = [o.food_pickup_wait_ticks for o in integrated_orders
                                        if o.food.handed_over_tick is not None]
        integrated_grocery_pickup_waits = [o.grocery_pickup_wait_ticks for o in integrated_orders
                                           if o.grocery is not None and o.grocery.handed_over_tick is not None]
        metrics.update({
            "food_completion_rate": len(completed_orders) / len(orders) if orders else None,
            "food_cancellation_rate": statuses[OrderStatus.CANCELLED] / len(orders) if orders else None,
            "food_unfinished_rate": len(self.active_orders) / len(orders) if orders else None,
            "mean_food_completion_ticks": mean_observed(food_completion_ticks),
            "mean_food_pickup_wait_ticks": mean_observed(food_pickup_waits),
            "mean_food_post_pickup_delivery_ticks": mean_observed(food_post_pickup_ticks),
            "integrated_completion_rate": (len(completed_integrated_orders) / len(integrated_orders)
                                            if integrated_orders else None),
            "integrated_cancellation_rate": (
                sum(o.status == OrderStatus.CANCELLED for o in integrated_orders) / len(integrated_orders)
                if integrated_orders else None),
            "integrated_unfinished_rate": (
                sum(o.order_id in self.active_orders for o in integrated_orders) / len(integrated_orders)
                if integrated_orders else None),
            "mean_integrated_completion_ticks": mean_observed(integrated_completion_ticks),
            "mean_integrated_food_pickup_wait_ticks": mean_observed(integrated_food_pickup_waits),
            "mean_integrated_grocery_pickup_wait_ticks": mean_observed(integrated_grocery_pickup_waits),
        })
        for state in DriverState:
            metrics[f"drivers_{state.value.lower()}"] = sum(d.state == state for d in self.drivers)
        for kind in ("FOOD", "GROCERY"):
            for status in ComponentStatus:
                metrics[f"{kind.lower()}_{status.value.lower()}_components"] = sum(
                    c.kind == kind and c.status == status for c in components
                )
        for label, merchants in (("restaurant", self.restaurants), ("store", self.stores)):
            metrics[f"active_{label}s"] = sum(m.state == MerchantState.ACTIVE for m in merchants)
            metrics[f"{label}_preparing_cancelled_units"] = sum(
                m.cancellation_kpis["preparing_cancelled_units"] for m in merchants)
            metrics[f"{label}_preparing_cancelled_product_value"] = sum(
                (m.cancellation_kpis["preparing_cancelled_product_value"] for m in merchants), ZERO)
        time_rows = [order.time_metrics() for order in orders]
        time_keys = (
            "assignment_wait_ticks", "delivery_service_ticks", "acknowledgement_delay_ticks",
            "completion_total_ticks", "cancellation_wait_ticks", "food_pickup_wait_ticks",
            "grocery_pickup_wait_ticks",
        )
        for key in time_keys:
            values = [row[key] for row in time_rows]
            metrics[f"mean_{key}"] = mean_observed(values)
            metrics[f"observed_{key}_count"] = sum(value is not None for value in values)
        return metrics

    def _order_row(self, order: Order) -> dict:
        row = {field.name: getattr(order, field.name) for field in fields(Order)
               if field.name not in ("food", "grocery", "customer_pos")}
        for kind in ("food", "grocery"):
            key = f"{kind}_pickup_wait_ticks"
            row[f"{key}_so_far"] = row.pop(key)
        row.update(order.time_metrics())
        row.update({
            "customer_x": order.customer_pos[0], "customer_y": order.customer_pos[1],
            "restaurant_id": order.food.merchant_id,
            "store_id": order.grocery.merchant_id if order.grocery is not None else None,
            "food_item_value": order.food.item_value,
            "grocery_item_value": order.grocery.item_value if order.grocery is not None else 0,
            "movement_count": order.movement_count,
            "travel_distance_km": order.movement_count * KM_PER_STEP,
            "driver_operating_cost": order.movement_count * COST_PER_STEP,
            "emission_units": order.movement_count * EMISSION_PER_STEP,
            "merchant_revenue": sum((c.settled_revenue for c in order.components), ZERO),
            "driver_profit": order.driver_revenue - order.movement_count * COST_PER_STEP,
            "active_at_export": order.order_id in self.active_orders,
        })
        return row

    def _driver_row(self, driver: Driver) -> dict:
        orders = [o for o in self.orders.values() if o.assigned_driver_id == driver.unique_id]
        return {
            "driver_id": driver.unique_id, "x": driver.pos[0], "y": driver.pos[1],
            "state": driver.state, "current_order_id": driver.current_order_id,
            "assigned_order_count": len(orders),
            "delivered_order_count": sum(o.delivered_tick is not None for o in orders),
            "completed_order_count": sum(o.status == OrderStatus.COMPLETED for o in orders),
            "movement_count": driver.movement_count,
            "travel_distance_km": driver.movement_count * KM_PER_STEP,
            "operating_cost": driver.movement_count * COST_PER_STEP,
            "emission_units": driver.movement_count * EMISSION_PER_STEP,
            "busy_ticks": driver.busy_ticks,
            "utilization": driver.busy_ticks / self.tick_counter if self.tick_counter else None,
            "food_pickup_wait_ticks_so_far": sum(o.food_pickup_wait_ticks for o in orders),
            "grocery_pickup_wait_ticks_so_far": sum(o.grocery_pickup_wait_ticks for o in orders),
            "revenue": driver.revenue,
            "profit": driver.revenue - driver.movement_count * COST_PER_STEP,
        }

    def _merchant_row(self, merchant: Merchant) -> dict:
        components = list(merchant.components.values())
        return {
            "merchant_id": merchant.unique_id, "kind": merchant.kind,
            "x": merchant.pos[0], "y": merchant.pos[1], "state": merchant.state,
            "received_components": len(components),
            **{f"{s.value.lower()}_components": sum(c.status == s for c in components) for s in ComponentStatus},
            "completed_components": sum(self.orders[c.order_id].settled for c in components),
            "food_disposal_events": sum(c.cancellation_event == "PREPARED_FOOD_DISPOSAL" for c in components),
            "grocery_return_events": sum(c.cancellation_event == "GROCERY_CANCELLATION_RETURN" for c in components),
            "preparation_stopped_events": sum(c.cancellation_event == "PREPARATION_STOPPED" for c in components),
            **merchant.cancellation_kpis,
            "revenue": merchant.revenue,
        }

    def summary(self) -> dict:
        return {
            **self._calculate_kpis(self.tick_counter - 1 if self.tick_counter else None, self.tick_counter),
            "finished": self.finished,
            "run_status": "FINISHED" if self.finished else "PARTIAL",
            "horizon_ticks": HORIZON,
            "financial_note": FINANCIAL_NOTE,
        }

    def snapshot(self) -> dict:
        """Detached display data: no mutable agent/order references and no RNG draws."""
        return {
            "width": GRID_WIDTH, "height": GRID_HEIGHT,
            "elapsed_ticks": self.tick_counter,
            "last_tick": self.tick_counter - 1 if self.tick_counter else None,
            "horizon_ticks": HORIZON, "finished": self.finished,
            "p_integration": self.config.p_integration, "seed": self.config.seed,
            "drivers": [{**self._driver_row(d), "target_pos": d.target_pos,
                         "route": d.remaining_route()} for d in self.drivers],
            "merchants": [self._merchant_row(m) for m in self.merchants_by_id.values()],
            "customers": [{"customer_id": c.unique_id, "x": c.pos[0], "y": c.pos[1],
                           "state": c.state, "order_id": c.order_id,
                           "order_type": self.orders[c.order_id].order_type,
                           "order_status": self.orders[c.order_id].status} for c in self.customers.values()],
            "kpis": self.summary(),
        }

    def _metadata(self) -> dict:
        packages = {}
        for name in ("mesa", "numpy", "pandas", "matplotlib"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        return {
            "schema_version": 9 if self.experiment_scenario is not None else 8,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(), "libraries": packages,
            "config": {"p_integration": self.config.p_integration, "seed": self.config.seed,
                       "daily_customers": self.config.daily_customers, "num_drivers": self.config.num_drivers,
                       "num_stores": self.config.num_stores,
                       "food_preparation_multiplier": self.config.food_preparation_multiplier,
                       "grocery_preparation_multiplier": self.config.grocery_preparation_multiplier,
                       "store_selection": self.config.store_selection},
            "experiment_scenario": ({"sha256": self.experiment_scenario.sha256,
                                     "seed": self.experiment_scenario.seed,
                                     "demand_sha256": self.experiment_scenario.demand_sha256}
                                    if self.experiment_scenario is not None else None),
            "baseline": {"width": GRID_WIDTH, "height": GRID_HEIGHT, "torus": False,
                         "drivers": self.config.num_drivers, "restaurants": NUM_RESTAURANTS,
                         "stores": self.config.num_stores, "daily_customers": self.config.daily_customers,
                          "horizon_ticks": HORIZON,
                          "assignment_timeout_ticks": ASSIGNMENT_TIMEOUT,
                          "handover_timeout_ticks": HANDOVER_TIMEOUT,
                         "km_per_step": KM_PER_STEP, "cost_per_step": COST_PER_STEP,
                         "emission_units_per_step": EMISSION_PER_STEP},
            "parameter_file": {"path": str(Path(__file__).with_name("simulation_parameters.py")),
                               "sha256": hashlib.sha256(
                                   Path(__file__).with_name("simulation_parameters.py").read_bytes()
                               ).hexdigest()},
            "distribution_parameters": {
                "food_preparation_mean_std": FOOD_PREPARATION,
                "food_preparation_model": "Normal; clamp below zero; ceil to whole-minute ticks",
                "food_preparation_source": ("Meituan positive estimated-ready offset "
                                            "(estimate_meal_prepare_time - order_push_time) / 60; "
                                            "proxy, not observed cooking duration"),
                "food_item_value_mean_std": FOOD_ITEM_VALUE,
                "grocery_preparation_mean_std": GROCERY_PREPARATION,
                "grocery_item_value_mean_std": GROCERY_ITEM_VALUE,
            },
            "financial_parameters": {
                "delivery_minimum_fee": DELIVERY_MINIMUM_FEE,
                "delivery_fee_per_km": DELIVERY_FEE_PER_KM,
                "tax_rate": TAX_RATE,
                "driver_platform_share": DRIVER_PLATFORM_SHARE,
                "driver_delivery_fee_share": DRIVER_DELIVERY_FEE_SHARE,
                "merchant_platform_share": MERCHANT_PLATFORM_SHARE,
                "merchant_share_basis": "item value plus modeled tax",
            },
            "demand_input": {"path": str(self.config.demand_csv), "sha256": self.demand_sha256,
                             "hour_column": "hour", "weight_column": "avg_demand",
                             "aggregation": "mean across daily rows for each hour, then normalize"},
            "specification": {"path": str(SPECIFICATION_PDF), "sha256": self.specification_sha256},
            "policies": {
                "merchant_selection": ("restaurant uniform; store " + self.config.store_selection +
                                       "; nearest uses Manhattan distance and lowest index for ties"),
                "integration": "one Bernoulli draw at order creation; no independent grocery demand",
                "activation": "generate, customers, stores, restaurants, shuffled drivers, resolve, KPI",
                "driver_shuffle": ("shared pre-drawn permutation each tick" if self.experiment_scenario is not None
                                   else "NumPy model RNG permutation each tick"),
                "hour_allocation": "largest remainder; earlier hour wins ties",
                "minute_allocation": "divmod; random distinct minute slots for remainder",
                "ready_tick": "created_tick + preparation_duration (zero allowed)",
                "matching": "Manhattan to first merchant, creation tick, order ID",
                "billable_distance": (f"{KM_PER_STEP} * (Manhattan Restaurant-Customer + "
                                      "Manhattan Store-Restaurant if integrated)"),
                "movement": "Von Neumann four-direction neighbors and current cell; squared Euclidean to target, then (x, y)",
                "cancellation_accounting": "PREPARING only; merchant unit/product-value KPIs; no payment postings",
                "replications": 1,
                "random_inputs": ("pre-drawn scenario shared across paired runs" if self.experiment_scenario is not None
                                  else "single model RNG"),
                "horizon": (f"hard stop after tick {HORIZON - 1}; "
                            "no drain or forced terminal status"),
                "busy_tick": "initial state non-IDLE or successful claim during IDLE activation",
            },
            "metric_definitions": {
                "preparing_cancelled_units": "one unit per food/grocery component cancelled while PREPARING; READY excluded",
                "preparing_cancelled_product_value": "sum of item_value in rupiah for PREPARING cancellations; no tax, fee, commission or revenue posting",
                "assignment_wait_ticks": "assigned_tick - created_tick",
                "delivery_service_ticks": "delivered_tick - created_tick",
                "acknowledgement_delay_ticks": "completed_tick - delivered_tick",
                "completion_total_ticks": "completed_tick - created_tick",
                "cancellation_wait_ticks": "cancelled_tick - created_tick",
                "pickup_wait_ticks": "unsatisfied pickup-state activations; observed mean only after handover",
                "pickup_wait_ticks_so_far": "all unsatisfied pickup activations, including unfinished pickups",
                "driver_utilization": "busy ticks / elapsed ticks (fleet: sum / drivers / elapsed ticks)",
                "completed_service_units": "one food unit per COMPLETED order plus one grocery unit per COMPLETED integrated order; equal unweighted units",
                "service_units_per_driver_hour": "completed_service_units / (drivers * elapsed_ticks / 60)",
                "service_units_per_busy_driver_hour": "completed_service_units / (driver_busy_ticks / 60)",
                "service_units_per_km": "completed_service_units / travel_distance_km",
                "emission_intensity_per_service_unit": "emission_units / completed_service_units; model emission units, not calibrated CO2e",
                "food_service_protection": "all generated orders contain food; completion, cancellation, unfinished, and mean completion, pickup-wait, and post-pickup delivery metrics cover all food demand",
                "integrated_customer_experience": "completion, cancellation, unfinished, and mean completion and component pickup waits restricted to integrated orders; null when no integrated orders exist",
                "time_aggregation": "arithmetic mean over observed order-level values; unobserved values omitted",
                "delivered_orders": "current DELIVERED status, excludes COMPLETED",
                "delivered_cumulative": "all delivered events, includes COMPLETED",
                "missing_values": "CSV blank / JSON null; no zero substitution for unobserved metrics",
                "money_encoding": "exact Decimal values; CSV decimal text / JSON strings, unrounded",
                "customer_payment": "quoted payment; settled_customer_payment includes only COMPLETED orders",
                "emission_units": "model units per PDF; no physical mass unit inferred",
            },
            "hourly_weights": self.hourly_weights.tolist(),
            "hourly_demand": list(self.hourly_demand),
            "minute_demand_schedule": list(self.demand_schedule),
            "finished": self.finished, "elapsed_ticks": self.tick_counter,
            "financial_note": FINANCIAL_NOTE,
        }

    def export_results(self, output_dir: str | Path | None = None) -> Path:
        """Export the present state, including partial runs, without stepping or overwriting."""
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            output_dir = DEFAULT_OUTPUT_ROOT / f"run_{stamp}_{uuid4().hex[:8]}"
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"Output directory must be new or empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        order_rows = [self._order_row(o) for o in self.orders.values()]
        component_rows = []
        for order in self.orders.values():
            for component in order.components:
                component_rows.append({
                    **{f.name: getattr(component, f.name) for f in fields(MerchantOrderComponent)},
                    "planned_ready_tick": component.planned_ready_tick,
                    "order_status": order.status,
                })
        # Stable headers remain available even when a partial run contains no orders.
        order_columns = [f.name for f in fields(Order) if f.name not in ("food", "grocery", "customer_pos")]
        for kind in ("food", "grocery"):
            order_columns.remove(f"{kind}_pickup_wait_ticks")
            order_columns.append(f"{kind}_pickup_wait_ticks_so_far")
        order_columns += ["assignment_wait_ticks", "delivery_service_ticks", "acknowledgement_delay_ticks",
                          "completion_total_ticks", "cancellation_wait_ticks", "food_pickup_wait_ticks",
                          "grocery_pickup_wait_ticks", "customer_x", "customer_y", "restaurant_id", "store_id",
                          "food_item_value", "grocery_item_value", "movement_count", "travel_distance_km",
                          "driver_operating_cost", "emission_units", "merchant_revenue", "driver_profit",
                          "active_at_export"]
        component_columns = [f.name for f in fields(MerchantOrderComponent)] + [
            "planned_ready_tick", "order_status"]
        tables = {
            "kpi_ticks.csv": (self.kpi_records, list(self._calculate_kpis(None, 0))),
            "orders.csv": (order_rows, order_columns),
            "components.csv": (component_rows, component_columns),
            "drivers.csv": ([self._driver_row(d) for d in self.drivers], list(self._driver_row(self.drivers[0]))),
            "merchants.csv": ([self._merchant_row(m) for m in self.merchants_by_id.values()],
                              list(self._merchant_row(self.restaurants[0]))),
        }
        for name, (rows, columns) in tables.items():
            with (destination / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
        for name, data in (("summary.json", self.summary()), ("metadata.json", self._metadata())):
            with (destination / name).open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
                handle.write("\n")
        return destination


def _json_default(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def add_simulation_arguments(parser: argparse.ArgumentParser):
    """Shared CLI contract for the engine and its visualizer."""
    parser.add_argument("--p-integration", type=float, default=P_INTEGRATION,
                        required=P_INTEGRATION is None,
                        help="Integration probability, 0..1; required when P_INTEGRATION is None.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--demand-csv", type=Path, default=DEFAULT_DEMAND_CSV)
    parser.add_argument("--daily-customers", type=int, default=DAILY_CUSTOMERS)
    parser.add_argument("--num-drivers", type=int, default=NUM_DRIVERS)
    parser.add_argument("--num-stores", type=int, default=NUM_STORES)
    parser.add_argument("--food-preparation-multiplier", type=float, default=1.0)
    parser.add_argument("--grocery-preparation-multiplier", type=float, default=1.0)
    parser.add_argument("--store-selection", choices=("uniform", "nearest_restaurant", "nearest_customer"),
                        default="uniform")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New or empty output directory; otherwise a unique run directory is created.")


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(p_integration=args.p_integration, seed=args.seed, demand_csv=args.demand_csv,
                            daily_customers=args.daily_customers, num_drivers=args.num_drivers,
                            num_stores=args.num_stores,
                            food_preparation_multiplier=args.food_preparation_multiplier,
                            grocery_preparation_multiplier=args.grocery_preparation_multiplier,
                            store_selection=args.store_selection)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one day of the PDF food-grocery baseline.")
    add_simulation_arguments(parser)
    args = parser.parse_args(argv)
    try:
        model = IntegratedDeliveryModel(config_from_args(args))
        result = model.run()
        destination = model.export_results(args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Finished {result['elapsed_ticks']} ticks; generated {result['generated_orders']} orders.")
    print(f"Completed: {result['completed_orders']} | Cancelled: {result['cancelled_orders']} | "
          f"Still active: {result['active_orders']}")
    print(f"Travel: {result['travel_distance_km']} km | Emission: {result['emission_units']} model units")
    print(f"Driver revenue: Rp{result['driver_revenue']:,.2f} | Profit: Rp{result['driver_profit']:,.2f}")
    print(FINANCIAL_NOTE)
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()
