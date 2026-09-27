import mesa
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
DEFAULT_GRID_WIDTH = 50
DEFAULT_GRID_HEIGHT = 50
DEFAULT_FOOD_DEMAND = 300
DEFAULT_GROCERY_DEMAND = 200
DEFAULT_NUM_DRIVERS = 5
DEFAULT_SIMULATION_TICKS = 1440

ROOT = Path(__file__).resolve().parent.parent
FOOD_DEMAND_CSV = ROOT / "dataset" / "food_hourly_demand_profile.csv"
GROCERY_DEMAND_CSV = ROOT / "dataset" / "grocery_hourly_demand_profile.csv"


def _load_hourly_demand_profile(csv_path: Path = FOOD_DEMAND_CSV) -> np.ndarray:
    """Load hourly demand, average across all days, and return 24 percentages."""
    df = pd.read_csv(csv_path)
    hourly_avg = df.groupby("hour")["avg_demand"].mean().sort_index()
    pcts = hourly_avg.values / hourly_avg.values.sum()
    return pcts


# ============================================================
# Agents
# ============================================================

class Driver(mesa.Agent):
    """Agent yang berfungsi mengantarkan pesanan food atau grocery."""

    def __init__(
            self,
            model: mesa.Model,
            spawn_pos: tuple = (0, 0),
    ):
        super().__init__(model)

        self.spawn_pos = spawn_pos

        # Movement state
        self.status = "idle"               # idle, moving, arrived, waiting_pickup
        self.target_pos = None              # target (x, y) coordinate
        self.target_agent_id = None         # unique_id of current route-stop agent

        # Multi-order carrying state
        self.carrying_orders = []           # list of order IDs currently carried

        # Assignment tracking (set by Dispatcher)
        self.assigned_restaurant = None
        self.assigned_restaurant_customer = None
        self.assigned_grocery = None
        self.assigned_grocery_customer = None

        # Route format:
        # [(action, target_agent_id, order_id), ...]
        # action: pickup_food, pickup_grocery, deliver_food, deliver_grocery
        self.route = []
        self.route_index = 0

        # Driver KPI
        self.total_emission = 0             # increases by 60 per grid move
        self.total_restaurant_completion = 0
        self.total_grocery_completion = 0

    def set_target(self, target_pos, agent_id=None):
        """Command driver to move to a target coordinate."""
        self.target_pos = target_pos
        self.target_agent_id = agent_id

        if self.pos == target_pos:
            self.status = "arrived"
        else:
            self.status = "moving"

    def set_route(self, route):
        """Assign a complete ordered route to the driver."""
        self.route = list(route)
        self.route_index = 0

        if not self.route:
            self.clear_assignment()
            return

        self._activate_current_stop()

    def pickup_order(self, order_id):
        """Add one order to the driver's carried orders."""
        if order_id not in self.carrying_orders:
            self.carrying_orders.append(order_id)

    def deliver_order(self, order_id):
        """Remove one delivered order from the driver's carried orders."""
        if order_id in self.carrying_orders:
            self.carrying_orders.remove(order_id)
            return True
        return False

    def _activate_current_stop(self):
        """Activate the target of the current route stop."""
        if self.route_index >= len(self.route):
            self.clear_assignment()
            return

        action, target_agent_id, order_id = self.route[self.route_index]
        target_agent = self._get_agent_by_id(target_agent_id)

        if target_agent is None:
            # Invalid/missing stop should not permanently trap the driver.
            self.advance_route()
            return

        self.set_target(target_agent.pos, agent_id=target_agent.unique_id)

    def advance_route(self):
        """Advance to the next route stop or return the driver to idle."""
        self.route_index += 1

        if self.route_index >= len(self.route):
            self.clear_assignment()
            return

        self._activate_current_stop()

    def clear_assignment(self):
        """Clear all trip state after a route is completed or cancelled."""
        self.status = "idle"
        self.target_pos = None
        self.target_agent_id = None

        self.carrying_orders = []

        self.assigned_restaurant = None
        self.assigned_restaurant_customer = None
        self.assigned_grocery = None
        self.assigned_grocery_customer = None

        self.route = []
        self.route_index = 0

    def _move_toward_target(self):
        """Move one grid step toward target; diagonal movement is allowed."""
        if self.target_pos is None:
            self.status = "idle"
            return

        x, y = self.pos
        tx, ty = self.target_pos

        if (x, y) == (tx, ty):
            self.status = "arrived"
            return

        best_pos = (x, y)
        best_dist = (tx - x) ** 2 + (ty - y) ** 2

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height:
                    dist = (tx - nx) ** 2 + (ty - ny) ** 2

                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (nx, ny)

        if best_pos != (x, y):
            self.model.grid.move_agent(self, best_pos)
            self.total_emission += 60

        if self.pos == self.target_pos:
            self.status = "arrived"

    def step(self):
        # Actions occur at the current tick boundary. If an action creates a
        # new movement target, the Driver may start that movement in the same
        # simulation step. Arrival actions after movement are handled on the
        # next tick, keeping travel-time estimates consistent with grid steps.
        if self.status in ["arrived", "waiting_pickup"]:
            self._handle_arrival()

        if self.status == "moving" and self.target_pos is not None:
            self._move_toward_target()

    def _get_agent_by_id(self, agent_id):
        """Get an agent from the model by unique_id."""
        for agent in self.model.agents:
            if agent.unique_id == agent_id:
                return agent
        return None

    def _remove_active_order_agent(self, agent):
        """Remove a picked-up Restaurant/Grocery from active orders and model."""
        self.model.active_orders = [
            (active_agent, expire_tick)
            for active_agent, expire_tick in self.model.active_orders
            if active_agent.unique_id != agent.unique_id
        ]

        self.model.grid.remove_agent(agent)
        agent.remove()

    def _handle_arrival(self):
        """Execute the action associated with the current route stop."""
        if not self.route or self.route_index >= len(self.route):
            self.clear_assignment()
            return

        current_tick = self.model.tick_counter
        action, target_agent_id, order_id = self.route[self.route_index]
        target_agent = self._get_agent_by_id(target_agent_id)

        if target_agent is None:
            # Skip missing stop to avoid a permanently stuck driver.
            # This is defensive; a correctly assigned active route should not lose its target.
            if action in ["deliver_food", "deliver_grocery"]:
                self.deliver_order(order_id)
            self.advance_route()
            return

        if target_agent.pos != self.pos:
            self.set_target(target_agent.pos, agent_id=target_agent.unique_id)
            return

        if action == "pickup_food":
            self._pickup_food(target_agent, order_id, current_tick)
            return

        if action == "pickup_grocery":
            self._pickup_grocery(target_agent, order_id, current_tick)
            return

        if action == "deliver_food":
            self._deliver_food(target_agent, order_id, current_tick)
            return

        if action == "deliver_grocery":
            self._deliver_grocery(target_agent, order_id, current_tick)
            return

        # Unknown action: skip it rather than trapping the driver.
        self.advance_route()

    def _pickup_food(self, restaurant, order_id, current_tick):
        """Pickup a food order once the Restaurant is ready."""
        if not isinstance(restaurant, Restaurant):
            self.advance_route()
            return

        ready_tick = restaurant.spawn_tick + restaurant.food_prep_duration

        if current_tick < ready_tick:
            self.status = "waiting_pickup"
            return

        if restaurant.is_picked_up:
            self.advance_route()
            return

        self.pickup_order(order_id)
        restaurant.is_picked_up = True
        self.assigned_restaurant = None

        self._remove_active_order_agent(restaurant)
        self.advance_route()

    def _pickup_grocery(self, grocery, order_id, current_tick):
        """Pickup a grocery order once the Grocery is ready."""
        if not isinstance(grocery, Grocery):
            self.advance_route()
            return

        ready_tick = grocery.spawn_tick + grocery.grocery_prep_duration

        if current_tick < ready_tick:
            self.status = "waiting_pickup"
            return

        if grocery.is_picked_up:
            self.advance_route()
            return

        self.pickup_order(order_id)
        grocery.is_picked_up = True
        self.assigned_grocery = None

        self._remove_active_order_agent(grocery)
        self.advance_route()

    def _deliver_food(self, customer, order_id, current_tick):
        """Deliver a food order to its linked Customer."""
        if not isinstance(customer, Customer):
            self.advance_route()
            return

        if customer.order_id != order_id:
            self.advance_route()
            return

        if not self.deliver_order(order_id):
            self.advance_route()
            return

        waiting_time = current_tick - customer.spawn_tick
        self.model.food_waiting_times.append(waiting_time)

        customer.receive_delivery(current_tick)
        self.total_restaurant_completion += 1
        self.assigned_restaurant_customer = None

        self.advance_route()

    def _deliver_grocery(self, customer, order_id, current_tick):
        """Deliver a grocery order to its linked Customer."""
        if not isinstance(customer, Customer):
            self.advance_route()
            return

        if customer.order_id != order_id:
            self.advance_route()
            return

        if not self.deliver_order(order_id):
            self.advance_route()
            return

        waiting_time = current_tick - customer.spawn_tick
        self.model.grocery_waiting_times.append(waiting_time)

        customer.receive_delivery(current_tick)
        self.total_grocery_completion += 1
        self.assigned_grocery_customer = None

        self.advance_route()


class Dispatcher(mesa.Agent):
    """Agent yang mengatur matching order dengan Driver dan membangun route."""

    def __init__(
            self,
            model: mesa.Model,
            mode: str = "separated",
    ):
        super().__init__(model)

        self.mode = mode

        # Arrival-ordered list of (order_agent, expire_tick).
        # Dispatch policy determines which candidate is selected.
        self.order_queue = []
        self._tracked_ids = set()

    def track_new_orders(self):
        """Track active, unexpired orders that have not been seen before."""
        current_tick = self.model.tick_counter

        for agent, expire_tick in self.model.active_orders:
            if current_tick >= expire_tick:
                continue

            if agent.unique_id not in self._tracked_ids:
                self.order_queue.append((agent, expire_tick))
                self._tracked_ids.add(agent.unique_id)

    def _clean_expired(self):
        """Synchronize the queue with active and unexpired model orders."""
        current_tick = self.model.tick_counter
        active_ids = {agent.unique_id for agent, _ in self.model.active_orders}

        cleaned_queue = []
        cleaned_ids = set()

        for agent, expire_tick in self.order_queue:
            if agent.unique_id not in active_ids:
                continue

            if current_tick >= expire_tick:
                continue

            cleaned_queue.append((agent, expire_tick))
            cleaned_ids.add(agent.unique_id)

        self.order_queue = cleaned_queue

        # Keep IDs of assigned active orders tracked as well, so they are not
        # re-added while waiting for pickup.
        assigned_ids = self._get_assigned_order_ids()
        self._tracked_ids = (self._tracked_ids & active_ids) | assigned_ids | cleaned_ids

    def _get_assigned_order_ids(self):
        """Return all order IDs currently assigned to Drivers."""
        assigned_ids = set()

        for agent in self.model.agents:
            if not isinstance(agent, Driver):
                continue

            if agent.assigned_restaurant is not None:
                assigned_ids.add(agent.assigned_restaurant)

            if agent.assigned_grocery is not None:
                assigned_ids.add(agent.assigned_grocery)

        return assigned_ids

    def _find_idle_drivers(self):
        """Return Drivers with no active route and no carried orders."""
        return [
            agent
            for agent in self.model.agents
            if isinstance(agent, Driver)
            and agent.status == "idle"
            and not agent.route
            and not agent.carrying_orders
        ]

    def _find_customer(self, order_id):
        """Find the Customer linked to a Restaurant/Grocery order ID."""
        for agent in self.model.agents:
            if isinstance(agent, Customer) and agent.order_id == order_id:
                return agent
        return None

    def _get_order_by_id(self, order_id):
        """Get a Restaurant/Grocery agent by unique_id."""
        for agent in self.model.agents:
            if agent.unique_id == order_id:
                return agent
        return None

    def _travel_time(self, pos_a, pos_b):
        """Travel ticks matching Driver diagonal movement (Chebyshev distance)."""
        dx = abs(pos_a[0] - pos_b[0])
        dy = abs(pos_a[1] - pos_b[1])
        return max(dx, dy)

    def _is_dispatchable(self, agent, expire_tick):
        """Check whether an order is still valid for a new assignment."""
        if self.model.tick_counter >= expire_tick:
            return False

        active_ids = {
            active_agent.unique_id
            for active_agent, _ in self.model.active_orders
        }

        if agent.unique_id not in active_ids:
            return False

        if self._find_customer(agent.unique_id) is None:
            return False

        return True

    def _remove_order_from_queue(self, order_id):
        """Remove an assigned order from queue without untracking it yet."""
        self.order_queue = [
            (agent, expire_tick)
            for agent, expire_tick in self.order_queue
            if agent.unique_id != order_id
        ]

    def _find_nearest_driver(self, drivers, target_pos):
        """Return the nearest Driver to a pickup position."""
        if not drivers:
            return None

        return min(
            drivers,
            key=lambda driver: (
                self._travel_time(driver.pos, target_pos),
                driver.unique_id,
            )
        )

    def _select_oldest_order(self, order_type=None):
        """Select the oldest valid queue order, optionally filtered by type."""
        candidates = []

        for queue_index, (agent, expire_tick) in enumerate(self.order_queue):
            if order_type is not None and not isinstance(agent, order_type):
                continue

            if not self._is_dispatchable(agent, expire_tick):
                continue

            candidates.append((queue_index, agent, expire_tick))

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: (
                item[1].spawn_tick,
                item[2],
                item[0],
            )
        )

    def step(self):
        self.track_new_orders()
        self._clean_expired()

        if self.mode == "separated":
            self._handle_separated()
        elif self.mode == "integrated":
            self._handle_integrated()
        else:
            raise ValueError(
                f"Unknown dispatcher mode: {self.mode}. "
                "Use 'separated' or 'integrated'."
            )

    def _handle_separated(self):
        """Assign each order independently using oldest-order + nearest-driver matching."""
        while True:
            idle_drivers = self._find_idle_drivers()

            if not idle_drivers:
                break

            selected = self._select_oldest_order()

            if selected is None:
                break

            queue_index, order_agent, expire_tick = selected
            customer = self._find_customer(order_agent.unique_id)

            # Validation first; queue mutation happens only after all data exist.
            if customer is None:
                break

            driver = self._find_nearest_driver(idle_drivers, order_agent.pos)

            if driver is None:
                break

            self._assign_single_order(driver, order_agent, customer)
            self._remove_order_from_queue(order_agent.unique_id)

    def _assign_single_order(self, driver, order_agent, customer):
        """Build and assign a normal two-stop route."""
        if isinstance(order_agent, Restaurant):
            driver.assigned_restaurant = order_agent.unique_id
            driver.assigned_restaurant_customer = customer.unique_id

            route = [
                ("pickup_food", order_agent.unique_id, order_agent.unique_id),
                ("deliver_food", customer.unique_id, order_agent.unique_id),
            ]

            driver.set_route(route)
            return

        if isinstance(order_agent, Grocery):
            driver.assigned_grocery = order_agent.unique_id
            driver.assigned_grocery_customer = customer.unique_id

            route = [
                ("pickup_grocery", order_agent.unique_id, order_agent.unique_id),
                ("deliver_grocery", customer.unique_id, order_agent.unique_id),
            ]

            driver.set_route(route)

    def _handle_integrated(self):
        """Assign Food routes with feasible Grocery piggyback, then standalone Grocery."""
        while True:
            idle_drivers = self._find_idle_drivers()

            if not idle_drivers:
                break

            selected_restaurant = self._select_oldest_order(Restaurant)

            if selected_restaurant is None:
                break

            queue_index, restaurant, rest_expire = selected_restaurant
            restaurant_customer = self._find_customer(restaurant.unique_id)

            if restaurant_customer is None:
                break

            best_candidate = self._find_best_integrated_candidate(
                restaurant,
                idle_drivers,
            )

            if best_candidate is not None:
                driver = best_candidate["driver"]
                grocery = best_candidate["grocery"]
                grocery_customer = best_candidate["grocery_customer"]

                self._assign_integrated_route(
                    driver,
                    restaurant,
                    restaurant_customer,
                    grocery,
                    grocery_customer,
                )

                self._remove_order_from_queue(restaurant.unique_id)
                self._remove_order_from_queue(grocery.unique_id)
                continue

            # No feasible Grocery piggyback: assign Food normally.
            driver = self._find_nearest_driver(idle_drivers, restaurant.pos)

            if driver is None:
                break

            self._assign_single_order(
                driver,
                restaurant,
                restaurant_customer,
            )
            self._remove_order_from_queue(restaurant.unique_id)

        # Use any Drivers still available for standalone Grocery orders.
        self._assign_remaining_grocery()

    def _find_best_integrated_candidate(self, restaurant, idle_drivers):
        """Evaluate all feasible Driver-Grocery combinations for one Restaurant."""
        candidates = []

        for grocery, groc_expire in self._iter_valid_grocery_orders():
            grocery_customer = self._find_customer(grocery.unique_id)

            if grocery_customer is None:
                continue

            for driver in idle_drivers:
                feasibility = self._is_piggyback_feasible(
                    driver,
                    restaurant,
                    grocery,
                )

                if not feasibility["feasible"]:
                    continue

                candidates.append({
                    "driver": driver,
                    "grocery": grocery,
                    "grocery_customer": grocery_customer,
                    "arrival_restaurant": feasibility["arrival_restaurant"],
                    "total_travel": feasibility["total_travel"],
                })

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda candidate: (
                candidate["arrival_restaurant"],
                candidate["total_travel"],
                candidate["grocery"].spawn_tick,
                candidate["driver"].unique_id,
            )
        )

    def _iter_valid_grocery_orders(self):
        """Yield valid Grocery orders currently waiting in queue."""
        for grocery, expire_tick in self.order_queue:
            if not isinstance(grocery, Grocery):
                continue

            if not self._is_dispatchable(grocery, expire_tick):
                continue

            yield grocery, expire_tick

    def _is_piggyback_feasible(self, driver, restaurant, grocery):
        """Check whether Grocery can be collected without delaying Food readiness."""
        current_tick = self.model.tick_counter

        travel_driver_to_grocery = self._travel_time(
            driver.pos,
            grocery.pos,
        )

        arrival_grocery = current_tick + travel_driver_to_grocery
        grocery_ready = grocery.spawn_tick + grocery.grocery_prep_duration
        pickup_grocery = max(arrival_grocery, grocery_ready)

        travel_grocery_to_restaurant = self._travel_time(
            grocery.pos,
            restaurant.pos,
        )

        arrival_restaurant = pickup_grocery + travel_grocery_to_restaurant
        restaurant_ready = restaurant.spawn_tick + restaurant.food_prep_duration

        feasible = arrival_restaurant <= restaurant_ready

        return {
            "feasible": feasible,
            "arrival_grocery": arrival_grocery,
            "grocery_ready": grocery_ready,
            "pickup_grocery": pickup_grocery,
            "arrival_restaurant": arrival_restaurant,
            "restaurant_ready": restaurant_ready,
            "total_travel": travel_driver_to_grocery + travel_grocery_to_restaurant,
        }

    def _assign_integrated_route(
            self,
            driver,
            restaurant,
            restaurant_customer,
            grocery,
            grocery_customer,
    ):
        """Assign Grocery -> Restaurant -> Food Customer -> Grocery Customer."""
        driver.assigned_restaurant = restaurant.unique_id
        driver.assigned_restaurant_customer = restaurant_customer.unique_id
        driver.assigned_grocery = grocery.unique_id
        driver.assigned_grocery_customer = grocery_customer.unique_id

        route = [
            ("pickup_grocery", grocery.unique_id, grocery.unique_id),
            ("pickup_food", restaurant.unique_id, restaurant.unique_id),
            ("deliver_food", restaurant_customer.unique_id, restaurant.unique_id),
            ("deliver_grocery", grocery_customer.unique_id, grocery.unique_id),
        ]

        driver.set_route(route)

    def _assign_remaining_grocery(self):
        """Assign remaining Grocery orders independently to idle Drivers."""
        while True:
            idle_drivers = self._find_idle_drivers()

            if not idle_drivers:
                break

            selected_grocery = self._select_oldest_order(Grocery)

            if selected_grocery is None:
                break

            queue_index, grocery, groc_expire = selected_grocery
            grocery_customer = self._find_customer(grocery.unique_id)

            if grocery_customer is None:
                break

            driver = self._find_nearest_driver(idle_drivers, grocery.pos)

            if driver is None:
                break

            self._assign_single_order(
                driver,
                grocery,
                grocery_customer,
            )
            self._remove_order_from_queue(grocery.unique_id)


class Restaurant(mesa.Agent):
    """Agent yang merepresentasikan food order yang sedang disiapkan."""

    def __init__(
            self,
            model: mesa.Model,
            spawn_tick: int = 0,
    ):
        super().__init__(model)

        self.spawn_tick = spawn_tick
        self.food_prep_duration = max(
            0,
            int(self.model.np_rng.normal(15, 3))
        )
        self.is_picked_up = False


class Grocery(mesa.Agent):
    """Agent yang merepresentasikan grocery order yang sedang disiapkan."""

    def __init__(
            self,
            model: mesa.Model,
            spawn_tick: int = 0,
    ):
        super().__init__(model)

        self.spawn_tick = spawn_tick
        self.grocery_prep_duration = max(
            0,
            int(self.model.np_rng.normal(20, 10))
        )
        self.is_picked_up = False


class Customer(mesa.Agent):
    """Agent yang berperan menerima hasil pengantaran order food atau grocery."""

    def __init__(
            self,
            model: mesa.Model,
            spawn_tick: int = 0,
            order_id: int = None,
    ):
        super().__init__(model)

        self.order_id = order_id
        self.spawn_tick = spawn_tick
        self.spawn_pos = None
        self.waiting_time = 0

    def receive_delivery(self, delivery_tick):
        """Receive an order, record waiting time, then remove Customer."""
        self.waiting_time = delivery_tick - self.spawn_tick
        self.model.grid.remove_agent(self)
        self.remove()


# ============================================================
# Utility / Demand Generators
# ============================================================

class FoodGenerator:
    def __init__(self, hourly_pcts: np.ndarray = None, random_generator=None):
        self.hourly_pcts = (
            hourly_pcts
            if hourly_pcts is not None
            else _load_hourly_demand_profile()
        )
        self.rng = (
            random_generator
            if random_generator is not None
            else np.random.default_rng(RANDOM_SEED)
        )

    def allocate_daily_orders(self, total: int) -> list[int]:
        """Allocate exactly `total` orders across 24 hours using hourly percentages."""
        pcts = np.asarray(self.hourly_pcts, dtype=float)
        pcts = pcts / pcts.sum() # ini untuk double check total sum = 1 karena berpotensi terjadi floating point dari code _load_hourly_demand profile()

        raw_orders = total * pcts
        allocated = np.floor(raw_orders).astype(int)
        remaining = total - allocated.sum()

        remainders = raw_orders - allocated
        priority = np.argsort(remainders)[::-1]
        allocated[priority[:remaining]] += 1

        return allocated.tolist()

    def spread_to_minutes(self, hourly_orders: list[int]) -> list[int]:
        """Spread hourly orders uniformly across 1440 minutes (1 tick = 1 minute)."""
        schedule = np.zeros(1440, dtype=int)

        for hour, count in enumerate(hourly_orders):
            if count > 0:
                start = hour * 60
                base_per_minute, remainder = divmod(count, 60)
                schedule[start:start + 60] = base_per_minute

                if remainder > 0:
                    ticks = self.rng.choice(
                        range(start, start + 60),
                        size=remainder,
                        replace=False,
                    )
                    schedule[ticks] += 1

        return schedule.tolist()


class GroceryGenerator:
    def __init__(self, hourly_pcts: np.ndarray = None, random_generator=None):
        self.hourly_pcts = (
            hourly_pcts
            if hourly_pcts is not None
            else _load_hourly_demand_profile(GROCERY_DEMAND_CSV)
        )
        self.rng = (
            random_generator
            if random_generator is not None
            else np.random.default_rng(RANDOM_SEED)
        )

    def allocate_daily_orders(self, total: int) -> list[int]:
        """Allocate exactly `total` orders across 24 hours using hourly percentages."""
        pcts = np.asarray(self.hourly_pcts, dtype=float)
        pcts = pcts / pcts.sum() # ini untuk double check total sum = 1 karena berpotensi terjadi floating point dari code _load_hourly_demand profile()

        raw_orders = total * pcts
        allocated = np.floor(raw_orders).astype(int)
        remaining = total - allocated.sum()

        remainders = raw_orders - allocated
        priority = np.argsort(remainders)[::-1]
        allocated[priority[:remaining]] += 1

        return allocated.tolist()

    def spread_to_minutes(self, hourly_orders: list[int]) -> list[int]:
        """Spread hourly orders uniformly across 1440 minutes (1 tick = 1 minute)."""
        schedule = np.zeros(1440, dtype=int)

        for hour, count in enumerate(hourly_orders):
            if count > 0:
                start = hour * 60
                base_per_minute, remainder = divmod(count, 60)
                schedule[start:start + 60] = base_per_minute

                if remainder > 0:
                    ticks = self.rng.choice(
                        range(start, start + 60),
                        size=remainder,
                        replace=False,
                    )
                    schedule[ticks] += 1

        return schedule.tolist()


# Tidak perlu Driver generator karena jumlah Driver dibentuk langsung oleh Model.
class DriverGenerator:
    pass


# ============================================================
# Model
# ============================================================

class BaselineModel(mesa.Model):
    """Model untuk membandingkan separated dan integrated dispatch policy."""

    RESTAURANT_LIFETIME = 20  # dispatch waiting allowance after food preparation
    GROCERY_LIFETIME = 30     # dispatch waiting allowance after grocery preparation

    def __init__(
            self,
            width: int = DEFAULT_GRID_WIDTH,
            height: int = DEFAULT_GRID_HEIGHT,
            total_food_demand: int = DEFAULT_FOOD_DEMAND,
            total_grocery_demand: int = DEFAULT_GROCERY_DEMAND,
            num_driver: int = DEFAULT_NUM_DRIVERS,
            dispatcher_mode: str = "separated",
            seed: int = RANDOM_SEED,
    ):
        super().__init__(seed=seed)

        # Model-specific NumPy RNG makes scenarios reproducible and allows
        # separated/integrated runs to use the same exogenous random scenario.
        self.np_rng = np.random.default_rng(seed)

        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.total_food_demand = total_food_demand
        self.total_grocery_demand = total_grocery_demand
        self.current_hour = 0

        self.food_gen = FoodGenerator(random_generator=self.np_rng)
        self.grocery_gen = GroceryGenerator(random_generator=self.np_rng)

        self.hourly_food_orders = self.food_gen.allocate_daily_orders(total_food_demand)
        self.hourly_grocery_orders = self.grocery_gen.allocate_daily_orders(total_grocery_demand)

        self.food_schedule = self.food_gen.spread_to_minutes(self.hourly_food_orders)
        self.grocery_schedule = self.grocery_gen.spread_to_minutes(self.hourly_grocery_orders)

        # Active order format: (Restaurant/Grocery agent, dispatch_expire_tick)
        self.active_orders = []
        self.tick_counter = 0

        # KPI records retained after Customer agents disappear.
        self.food_waiting_times = []
        self.grocery_waiting_times = []
        self.expired_restaurant_orders = 0
        self.expired_grocery_orders = 0

        self.dispatcher = Dispatcher(self, mode=dispatcher_mode)

        self._spawn_drivers(num_driver)

    def _spawn_drivers(self, num_driver):
        """Spawn Drivers at random positions."""
        for _ in range(num_driver):
            x = int(self.np_rng.integers(0, self.grid.width))
            y = int(self.np_rng.integers(0, self.grid.height))

            driver = Driver(self, spawn_pos=(x, y))
            self.grid.place_agent(driver, (x, y))

    def step(self):
        minute_in_day = self.tick_counter % 1440
        current_tick = self.tick_counter

        # 1. Spawn new agents based on minute-of-day demand schedule.
        for _ in range(self.food_schedule[minute_in_day]):
            self._spawn_restaurant(current_tick)

        for _ in range(self.grocery_schedule[minute_in_day]):
            self._spawn_grocery(current_tick)

        # 2. Expire overdue UNASSIGNED orders before Dispatcher can assign them.
        self._expire_orders(current_tick)

        # 3. Dispatcher builds and assigns routes.
        self.dispatcher.step()

        # 4. Drivers execute movement / wait / pickup / delivery actions.
        drivers = [
            agent
            for agent in self.agents
            if isinstance(agent, Driver)
        ]

        for driver in drivers:
            driver.step()

        self.tick_counter += 1

    def _spawn_restaurant(self, tick):
        restaurant = Restaurant(self, spawn_tick=tick)

        x = int(self.np_rng.integers(0, self.grid.width))
        y = int(self.np_rng.integers(0, self.grid.height))
        self.grid.place_agent(restaurant, (x, y))

        expire_tick = (
            tick
            + restaurant.food_prep_duration
            + self.RESTAURANT_LIFETIME
        )
        self.active_orders.append((restaurant, expire_tick))

        customer = Customer(
            self,
            spawn_tick=tick,
            order_id=restaurant.unique_id,
        )

        cx = int(self.np_rng.integers(0, self.grid.width))
        cy = int(self.np_rng.integers(0, self.grid.height))
        self.grid.place_agent(customer, (cx, cy))
        customer.spawn_pos = customer.pos

    def _spawn_grocery(self, tick):
        grocery = Grocery(self, spawn_tick=tick)

        x = int(self.np_rng.integers(0, self.grid.width))
        y = int(self.np_rng.integers(0, self.grid.height))
        self.grid.place_agent(grocery, (x, y))

        expire_tick = (
            tick
            + grocery.grocery_prep_duration
            + self.GROCERY_LIFETIME
        )
        self.active_orders.append((grocery, expire_tick))

        customer = Customer(
            self,
            spawn_tick=tick,
            order_id=grocery.unique_id,
        )

        cx = int(self.np_rng.integers(0, self.grid.width))
        cy = int(self.np_rng.integers(0, self.grid.height))
        self.grid.place_agent(customer, (cx, cy))
        customer.spawn_pos = customer.pos

    def _get_assigned_order_ids(self):
        """Return order IDs protected from expiration because they are assigned."""
        assigned_ids = set()

        for agent in self.agents:
            if not isinstance(agent, Driver):
                continue

            if agent.assigned_restaurant is not None:
                assigned_ids.add(agent.assigned_restaurant)

            if agent.assigned_grocery is not None:
                assigned_ids.add(agent.assigned_grocery)

        return assigned_ids

    def _find_customer(self, order_id):
        """Find a Customer linked to an order ID."""
        for agent in self.agents:
            if isinstance(agent, Customer) and agent.order_id == order_id:
                return agent
        return None

    def _expire_orders(self, tick):
        """Expire overdue orders that have not yet been assigned to a Driver."""
        assigned_ids = self._get_assigned_order_ids()
        still_active = []

        for order_agent, expire_tick in self.active_orders:
            if tick < expire_tick or order_agent.unique_id in assigned_ids:
                still_active.append((order_agent, expire_tick))
                continue

            customer = self._find_customer(order_agent.unique_id)

            if isinstance(order_agent, Restaurant):
                self.expired_restaurant_orders += 1
            elif isinstance(order_agent, Grocery):
                self.expired_grocery_orders += 1

            self.grid.remove_agent(order_agent)
            order_agent.remove()

            if customer is not None:
                self.grid.remove_agent(customer)
                customer.remove()

        self.active_orders = still_active


# ============================================================
# Run comparison
# ============================================================

def _mean_or_zero(values):
    """Return mean value or 0.0 for an empty KPI list."""
    if not values:
        return 0.0
    return float(np.mean(values))


def _collect_driver_metrics(model):
    """Collect aggregate Driver metrics from one model run."""
    drivers = [
        agent
        for agent in model.agents
        if isinstance(agent, Driver)
    ]

    total_emission = sum(driver.total_emission for driver in drivers)
    total_restaurant_completion = sum(
        driver.total_restaurant_completion
        for driver in drivers
    )
    total_grocery_completion = sum(
        driver.total_grocery_completion
        for driver in drivers
    )

    return {
        "drivers": drivers,
        "total_emission": total_emission,
        "restaurant_completion": total_restaurant_completion,
        "grocery_completion": total_grocery_completion,
        "avg_food_waiting": _mean_or_zero(model.food_waiting_times),
        "avg_grocery_waiting": _mean_or_zero(model.grocery_waiting_times),
        "expired_restaurant": model.expired_restaurant_orders,
        "expired_grocery": model.expired_grocery_orders,
    }


if __name__ == "__main__":
    FOOD_DEMAND = DEFAULT_FOOD_DEMAND
    GROCERY_DEMAND = DEFAULT_GROCERY_DEMAND
    NUM_DRIVER = DEFAULT_NUM_DRIVERS
    SIMULATION_TICKS = DEFAULT_SIMULATION_TICKS

    print(f"{'=' * 60}")
    print("SEPARATED MODE")
    print(f"{'=' * 60}")

    model_sep = BaselineModel(
        total_food_demand=FOOD_DEMAND,
        total_grocery_demand=GROCERY_DEMAND,
        num_driver=NUM_DRIVER,
        dispatcher_mode="separated",
        seed=RANDOM_SEED,
    )

    for _ in range(SIMULATION_TICKS):
        model_sep.step()

    sep = _collect_driver_metrics(model_sep)

    for driver in sep["drivers"]:
        print(
            f"Driver {driver.unique_id}: "
            f"emission={driver.total_emission} "
            f"rest_comp={driver.total_restaurant_completion} "
            f"groc_comp={driver.total_grocery_completion}"
        )

    print(f"Total emission: {sep['total_emission']}")
    print(f"Restaurant completions: {sep['restaurant_completion']}")
    print(f"Grocery completions: {sep['grocery_completion']}")
    print(f"Average food waiting time: {sep['avg_food_waiting']:.2f}")
    print(f"Average grocery waiting time: {sep['avg_grocery_waiting']:.2f}")
    print(f"Expired restaurant orders: {sep['expired_restaurant']}")
    print(f"Expired grocery orders: {sep['expired_grocery']}")

    print(f"\n{'=' * 60}")
    print("INTEGRATED MODE")
    print(f"{'=' * 60}")

    model_int = BaselineModel(
        total_food_demand=FOOD_DEMAND,
        total_grocery_demand=GROCERY_DEMAND,
        num_driver=NUM_DRIVER,
        dispatcher_mode="integrated",
        seed=RANDOM_SEED,
    )

    for _ in range(SIMULATION_TICKS):
        model_int.step()

    integrated = _collect_driver_metrics(model_int)

    for driver in integrated["drivers"]:
        print(
            f"Driver {driver.unique_id}: "
            f"emission={driver.total_emission} "
            f"rest_comp={driver.total_restaurant_completion} "
            f"groc_comp={driver.total_grocery_completion}"
        )

    print(f"Total emission: {integrated['total_emission']}")
    print(f"Restaurant completions: {integrated['restaurant_completion']}")
    print(f"Grocery completions: {integrated['grocery_completion']}")
    print(f"Average food waiting time: {integrated['avg_food_waiting']:.2f}")
    print(f"Average grocery waiting time: {integrated['avg_grocery_waiting']:.2f}")
    print(f"Expired restaurant orders: {integrated['expired_restaurant']}")
    print(f"Expired grocery orders: {integrated['expired_grocery']}")

    print(f"\n{'=' * 82}")
    print("COMPARISON")
    print(f"{'=' * 82}")
    print(
        f"{'Metric':<32} "
        f"{'Separated':>20} "
        f"{'Integrated':>20}"
    )
    print(f"{'-' * 74}")
    print(
        f"{'Total emission':<32} "
        f"{sep['total_emission']:>20} "
        f"{integrated['total_emission']:>20}"
    )
    print(
        f"{'Restaurant completions':<32} "
        f"{sep['restaurant_completion']:>20} "
        f"{integrated['restaurant_completion']:>20}"
    )
    print(
        f"{'Grocery completions':<32} "
        f"{sep['grocery_completion']:>20} "
        f"{integrated['grocery_completion']:>20}"
    )
    print(
        f"{'Avg food waiting':<32} "
        f"{sep['avg_food_waiting']:>20.2f} "
        f"{integrated['avg_food_waiting']:>20.2f}"
    )
    print(
        f"{'Avg grocery waiting':<32} "
        f"{sep['avg_grocery_waiting']:>20.2f} "
        f"{integrated['avg_grocery_waiting']:>20.2f}"
    )
    print(
        f"{'Expired restaurant orders':<32} "
        f"{sep['expired_restaurant']:>20} "
        f"{integrated['expired_restaurant']:>20}"
    )
    print(
        f"{'Expired grocery orders':<32} "
        f"{sep['expired_grocery']:>20} "
        f"{integrated['expired_grocery']:>20}"
    )
