"""Observable grid animation for the food-grocery Mesa simulation.

The visualizer keeps the simulation engine separate while making a run easy to
interpret. It renders spatial agents, driver state and active targets, plus a
small live dashboard and playback controls.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from math import cos, pi, sin
from typing import Callable, Dict, Iterable, List, Tuple, Type

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, Slider
import numpy as np

from main_abms import (
    DEFAULT_FOOD_DEMAND,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
    DEFAULT_GROCERY_DEMAND,
    DEFAULT_NUM_DRIVERS,
    DEFAULT_SIMULATION_TICKS,
    RANDOM_SEED,
    BaselineModel,
    Customer,
    Driver,
    Grocery,
    Restaurant,
)


DRIVER_STYLES = {
    "idle": {
        "color": "#7F7F7F",
        "label": "Driver: idle",
    },
    "moving": {
        "color": "#0072B2",
        "label": "Driver: moving",
    },
    "arrived": {
        "color": "#CC79A7",
        "label": "Driver: arrived",
    },
    "waiting_pickup": {
        "color": "#E69F00",
        "label": "Driver: waiting pickup",
    },
}


class GridVisualizer:
    """Render a ``BaselineModel`` as an observable, controllable animation."""

    MAX_VISIBLE_DRIVER_IDS = 50

    def __init__(
        self,
        model: BaselineModel,
        simulation_ticks: int = DEFAULT_SIMULATION_TICKS,
        interval_ms: int = 50,
        model_factory: Callable[[], BaselineModel] | None = None,
        show_driver_ids: bool = True,
        show_routes: bool = True,
    ):
        if simulation_ticks <= 0:
            raise ValueError("simulation_ticks must be greater than 0.")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be greater than 0.")

        self.model = model
        self.model_factory = model_factory
        self.simulation_ticks = simulation_ticks
        self.interval_ms = interval_ms
        self.show_driver_ids = show_driver_ids
        self.show_routes = show_routes

        self.animation = None
        self._is_paused = False
        self._is_finished = False
        self._run_state = "READY"
        self._driver_labels = {}

        self.figure = plt.figure(figsize=(12.5, 8.5))
        self.figure.suptitle(
            "Food–Grocery Delivery Simulation",
            x=0.055,
            y=0.975,
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        self.axis = self.figure.add_axes((0.06, 0.16, 0.64, 0.75))
        self.info_axis = self.figure.add_axes((0.735, 0.15, 0.245, 0.77))

        self._setup_grid()
        self._setup_agent_artists()
        self._setup_info_panel()
        self._setup_controls()

    def _setup_grid(self) -> None:
        """Configure a Cartesian view matching the Mesa ``MultiGrid``."""
        width = self.model.grid.width
        height = self.model.grid.height

        self.axis.set_xlim(-0.5, width - 0.5)
        self.axis.set_ylim(-0.5, height - 0.5)
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.set_xlabel("Grid X")
        self.axis.set_ylabel("Grid Y")

        major_x_step = max(1, width // 10)
        major_y_step = max(1, height // 10)
        self.axis.set_xticks(np.arange(0, width, major_x_step))
        self.axis.set_yticks(np.arange(0, height, major_y_step))

        self.axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
        self.axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
        self.axis.grid(which="minor", linewidth=0.35, alpha=0.28)
        self.axis.grid(which="major", linewidth=0)
        self.axis.tick_params(which="minor", bottom=False, left=False)
        self.axis.set_facecolor("#FAFAFA")

    def _marker_scale(self) -> float:
        """Scale markers conservatively when the configured grid size changes."""
        largest_dimension = max(self.model.grid.width, self.model.grid.height)
        return float(np.clip(50 / largest_dimension, 0.55, 1.35))

    def _setup_agent_artists(self) -> None:
        """Create reusable artists with explicit, stable visual semantics."""
        marker_scale = self._marker_scale()

        self.driver_scatters = {}
        for status, style in DRIVER_STYLES.items():
            self.driver_scatters[status] = self.axis.scatter(
                [],
                [],
                marker="o",
                s=76 * marker_scale**2,
                color=style["color"],
                edgecolor="white",
                linewidth=0.7,
                label=style["label"],
                zorder=6,
            )

        self.restaurant_scatter = self.axis.scatter(
            [],
            [],
            marker="s",
            s=66 * marker_scale**2,
            color="#D55E00",
            edgecolor="white",
            linewidth=0.7,
            label="Food order",
            zorder=5,
        )
        self.grocery_scatter = self.axis.scatter(
            [],
            [],
            marker="^",
            s=72 * marker_scale**2,
            color="#009E73",
            edgecolor="white",
            linewidth=0.7,
            label="Grocery order",
            zorder=5,
        )
        self.customer_scatter = self.axis.scatter(
            [],
            [],
            marker="D",
            s=50 * marker_scale**2,
            color="#6A3D9A",
            edgecolor="white",
            linewidth=0.6,
            label="Customer",
            zorder=4,
            alpha=0.85,
        )

        self.route_lines = LineCollection(
            [],
            colors="#0072B2",
            linewidths=1.2,
            linestyles="dashed",
            alpha=0.35,
            label="Active driver target",
            zorder=2,
        )
        self.axis.add_collection(self.route_lines)
        self.status_title = self.axis.set_title(
            "",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )

    def _setup_info_panel(self) -> None:
        """Reserve a right-hand panel for the legend and live simulation KPIs."""
        self.info_axis.set_axis_off()
        self.info_axis.set_xlim(0, 1)
        self.info_axis.set_ylim(0, 1)

        handles = [
            *self.driver_scatters.values(),
            self.restaurant_scatter,
            self.grocery_scatter,
            self.customer_scatter,
        ]
        if self.show_routes:
            handles.append(self.route_lines)

        self.info_axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0, 1.01),
            borderaxespad=0,
            frameon=False,
            title="LEGEND",
            title_fontsize=10,
            fontsize=8.5,
            labelspacing=0.65,
            handletextpad=0.7,
        )
        self.info_axis.text(
            0,
            0.68,
            "SIMULATION TIME",
            transform=self.info_axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#555555",
        )
        self.clock_text = self.info_axis.text(
            0,
            0.63,
            "",
            transform=self.info_axis.transAxes,
            va="top",
            ha="left",
            fontsize=16,
            fontweight="bold",
            family="monospace",
            color="#111111",
        )
        self.info_text = self.info_axis.text(
            0,
            0.49,
            "",
            transform=self.info_axis.transAxes,
            va="top",
            ha="left",
            fontsize=8.7,
            family="monospace",
            linespacing=1.25,
            color="#222222",
        )

    def _setup_controls(self) -> None:
        """Add pause, single-step, reset and playback-speed controls."""
        pause_axis = self.figure.add_axes((0.06, 0.055, 0.09, 0.045))
        step_axis = self.figure.add_axes((0.16, 0.055, 0.09, 0.045))
        reset_axis = self.figure.add_axes((0.26, 0.055, 0.09, 0.045))
        speed_axis = self.figure.add_axes((0.43, 0.065, 0.25, 0.025))

        self.pause_button = Button(pause_axis, "Pause")
        self.step_button = Button(step_axis, "Step")
        self.reset_button = Button(reset_axis, "Reset")
        self.speed_slider = Slider(
            speed_axis,
            "Speed",
            valmin=0.25,
            valmax=10.0,
            valinit=1.0,
            valstep=0.25,
            valfmt="%0.2g×",
        )

        self.pause_button.on_clicked(self._toggle_pause)
        self.step_button.on_clicked(self._step_once)
        self.reset_button.on_clicked(self._reset)
        self.speed_slider.on_changed(self._change_speed)

        if self.model_factory is None:
            self.reset_button.ax.set_visible(False)

    def _collect_spatial_agents(self) -> Dict[Type, List]:
        """Read every living spatial agent, including all configured Drivers."""
        agents: Dict[Type, List] = {
            Driver: [],
            Restaurant: [],
            Grocery: [],
            Customer: [],
        }

        for agent in self.model.agents:
            if agent.pos is None:
                continue

            for agent_type in agents:
                if isinstance(agent, agent_type):
                    agents[agent_type].append(agent)
                    break

        return agents

    @staticmethod
    def _display_positions(agents: Iterable) -> Dict[int, Tuple[float, float]]:
        """Give co-located agents small deterministic offsets within their cell."""
        by_cell = defaultdict(list)
        for agent in agents:
            by_cell[agent.pos].append(agent)

        display_positions = {}
        for cell, cell_agents in by_cell.items():
            ordered_agents = sorted(cell_agents, key=lambda item: item.unique_id)
            if len(ordered_agents) == 1:
                display_positions[id(ordered_agents[0])] = tuple(map(float, cell))
                continue

            radius = min(0.28, 0.16 + 0.012 * len(ordered_agents))
            for index, agent in enumerate(ordered_agents):
                angle = (2 * pi * index / len(ordered_agents)) + (pi / 4)
                display_positions[id(agent)] = (
                    cell[0] + radius * cos(angle),
                    cell[1] + radius * sin(angle),
                )

        return display_positions

    @staticmethod
    def _as_offsets(points: List[Tuple[float, float]]) -> np.ndarray:
        """Return an N×2 array accepted by ``PathCollection.set_offsets``."""
        if not points:
            return np.empty((0, 2), dtype=float)
        return np.asarray(points, dtype=float)

    @staticmethod
    def _mean_or_zero(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _format_clock(tick: int) -> str:
        """Convert elapsed simulation minutes to a one-day HH:MM clock."""
        hour, minute = divmod(tick, 60)
        return f"{hour:02d}:{minute:02d}"

    def _sync_driver_labels(
        self,
        drivers: List[Driver],
        display_positions: Dict[int, Tuple[float, float]],
    ) -> List:
        """Show compact IDs for small scenarios without flooding full-scale runs."""
        labels_enabled = (
            self.show_driver_ids
            and len(drivers) <= self.MAX_VISIBLE_DRIVER_IDS
        )
        desired_ids = {driver.unique_id for driver in drivers} if labels_enabled else set()

        for driver_id in set(self._driver_labels) - desired_ids:
            self._driver_labels.pop(driver_id).remove()

        for driver in drivers:
            if driver.unique_id not in desired_ids:
                continue

            label = self._driver_labels.get(driver.unique_id)
            if label is None:
                label = self.axis.text(
                    0,
                    0,
                    f"D{driver.unique_id}",
                    fontsize=6.8,
                    color="#202020",
                    zorder=8,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                    },
                )
                self._driver_labels[driver.unique_id] = label

            x, y = display_positions[id(driver)]
            label.set_position((x + 0.11, y + 0.11))

        return list(self._driver_labels.values())

    def _update_dashboard(self, spatial_agents: Dict[Type, List]) -> None:
        drivers = spatial_agents[Driver]
        status_counts = {
            status: sum(driver.status == status for driver in drivers)
            for status in DRIVER_STYLES
        }
        busy_drivers = len(drivers) - status_counts["idle"]
        carrying_orders = sum(len(driver.carrying_orders) for driver in drivers)
        food_completed = sum(
            driver.total_restaurant_completion for driver in drivers
        )
        grocery_completed = sum(
            driver.total_grocery_completion for driver in drivers
        )
        total_emission = sum(driver.total_emission for driver in drivers)
        queued_orders = len(getattr(self.model.dispatcher, "order_queue", []))

        mode = getattr(self.model.dispatcher, "mode", "unknown").upper()
        demand_total = (
            self.model.total_food_demand + self.model.total_grocery_demand
        )
        driver_id_state = (
            "shown"
            if self.show_driver_ids and len(drivers) <= self.MAX_VISIBLE_DRIVER_IDS
            else "hidden"
        )

        self.info_text.set_text(
            "SCENARIO\n"
            f"Mode          {mode}\n"
            f"Grid          {self.model.grid.width} × {self.model.grid.height}\n"
            f"Demand/day    {demand_total:,}\n"
            f"Drivers       {len(drivers):,} (IDs {driver_id_state})\n"
            "Scale         configurable scenario\n"
            "\nLIVE STATUS\n"
            f"Idle / busy   {status_counts['idle']:,} / {busy_drivers:,}\n"
            f"Moving        {status_counts['moving']:,}\n"
            f"Waiting       {status_counts['waiting_pickup']:,}\n"
            f"Active orders {len(self.model.active_orders):,} "
            f"(F {len(spatial_agents[Restaurant]):,}, "
            f"G {len(spatial_agents[Grocery]):,})\n"
            f"Queue/carried {queued_orders:,} / {carrying_orders:,}\n"
            "\nCUMULATIVE KPI\n"
            f"Completed     F {food_completed:,}, G {grocery_completed:,}\n"
            f"Expired       F {self.model.expired_restaurant_orders:,}, "
            f"G {self.model.expired_grocery_orders:,}\n"
            f"Avg wait (m)  F "
            f"{self._mean_or_zero(self.model.food_waiting_times):.1f}, "
            f"G {self._mean_or_zero(self.model.grocery_waiting_times):.1f}\n"
            f"Emission      {total_emission:,}"
        )

        progress = min(self.model.tick_counter / self.simulation_ticks, 1.0)
        self.clock_text.set_text(self._format_clock(self.model.tick_counter))
        self.status_title.set_text(
            f"{mode}  •  Tick {self.model.tick_counter:,}/{self.simulation_ticks:,} "
            f"({progress:.1%})  •  {self._run_state}"
        )

    def _refresh_artists(self):
        """Update spatial artists, labels, routes and the live dashboard."""
        spatial_agents = self._collect_spatial_agents()
        all_spatial_agents = [
            agent
            for agents_of_type in spatial_agents.values()
            for agent in agents_of_type
        ]
        display_positions = self._display_positions(all_spatial_agents)

        driver_points = {status: [] for status in DRIVER_STYLES}
        for driver in spatial_agents[Driver]:
            status = driver.status if driver.status in DRIVER_STYLES else "idle"
            driver_points[status].append(display_positions[id(driver)])

        artists = []
        for status, scatter in self.driver_scatters.items():
            scatter.set_offsets(self._as_offsets(driver_points[status]))
            artists.append(scatter)

        for agent_type, scatter in (
            (Restaurant, self.restaurant_scatter),
            (Grocery, self.grocery_scatter),
            (Customer, self.customer_scatter),
        ):
            scatter.set_offsets(
                self._as_offsets(
                    [display_positions[id(agent)] for agent in spatial_agents[agent_type]]
                )
            )
            artists.append(scatter)

        route_segments = []
        if self.show_routes:
            for driver in spatial_agents[Driver]:
                if driver.status != "idle" and driver.target_pos is not None:
                    route_segments.append(
                        [display_positions[id(driver)], tuple(driver.target_pos)]
                    )
        self.route_lines.set_segments(route_segments)
        artists.append(self.route_lines)

        artists.extend(
            self._sync_driver_labels(
                spatial_agents[Driver],
                display_positions,
            )
        )
        self._update_dashboard(spatial_agents)
        artists.extend((self.status_title, self.clock_text, self.info_text))
        return tuple(artists)

    def _init_frame(self):
        """Render the initial state before tick 1 is executed."""
        return self._refresh_artists()

    def _update_frame(self, _frame_index: int):
        """Advance one tick, render it, and stop cleanly at the configured end."""
        if self.model.tick_counter >= self.simulation_ticks:
            self._finish_run()
            return self._refresh_artists()

        self._run_state = "RUNNING"
        self.model.step()

        if self.model.tick_counter >= self.simulation_ticks:
            self._finish_run()

        return self._refresh_artists()

    def _finish_run(self) -> None:
        self._is_finished = True
        self._is_paused = False
        self._run_state = "FINISHED"
        self.pause_button.label.set_text("Finished")
        if self.animation is not None:
            self.animation.event_source.stop()

    def _toggle_pause(self, _event) -> None:
        if self.animation is None or self._is_finished:
            return

        if self._is_paused:
            self._is_paused = False
            self._run_state = "RUNNING"
            self.pause_button.label.set_text("Pause")
            self.animation.event_source.start()
        else:
            self._is_paused = True
            self._run_state = "PAUSED"
            self.pause_button.label.set_text("Resume")
            self.animation.event_source.stop()

        self._refresh_artists()
        self.figure.canvas.draw_idle()

    def _step_once(self, _event) -> None:
        if self.animation is not None:
            self.animation.event_source.stop()

        self._is_paused = True
        self.pause_button.label.set_text("Resume")

        if self.model.tick_counter < self.simulation_ticks:
            self.model.step()

        if self.model.tick_counter >= self.simulation_ticks:
            self._finish_run()
        else:
            self._run_state = "PAUSED"

        self._refresh_artists()
        self.figure.canvas.draw_idle()

    def _reset(self, _event) -> None:
        if self.model_factory is None:
            return

        if self.animation is not None:
            self.animation.event_source.stop()

        old_grid_size = (self.model.grid.width, self.model.grid.height)
        reset_model = self.model_factory()
        new_grid_size = (reset_model.grid.width, reset_model.grid.height)
        if new_grid_size != old_grid_size:
            raise ValueError("Reset model must use the same grid dimensions.")

        self.model = reset_model
        self._is_finished = False
        self._is_paused = True
        self._run_state = "PAUSED"
        self.pause_button.label.set_text("Resume")
        self._refresh_artists()
        self.figure.canvas.draw_idle()

    def _change_speed(self, speed: float) -> None:
        if self.animation is None:
            return
        self.animation.event_source.interval = max(1, int(self.interval_ms / speed))

    def run(self) -> None:
        """Open the visualization and run until paused, closed, or complete."""
        self._run_state = "RUNNING"
        self.animation = FuncAnimation(
            self.figure,
            self._update_frame,
            init_func=self._init_frame,
            frames=None,
            interval=self.interval_ms,
            repeat=False,
            blit=True,
            cache_frame_data=False,
        )
        plt.show()


def build_model_from_args(args: argparse.Namespace) -> BaselineModel:
    """Create one simulation world from command-line configuration."""
    return BaselineModel(
        width=args.width,
        height=args.height,
        total_food_demand=args.food_demand,
        total_grocery_demand=args.grocery_demand,
        num_driver=args.drivers,
        dispatcher_mode=args.mode,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate the Mesa grid with a legend, live counters, driver state, "
            "routes and playback controls."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("separated", "integrated"),
        default="integrated",
        help="Dispatcher policy to visualize (default: integrated).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_GRID_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_GRID_HEIGHT)
    parser.add_argument("--food-demand", type=int, default=DEFAULT_FOOD_DEMAND)
    parser.add_argument(
        "--grocery-demand",
        type=int,
        default=DEFAULT_GROCERY_DEMAND,
    )
    parser.add_argument(
        "--drivers",
        type=int,
        default=DEFAULT_NUM_DRIVERS,
        help=(
            "Number of Drivers in this configurable scenario "
            f"(default: {DEFAULT_NUM_DRIVERS})."
        ),
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_SIMULATION_TICKS)
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=50,
        help="Delay between frames at 1× speed (default: 50 ms).",
    )
    parser.add_argument(
        "--hide-driver-ids",
        action="store_true",
        help="Hide Driver ID labels; IDs are automatically hidden above 50 Drivers.",
    )
    parser.add_argument(
        "--hide-routes",
        action="store_true",
        help="Hide dashed lines from busy Drivers to their active targets.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_factory = lambda: build_model_from_args(args)
    visualizer = GridVisualizer(
        model=model_factory(),
        simulation_ticks=args.ticks,
        interval_ms=args.interval_ms,
        model_factory=model_factory,
        show_driver_ids=not args.hide_driver_ids,
        show_routes=not args.hide_routes,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
