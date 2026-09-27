"""Matplotlib playback of simulation_engine, without independent model logic.

Run: python src/simulation_visualizer.py --p-integration 0.5
Closing exports the current run, even before its horizon. Reset starts a fresh
run with identical inputs and seed; an unfinished run is exported before reset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from math import cos, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, Slider
import numpy as np

if __package__:
    from .simulation_parameters import DEFAULT_VISUALIZER_INTERVAL_MS
    from .simulation_engine import (
        DriverState, IntegratedDeliveryModel,
        add_simulation_arguments, config_from_args,
    )
else:
    from simulation_parameters import DEFAULT_VISUALIZER_INTERVAL_MS
    from simulation_engine import (
        DriverState, IntegratedDeliveryModel,
        add_simulation_arguments, config_from_args,
    )


DRIVER_STYLES = {
    DriverState.IDLE: ("#777777", "Driver: idle"),
    DriverState.GROCERY_TRAVEL: ("#0072B2", "Driver: grocery travel"),
    DriverState.GROCERY_PICKUP: ("#56B4E9", "Driver: grocery pickup"),
    DriverState.FOOD_TRAVEL: ("#D55E00", "Driver: food travel"),
    DriverState.FOOD_PICKUP: ("#E69F00", "Driver: food pickup"),
    DriverState.DELIVER: ("#CC79A7", "Driver: delivery"),
}


class GridVisualizer:
    """All frame drawing reads detached snapshots; only playback advances time."""

    def __init__(
        self,
        model: IntegratedDeliveryModel,
        interval_ms: int = DEFAULT_VISUALIZER_INTERVAL_MS,
        show_driver_ids: bool = True,
        show_routes: bool = True,
        output_dir: Path | str | None = None,
    ):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be greater than zero.")
        self.model = model
        self.config = model.config
        self.interval_ms = interval_ms
        self.show_driver_ids = show_driver_ids
        self.show_routes = show_routes
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.animation = None
        self.is_paused = True
        self.is_closed = False
        self.run_state = "FINISHED" if model.finished else "READY"
        self.last_export_path: Path | None = None
        self.export_error: str | None = None
        self._exported = False
        self._run_index = 0
        self._driver_labels = {}

        self.figure = plt.figure(figsize=(14, 9), facecolor="#FFFFFF")
        self.figure.suptitle("Food-Grocery Delivery Simulation", x=0.06, y=0.975,
                             ha="left", fontsize=17, fontweight="bold")
        self.figure.text(0.06, 0.935,
                         f"{self.config.daily_customers} customers/day  |  {self.config.num_drivers} drivers  |  "
                         f"pI = {self.config.p_integration:g}  |  seed {self.config.seed}",
                         fontsize=10, color="#555555")
        self.axis = self.figure.add_axes((0.055, 0.19, 0.535, 0.70))
        self.info_axis = self.figure.add_axes((0.635, 0.17, 0.345, 0.74))
        self.info_axis.set_axis_off()
        width, height = model.grid.width, model.grid.height
        self.axis.set(xlim=(-0.5, width - 0.5), ylim=(-0.5, height - 0.5),
                      xlabel="Grid X", ylabel="Grid Y")
        self.axis.set_aspect("equal")
        self.axis.set_xticks(np.arange(0, width, 5))
        self.axis.set_yticks(np.arange(0, height, 5))
        self.axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
        self.axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
        self.axis.grid(which="minor", linewidth=0.35, alpha=0.25)
        self.axis.grid(which="major", linewidth=0)
        self.axis.tick_params(which="minor", bottom=False, left=False)
        self.axis.set_facecolor("#FAFAFA")
        self.status_title = self.axis.set_title("", loc="left", fontsize=10, pad=12)

        self.driver_scatters = {
            state: self.axis.scatter([], [], s=70, marker="o", c=color, edgecolors="white",
                                     linewidths=0.8, zorder=6, label=label)
            for state, (color, label) in DRIVER_STYLES.items()
        }
        self.restaurant_scatter = self.axis.scatter([], [], s=48, marker="s", c="#AA5B21",
                                                    zorder=4, label="Restaurant (permanent)")
        self.store_scatter = self.axis.scatter([], [], s=55, marker="^", c="#009E73",
                                               zorder=4, label="Store (permanent)")
        self.customer_scatter = self.axis.scatter([], [], s=32, marker="D", c="#6A3D9A",
                                                  alpha=0.65, zorder=3, label="Customer (active)")
        self.active_lines = LineCollection([], colors="#0072B2", linewidths=1.3, alpha=0.6, zorder=2)
        self.remaining_lines = LineCollection([], colors="#555555", linewidths=0.9,
                                              linestyles="dashed", alpha=0.3, zorder=1)
        self.axis.add_collection(self.remaining_lines)
        self.axis.add_collection(self.active_lines)
        self.info_axis.legend(handles=[*self.driver_scatters.values(), self.restaurant_scatter,
                                      self.store_scatter, self.customer_scatter],
                              loc="upper left", bbox_to_anchor=(0, 1.025), ncol=2,
                              frameon=False, fontsize=8, columnspacing=0.6,
                              handletextpad=0.35, labelspacing=0.65)
        self.clock_text = self.info_axis.text(0, 0.785, "", va="top", fontsize=15, fontweight="bold")
        self.info_text = self.info_axis.text(0, 0.710, "", va="top", family="monospace",
                                            fontsize=9.1, linespacing=1.3, color="#222222")
        self.figure.text(0.06, 0.130,
                         "Solid: active target    Dashed: remaining route    Merchant outline: active components",
                         fontsize=8.2, color="#555555")
        self.footer = self.figure.text(0.06, 0.025,
                                      "",
                                      fontsize=8.2, color="#555555")
        self._setup_controls()
        self._close_connection = self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._refresh_artists()

    def _setup_controls(self):
        self.pause_button = Button(self.figure.add_axes((0.06, 0.075, 0.095, 0.04)), "Resume")
        self.step_button = Button(self.figure.add_axes((0.165, 0.075, 0.095, 0.04)), "Step")
        self.reset_button = Button(self.figure.add_axes((0.270, 0.075, 0.095, 0.04)), "Reset")
        self.speed_slider = Slider(self.figure.add_axes((0.44, 0.085, 0.18, 0.022)), "Speed",
                                   valmin=0.25, valmax=10, valinit=1, valstep=0.25, valfmt="%0.2gx")
        self.pause_button.on_clicked(self._toggle_pause)
        self.step_button.on_clicked(self._step_once)
        self.reset_button.on_clicked(self._reset)
        self.speed_slider.on_changed(self._change_speed)

    @staticmethod
    def _display_positions(snapshot: dict) -> dict[tuple[str, int], tuple[float, float]]:
        """Deterministic sub-cell offsets are display-only, like the reference UI."""
        cells = defaultdict(list)
        for group, key in (("drivers", "driver_id"), ("merchants", "merchant_id"), ("customers", "customer_id")):
            for row in snapshot[group]:
                cells[(row["x"], row["y"])].append((group, row[key]))
        positions = {}
        for cell, members in cells.items():
            members.sort(key=lambda item: item[1])
            for index, key in enumerate(members):
                if len(members) == 1:
                    positions[key] = cell
                else:
                    angle = 2 * pi * index / len(members) + pi / 4
                    positions[key] = (cell[0] + 0.25 * cos(angle), cell[1] + 0.25 * sin(angle))
        return positions

    @staticmethod
    def _offsets(points) -> np.ndarray:
        return np.asarray(points, dtype=float).reshape((-1, 2))

    @staticmethod
    def _number(value, digits=1) -> str:
        return "\u2014" if value is None else f"{value:,.{digits}f}"

    def _update_dashboard(self, snapshot):
        kpi = snapshot["kpis"]
        tick = snapshot["last_tick"]
        clock = "00:00 | Initial state" if tick is None else f"{tick // 60:02d}:{tick % 60:02d} | Tick {tick}"
        self.clock_text.set_text(clock)
        self.status_title.set_text(f"{self.run_state}  |  {snapshot['elapsed_ticks']:,} / 1,440 ticks")
        n = self._number
        self.info_text.set_text("\n".join([
            f"ORDERS             {kpi['generated_orders']:>5} generated",
            f"Food-only / integ. {kpi['food_only_orders']:>5} / {kpi['integrated_orders']}",
            f"Available / assigned {kpi['available_orders']:>3} / {kpi['assigned_orders']}",
            f"Delivered, unacked   {kpi['delivered_orders']:>3}",
            f"Completed / cancelled {kpi['completed_orders']:>3} / {kpi['cancelled_orders']}",
            f"Service units        {kpi['completed_service_units']:>5}",
            f"Units / driver-hour  {n(kpi['service_units_per_driver_hour']):>8}",
            f"Delivered cumulative {kpi['delivered_cumulative']:>3}",
            "",
            "COMPONENTS          Preparing / Ready",
            f"Food                  {kpi['food_preparing_components']:>3} / {kpi['food_ready_components']}",
            f"Grocery               {kpi['grocery_preparing_components']:>3} / {kpi['grocery_ready_components']}",
            f"Disposal / return     {kpi['food_disposal_events']:>3} / {kpi['grocery_return_events']}",
            "",
            "SERVICE & DRIVERS",
            f"Assignment wait      {n(kpi['mean_assignment_wait_ticks']):>8} min",
            f"Delivery service     {n(kpi['mean_delivery_service_ticks']):>8} min",
            f"Mean completion F / I {n(kpi['mean_food_completion_ticks'])} / {n(kpi['mean_integrated_completion_ticks'])} min",
            f"Ack delay            {n(kpi['mean_acknowledgement_delay_ticks']):>8} min",
            f"Pickup wait F / G    {n(kpi['mean_food_pickup_wait_ticks'])} / {n(kpi['mean_grocery_pickup_wait_ticks'])} min",
            f"Fleet utilization    {n(None if kpi['driver_utilization'] is None else 100 * kpi['driver_utilization']):>8}%",
            f"Travel / emission    {n(kpi['travel_distance_km'])} km / {kpi['emission_units']:,} u",
            f"Emission / svc unit  {n(kpi['emission_intensity_per_service_unit']):>8} u",
            "",
            f"Driver cost       Rp {n(kpi['driver_operating_cost'], 0):>12}",
            f"Driver revenue    Rp {n(kpi['driver_revenue'], 0):>12}",
            f"Driver profit     Rp {n(kpi['driver_profit'], 0):>12}",
            f"Merchant revenue  Rp {n(kpi['merchant_revenue'], 0):>12}",
            f"Platform revenue  Rp {n(kpi['platform_revenue'], 0):>12}",
        ]))

    def _refresh_artists(self):
        snapshot = self.model.snapshot()
        positions = self._display_positions(snapshot)
        for state, scatter in self.driver_scatters.items():
            scatter.set_offsets(self._offsets([positions[("drivers", d["driver_id"])]
                                              for d in snapshot["drivers"] if d["state"] == state]))
        for kind, scatter in (("FOOD", self.restaurant_scatter), ("GROCERY", self.store_scatter)):
            merchants = [m for m in snapshot["merchants"] if m["kind"] == kind]
            scatter.set_offsets(self._offsets([positions[("merchants", m["merchant_id"])] for m in merchants]))
            scatter.set_edgecolors(["#111111" if m["state"] == "ACTIVE" else "white" for m in merchants])
            scatter.set_linewidths([1.2 if m["state"] == "ACTIVE" else 0.5 for m in merchants])
        self.customer_scatter.set_offsets(self._offsets([positions[("customers", c["customer_id"])]
                                                        for c in snapshot["customers"]]))
        active, remaining = [], []
        for driver in snapshot["drivers"]:
            key = driver["driver_id"]
            position = positions[("drivers", key)]
            if self.show_routes and driver["route"]:
                stops = [stop["pos"] for stop in driver["route"]]
                active.append([position, stops[0]])
                remaining.extend([[a, b] for a, b in zip(stops, stops[1:])])
            if self.show_driver_ids:
                if key not in self._driver_labels:
                    self._driver_labels[key] = self.axis.text(0, 0, f"D{key}", fontsize=7,
                                                             zorder=8, color="#111111")
                self._driver_labels[key].set_position((position[0] + 0.35, position[1] + 0.35))
        self.active_lines.set_segments(active)
        self.remaining_lines.set_segments(remaining)
        self._update_dashboard(snapshot)
        return (*self.driver_scatters.values(), self.restaurant_scatter, self.store_scatter,
                self.customer_scatter, self.active_lines, self.remaining_lines,
                *self._driver_labels.values(), self.clock_text, self.info_text, self.status_title)

    def _save_current_run(self) -> bool:
        if self._exported:
            return True
        destination = self.output_dir
        if destination is not None and self._run_index:
            destination = destination.parent / f"{destination.name}_reset_{self._run_index}"
        try:
            self.last_export_path = self.model.export_results(destination)
        except (OSError, ValueError) as exc:
            self.export_error = str(exc)
            self.footer.set_text(f"Export failed: {exc}")
            print(f"Export failed: {exc}")
            return False
        self._exported = True
        self.export_error = None
        print(f"{self.model.summary()['run_status']} results: {self.last_export_path}")
        self.footer.set_text("Results exported.")
        return True

    def _finish_run(self):
        self.run_state = "FINISHED"
        self.is_paused = True
        self.pause_button.label.set_text("Finished")
        if self.animation is not None:
            self.animation.event_source.stop()
        self._save_current_run()

    def _advance_once(self):
        if not self.model.finished:
            self.model.step()
        if self.model.finished:
            self._finish_run()

    def _init_frame(self):
        return self._refresh_artists()

    def _update_frame(self, _frame_index):
        if not self.is_paused and not self.is_closed:
            self.run_state = "RUNNING"
            self._advance_once()
        return self._refresh_artists()

    def _toggle_pause(self, _event=None):
        if self.model.finished or self.is_closed:
            return
        self.is_paused = not self.is_paused
        self.run_state = "PAUSED" if self.is_paused else "RUNNING"
        self.pause_button.label.set_text("Resume" if self.is_paused else "Pause")
        if self.animation is not None:
            source = self.animation.event_source
            source.stop() if self.is_paused else source.start()
        self._refresh_artists()
        self.figure.canvas.draw_idle()

    def _step_once(self, _event=None):
        if self.is_closed:
            return
        self.is_paused = True
        if self.animation is not None:
            self.animation.event_source.stop()
        self.run_state = "PAUSED"
        self.pause_button.label.set_text("Resume")
        self._advance_once()
        self._refresh_artists()
        self.figure.canvas.draw_idle()

    def _reset(self, _event=None):
        if self.is_closed:
            return
        self.is_paused = True
        if self.animation is not None:
            self.animation.event_source.stop()
        # Preserve the previous run and retain it on any export/input error.
        self.run_state = "PAUSED"
        self.pause_button.label.set_text("Resume")
        try:
            reset_model = IntegratedDeliveryModel(self.config, scenario=self.model.experiment_scenario)
            if reset_model.demand_sha256 != self.model.demand_sha256:
                raise ValueError("Demand CSV changed; restore it to reset with identical inputs.")
        except (OSError, ValueError) as exc:
            self.footer.set_text(f"Reset failed: {exc}")
            self.figure.canvas.draw_idle()
            return
        if not self._save_current_run():
            self.figure.canvas.draw_idle()
            return
        self.model = reset_model
        self._run_index += 1
        self._exported = False
        self.last_export_path = None
        self.footer.set_text("Reset with the same inputs and seed.")
        self._refresh_artists()
        self.figure.canvas.draw_idle()

    def _change_speed(self, speed):
        if self.animation is not None:
            delay = max(1, int(self.interval_ms / speed))
            self.animation.event_source.interval = delay
            # TimedAnimation restores its interval after each frame.
            self.animation._interval = delay

    def _on_close(self, _event=None):
        if self.is_closed:
            return
        self.is_closed = True
        self.is_paused = True
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()
        self._save_current_run()

    def run(self):
        self.is_paused = False
        self.run_state = "RUNNING"
        self.pause_button.label.set_text("Pause")
        self.animation = FuncAnimation(self.figure, self._update_frame, init_func=self._init_frame,
                                       frames=None, interval=self.interval_ms, blit=False,
                                       repeat=False, cache_frame_data=False)
        if self.model.finished:
            self._finish_run()
        plt.show()
        # Some noninteractive backends do not emit close_event.
        self._on_close()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Visualize the PDF food-grocery simulation baseline.")
    add_simulation_arguments(parser)
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_VISUALIZER_INTERVAL_MS)
    parser.add_argument("--hide-driver-ids", action="store_true")
    parser.add_argument("--hide-routes", action="store_true")
    args = parser.parse_args(argv)
    try:
        visualizer = GridVisualizer(IntegratedDeliveryModel(config_from_args(args)),
                                    interval_ms=args.interval_ms,
                                    show_driver_ids=not args.hide_driver_ids,
                                    show_routes=not args.hide_routes, output_dir=args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    visualizer.run()
    if visualizer.export_error is not None:
        parser.exit(1, f"Export failed: {visualizer.export_error}\n")


if __name__ == "__main__":
    main()
