"""Headless playback checks; no visualizer may introduce simulation decisions."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation_engine import HORIZON, IntegratedDeliveryModel, SimulationConfig
from src.simulation_visualizer import GridVisualizer


class VisualizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.output = Path(self.temporary.name) / "run"
        self.model = IntegratedDeliveryModel(SimulationConfig(0.5))
        self.view = GridVisualizer(self.model, output_dir=self.output)

    def tearDown(self):
        plt.close(self.view.figure)
        self.temporary.cleanup()

    def test_render_snapshot_and_speed_do_not_change_time_or_rng(self):
        before = self.model.snapshot()
        rng_before = self.model.rng.bit_generator.state
        self.view._init_frame()
        self.view._refresh_artists()
        self.view.figure.canvas.draw()
        self.view.animation = Mock()
        self.view._change_speed(5)
        self.assertEqual(self.view.animation.event_source.interval, 10)
        self.assertEqual(self.view.animation._interval, 10)
        self.assertEqual(self.model.snapshot(), before)
        self.assertEqual(self.model.rng.bit_generator.state, rng_before)
        self.assertIn("\u2014", self.view.info_text.get_text())

    def test_pause_step_and_resume_advance_exactly_one_tick(self):
        self.view._update_frame(0)
        self.assertEqual(self.model.tick_counter, 0)
        self.view._step_once()
        self.assertEqual(self.model.tick_counter, 1)
        self.assertTrue(self.view.is_paused)
        self.view._toggle_pause()
        self.view._update_frame(0)
        self.assertEqual(self.model.tick_counter, 2)
        self.view._toggle_pause()
        self.view._update_frame(1)
        self.assertEqual(self.model.tick_counter, 2)

    def test_reset_restores_seed_and_preserves_partial_export(self):
        initial = self.model.snapshot()
        for _ in range(15):
            self.view._advance_once()
        self.view._reset()
        self.assertEqual(self.view.model.snapshot(), initial)
        self.assertTrue(self.view.is_paused)
        exported = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(exported["elapsed_ticks"], 15)
        self.assertFalse(exported["finished"])
        self.view._step_once()
        expected = IntegratedDeliveryModel(SimulationConfig(0.5))
        expected.step()
        self.assertEqual(self.view.model.snapshot(), expected.snapshot())
        self.view._on_close()
        self.assertTrue((self.output.parent / "run_reset_1" / "summary.json").exists())

    def test_close_exports_partial_once_without_advancing(self):
        self.view._step_once()
        before = self.model.snapshot()
        with patch.object(self.model, "export_results", wraps=self.model.export_results) as export:
            self.view._on_close()
            self.view._on_close()
            self.view._update_frame(0)
            self.view._step_once()
            self.assertEqual(export.call_count, 1)
        self.assertEqual(self.model.snapshot(), before)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["run_status"], "PARTIAL")
        self.assertEqual(summary["elapsed_ticks"], 1)

    def test_export_failure_is_visible_and_reset_keeps_model(self):
        self.output.mkdir()
        (self.output / "existing.txt").write_text("keep")
        self.view._step_once()
        current = self.model.snapshot()
        self.view._reset()
        self.assertEqual(self.view.model.snapshot(), current)
        self.assertIn("Export failed", self.view.footer.get_text())
        self.assertIsNotNone(self.view.export_error)
        self.assertEqual((self.output / "existing.txt").read_text(), "keep")

    def test_changed_demand_blocks_reset_without_saving_stale_results(self):
        source = Path(self.temporary.name) / "demand.csv"
        source.write_bytes(self.model.config.demand_csv.read_bytes())
        self.view.config = SimulationConfig(0.5, demand_csv=source)
        source.write_bytes(source.read_bytes() + b"\n")
        self.view._step_once()
        before = self.model.snapshot()
        self.view._reset()
        self.assertEqual(self.view.model.snapshot(), before)
        self.assertIn("Demand CSV changed", self.view.footer.get_text())
        self.assertFalse(self.view._exported)
        self.assertFalse(self.output.exists())
        # Continuing the retained run must still export its eventual state.
        self.view._step_once()
        self.view._on_close()
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["elapsed_ticks"], 2)

    def test_visualized_full_run_matches_engine_and_exports_at_horizon(self):
        expected = IntegratedDeliveryModel(SimulationConfig(0.5))
        expected.run()
        self.view._toggle_pause()
        # Use the same frame callback as interactive playback. Suppress drawing
        # on most frames to keep this check fast; real snapshots still run hourly.
        real_refresh = self.view._refresh_artists
        with patch.object(self.view, "_refresh_artists", side_effect=lambda: real_refresh()
                          if self.model.tick_counter % 60 == 0 else ()):
            for tick in range(HORIZON):
                self.view._update_frame(tick)
        self.assertEqual(self.model.kpi_records, expected.kpi_records)
        self.assertEqual(self.model.snapshot(), expected.snapshot())
        self.assertEqual(self.view.run_state, "FINISHED")
        self.assertTrue(self.view.is_paused)
        self.assertIn("23:59", self.view.clock_text.get_text())
        self.assertTrue(json.loads((self.output / "summary.json").read_text())["finished"])
        before = self.model.snapshot()
        self.view._step_once()
        self.view._update_frame(1440)
        self.assertEqual(self.model.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
