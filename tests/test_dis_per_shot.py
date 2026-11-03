"""INC 按 Shot 与 DIS 按 Shot 必须是同一套面板；DIS 另外保留整片晶圆。"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from matplotlib.figure import Figure

from ovl_analyzer.mesr_parser import parse_mesr_file
from ovl_analyzer.mesr_reader import read_mesr
from ovl_analyzer.ovl_model import fit_10param_model
from ovl_analyzer import plots


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE = ROOT / "sample_data" / "260313_REG09.MESR"


def _quiver_points(ax):
    points = []
    for artist in ax.collections:
        if not hasattr(artist, "U"):
            continue
        xs = np.ravel(np.asarray(artist.X, dtype=float))
        ys = np.ravel(np.asarray(artist.Y, dtype=float))
        us = np.ravel(np.asarray(artist.U, dtype=float))
        vs = np.ravel(np.asarray(artist.V, dtype=float))
        for x, y, u, v in zip(xs, ys, us, vs):
            points.append((round(float(x), 5), round(float(y), 5), round(float(u), 5), round(float(v), 5)))
    return points


def _suptitle(fig):
    title = getattr(fig, "_suptitle", None)
    return "" if title is None else title.get_text()


class ReaderTests(unittest.TestCase):
    def test_sample_matches_existing_parser(self):
        parsed = parse_mesr_file(str(SAMPLE))
        mf = read_mesr(str(SAMPLE))
        self.assertEqual(mf.eval_mode, "BASFIA")
        self.assertEqual(mf.kind, "ovl")
        self.assertTrue(mf.is_vector_type)
        self.assertEqual(mf.n_fields, 5)
        self.assertEqual(mf.data_array().shape, (540, 6))
        self.assertEqual(mf.step_pitch, (26000.0, 33000.0))
        self.assertIn("Baseline", mf.get("COMMENT"))
        self.assertIn("\nMEAS_DATA", mf.pretty_text())

        frame = plots.shot_vectors(mf, mf.data_array())
        left = parsed.abs_coords_df.sort_values(["wafer_id", "shot_id", "mesp_id"]).reset_index(drop=True)
        right = frame.sort_values(["wafer_id", "shot_id", "point_id"]).reset_index(drop=True)
        np.testing.assert_allclose(left["x_abs"].to_numpy(), right["x_abs_mm"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(left["y_abs"].to_numpy(), right["y_abs_mm"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(left["dx"].to_numpy(), right["dx"].to_numpy(), atol=1e-9)
        np.testing.assert_allclose(left["dy"].to_numpy(), right["dy"].to_numpy(), atol=1e-9)

    def test_kind_does_not_treat_overlay_filename_as_dis(self):
        body = (FIXTURES / "mini_inc.MESR").read_text(encoding="utf-8").replace("EVAL_MODE:= 'INC'", "EVAL_MODE:= 'BASFIA'")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lot_DIS.MESR"
            path.write_text(body, encoding="utf-8")
            mf = read_mesr(str(path))
        self.assertEqual(mf.kind, "ovl")

    def test_filename_kind_when_mode_is_generic(self):
        body = (FIXTURES / "mini_inc.MESR").read_text(encoding="utf-8")
        body = body.replace("EVAL_MODE:= 'INC'", "EVAL_MODE:= 'MAP'")
        body = body.replace("synthetic INC per shot", "synthetic map")
        with tempfile.TemporaryDirectory() as folder:
            inc_path = Path(folder) / "lot_INC.MESR"
            dis_path = Path(folder) / "lot_DIS.MESR"
            inc_path.write_text(body, encoding="utf-8")
            dis_path.write_text(body, encoding="utf-8")
            self.assertEqual(read_mesr(str(inc_path)).kind, "inc")
            self.assertEqual(read_mesr(str(dis_path)).kind, "dis")

    def test_dual_vector_columns_keep_dis_last_for_autoscale(self):
        text = """
EVAL_MODE:= 'MAP'
WAFER_SIZE:= 300
STEP_PITCH:= 20000, 20000
SHOT(1):= 1, 0, 0
MESP(1):= 1, -8000, -8000, 'BT', '1'
WAFER(1):= 0
MEAS_DATA(1):= 1, 1, 0, 0.01, 0.02, 0.30, -0.40
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "both.MESR"
            path.write_text(text, encoding="utf-8")
            mf = read_mesr(str(path))
        self.assertEqual(mf.kind, "both")
        self.assertEqual(mf.column_names()[-2:], ["DIS_X", "DIS_Y"])
        frame = plots._valid(plots.shot_vectors(mf, mf.data_array()))
        self.assertEqual(len(frame), 1)
        self.assertAlmostEqual(float(frame["dx"].iloc[0]), 0.30)
        self.assertAlmostEqual(float(frame["dy"].iloc[0]), -0.40)


class PlotTests(unittest.TestCase):
    def test_overlay_stays_one_wafer_map(self):
        mf = read_mesr(str(SAMPLE))
        fig = Figure()
        plots.draw_vector(fig, mf, mf.data_array(), scale=5)
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(fig.axes[0].get_title(), "OVL Vector Map")
        self.assertNotIn("per shot", _suptitle(fig))

    def test_inc_and_dis_panels_share_local_coordinates(self):
        inc = read_mesr(str(FIXTURES / "mini_inc.MESR"))
        dis = read_mesr(str(FIXTURES / "mini_dis.MESR"))
        self.assertEqual(inc.kind, "inc")
        self.assertEqual(dis.kind, "dis")
        summary = plots.summarize(dis, dis.data_array())
        self.assertIn("按 Shot", summary)
        self.assertIn("整片", summary)
        self.assertNotIn("超标", summary)

        inc_fig = Figure()
        dis_fig = Figure()
        plots.draw_vector(inc_fig, inc, inc.data_array(), scale=1)
        plots.draw_vector(dis_fig, dis, dis.data_array(), scale=1)

        self.assertEqual(len(inc_fig.axes), 2)
        self.assertEqual(len(dis_fig.axes), 3)
        self.assertEqual(_suptitle(inc_fig), "INC per shot")
        self.assertEqual(_suptitle(dis_fig), "DIS per shot")
        self.assertEqual(dis_fig.axes[-1].get_title(), "DIS wafer")

        inc_panels = [_quiver_points(ax) for ax in inc_fig.axes]
        dis_panels = [_quiver_points(ax) for ax in dis_fig.axes[:-1]]
        self.assertEqual(inc_panels, dis_panels)
        for points in dis_panels:
            self.assertTrue(points)
            self.assertTrue(all(abs(x) < 10 and abs(y) < 10 for x, y, _, _ in points))
            self.assertNotIn(99.0, [u for _, _, u, _ in points])

        overview = _quiver_points(dis_fig.axes[-1])
        xs = [x for x, _, _, _ in overview]
        self.assertIn(-28.0, xs)
        self.assertIn(12.0, xs)
        self.assertGreater(max(abs(x) for x in xs), 20)
        self.assertNotIn(99.0, [u for _, _, u, _ in overview])

    def test_dis_shot_selection_matches_inc(self):
        inc = read_mesr(str(FIXTURES / "mini_inc.MESR"))
        dis = read_mesr(str(FIXTURES / "mini_dis.MESR"))
        inc_fig = Figure()
        dis_fig = Figure()
        plots.draw_vector(inc_fig, inc, inc.data_array(), scale=1, shot_id=2, show_wafer_overview=False)
        plots.draw_vector(dis_fig, dis, dis.data_array(), scale=1, shot_id=2, show_wafer_overview=False)
        self.assertEqual(len(inc_fig.axes), 1)
        self.assertEqual(len(dis_fig.axes), 1)
        self.assertEqual(_quiver_points(inc_fig.axes[0]), _quiver_points(dis_fig.axes[0]))
        self.assertTrue(all(u < 0 for _, _, u, _ in _quiver_points(dis_fig.axes[0])))
        self.assertIn("S2", dis_fig.axes[0].get_title())

    def test_absolute_dis_is_split_back_into_shots(self):
        mf = read_mesr(str(FIXTURES / "mini_dis_abs.MESR"))
        self.assertEqual(mf.kind, "dis")
        frame = plots._valid(plots.shot_vectors(mf, mf.data_array()))
        shot1 = frame[frame["shot_id"] == 1]
        self.assertEqual(len(shot1), 4)
        self.assertAlmostEqual(float(shot1["x_local_mm"].min()), -8.0)
        self.assertAlmostEqual(float(shot1["x_abs_mm"].min()), -28.0)

        fig = Figure()
        plots.draw_vector(fig, mf, mf.data_array(), scale=1)
        self.assertEqual(fig.axes[-1].get_title(), "DIS wafer")
        panel_x = [x for ax in fig.axes[:-1] for x, _, _, _ in _quiver_points(ax)]
        overview_x = [x for x, _, _, _ in _quiver_points(fig.axes[-1])]
        self.assertTrue(panel_x)
        self.assertLess(max(abs(x) for x in panel_x), 10)
        self.assertGreater(max(abs(x) for x in overview_x), 20)

    def test_residual_figure_still_has_raw_and_residual(self):
        data = parse_mesr_file(str(SAMPLE))
        result = fit_10param_model(data, 1)
        fig = Figure()
        plots.draw_residual_map_fig(fig, data, 1, result, 5)
        self.assertEqual(len(fig.axes), 2)
        self.assertIn("Raw Data", fig.axes[0].get_title())
        self.assertIn("Residual", fig.axes[1].get_title())
        self.assertEqual(len(result.residual_dx), int((data.abs_coords_df["wafer_id"].eq(1) & data.abs_coords_df["valid_flag"].eq(0)).sum()))


if __name__ == "__main__":
    unittest.main()
