import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import compare


class ComparisonMetricTests(unittest.TestCase):
    def test_rre_uses_consecutive_associated_poses_across_gaps(self):
        reference = {"t": np.arange(4.), "q": R.identity(4).as_quat()}
        estimate = {"t": np.array([.01, 1.01, 3.02]),
                    "q": R.from_rotvec(np.deg2rad([[0, 0, 0], [0, 0, 0], [0, 10, 0]])).as_quat()}
        result = compare.compute_rre(reference, estimate)
        np.testing.assert_array_equal(result["reference_ids"], [0, 1, 3])
        np.testing.assert_array_equal(result["timestamps"], [1.01, 3.02])
        np.testing.assert_allclose(result["rre_deg"], [0., 10.], atol=1e-12)
        self.assertAlmostEqual(result["rmse_deg"], np.sqrt(50.))

    def test_association_follows_shorter_input_and_allows_reused_nearest_pose(self):
        short = np.array([0., .01, 1.])
        long = np.array([0., .75, 1.25, 2.])
        for first, second, expected in [(short, long, ([0, 1, 2], [0, 0, 1])),
                                        (long, short, ([0, 0, 1], [0, 1, 2]))]:
            ids = compare.associate_timestamps(first, second, max_time_diff=.25)
            np.testing.assert_array_equal(ids, expected)

    def test_bell_uses_only_shared_adjacent_reference_pairs(self):
        times = np.arange(7.)
        reference = {"t": times, "q": R.identity(7).as_quat()}
        estimates = {}
        for name, ids in {"A": [0, 1, 2, 3, 4, 5], "B": [0, 1, 3, 4, 5, 6],
                          "C": [0, 1, 2, 3, 4, 6]}.items():
            angles = np.array([0., 4., 100., 21., 24., 91., 11.])[ids]
            estimates[name] = {"t": times[ids] + .01, "q": R.from_rotvec(
                np.deg2rad(np.column_stack((angles, angles * 0, angles * 0)))).as_quat()}
        methods = compare.compute_bell_overlap(reference, estimates, .05)
        for method in methods.values():
            np.testing.assert_array_equal(method["timestamps"], [1., 4.])
            np.testing.assert_allclose(method["rre_deg"], [4., 3.])
            self.assertAlmostEqual(method["rmse_deg"], np.sqrt(12.5))

    def test_flight_pairs_exclude_skipped_queries_and_tracking_restarts(self):
        reference = {"t": np.arange(7.), "q": R.identity(7).as_quat()}
        estimate = {"t": reference["t"][[0, 1, 3, 4, 5, 6]],
                    "q": R.identity(6).as_quat(), "segment_ids": np.array([1, 1, 1, 2, 2, 2])}

        result = compare.compute_adjacent_rre(reference, estimate)

        np.testing.assert_array_equal(result["pair_end_ids"], [1, 5, 6])
        np.testing.assert_array_equal(result["reference_ids"], [0, 1, 3, 4, 5, 6])

    def test_same_angle_around_different_axes_has_nonzero_rre(self):
        ref = {"t": np.arange(2.), "q": R.from_rotvec([[0., 0., 0.], [np.pi/2, 0., 0.]]).as_quat()}
        est = {"t": np.arange(2.), "q": R.from_rotvec([[0., 0., 0.], [0., np.pi/2, 0.]]).as_quat()}
        result = compare.compute_rre(ref, est)
        self.assertAlmostEqual(result["rmse_deg"], 120.)

    def test_failure_intervals_cover_only_missing_vpair_samples(self):
        self.assertTrue(hasattr(compare, "failure_intervals"),
                        "VPAIR failure interval detection is not implemented")

        intervals = compare.failure_intervals(
            np.arange(5, dtype=float), np.array([False, False, True, False, False]))

        np.testing.assert_allclose(intervals, [[0., 1.5], [2.5, 4.]])

    def test_vpair_plot_marks_orbslam_failure_intervals_in_red(self):
        self.assertTrue(hasattr(compare, "mark_failure_intervals"),
                        "VPAIR failure highlighting is not implemented")
        figure, axis = plt.subplots()

        compare.mark_failure_intervals(axis, [[0., 1.5]], label="ORB-SLAM3 failed")

        self.assertEqual(len(axis.patches), 1)
        self.assertGreater(axis.patches[0].get_facecolor()[0], .8)
        self.assertEqual(axis.patches[0].get_x(), 0.)
        self.assertAlmostEqual(axis.patches[0].get_width(), 1.5)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
