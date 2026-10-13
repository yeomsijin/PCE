"""Checks for figure geometry, time indexing and independent matching costs."""

import unittest
import numpy as np
from examples._two_loop_geometry import make_all_clouds, CASE_ORDER
from examples.figure1_two_loops import TUNING
from examples._spring import (
    simulate_spring_from_grid_params,
    paper_style_reconstructions,
)
from examples._baselines import wasserstein_distance
from tpce.entropy import entropy_excess, explanation_discrepancy


class PaperExamples(unittest.TestCase):
    def test_original_figure1_cloud_counts(self):
        clouds = make_all_clouds(TUNING, seed=7)
        self.assertEqual(
            [len(clouds[name]) for name in CASE_ORDER], [239, 758, 695, 605, 84]
        )

    def test_rk4_sampling_and_raw_joint_alignment(self):
        a, b = simulate_spring_from_grid_params(0.0, 0.0)
        t = 0.01 + 0.1 * np.arange(1000)
        np.testing.assert_allclose(a, np.cos(t), atol=1e-8)
        np.testing.assert_allclose(
            b, np.sin(np.sqrt(0.7) * t) / np.sqrt(0.7), atol=1e-8
        )
        clouds = paper_style_reconstructions(a, b, max_points=None)
        self.assertTrue(all(cloud.shape == (997, 4) for cloud in clouds.values()))
        np.testing.assert_array_equal(clouds["AB"][:, 0], a[:997])
        np.testing.assert_array_equal(clouds["AB"][:, 2], b[:997])

    def test_wasserstein_can_match_to_diagonal(self):
        empty = np.empty((0, 2))
        a = np.array([[0.0, 2.0]])
        b = np.array([[100.0, 2.0]])
        self.assertEqual(wasserstein_distance(a, a), 0.0)
        self.assertEqual(wasserstein_distance(a, empty), 1.0)
        self.assertAlmostEqual(wasserstein_distance(a, b), np.sqrt(2.0))
        c = np.array([[0.0, 3.0]])
        self.assertEqual(wasserstein_distance(a, c), 1.0)

    def test_discrepancy_matches_nonempty_energy(self):
        a = np.array([[0.1, 0.3], [0.4, 0.2]])
        b = np.array([[0.2, 0.6]])
        delta = explanation_discrepancy(a, b, [0.4, 0.15])
        p = a[:, 1] / a[:, 1].sum()
        self.assertAlmostEqual(
            float(p @ delta**2 / 8), entropy_excess(a, b, [0.4, 0.15], 2.0)
        )


if __name__ == "__main__":
    unittest.main()
