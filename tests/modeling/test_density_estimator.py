import unittest
import numpy as np
from modeling.density_estimator import DensityEstimator


class TestDensityEstimator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.normal(loc=0, scale=1, size=(500, 2))
        self.estimator = DensityEstimator(self.data, bandwidth=0.5)

    def test_initialization(self):
        self.assertEqual(self.estimator.data.shape, (500, 2))
        self.assertTrue(hasattr(self.estimator, 'kde_model'))
        self.assertTrue(hasattr(self.estimator, 'x_min'))
        self.assertTrue(hasattr(self.estimator, 'x_max'))
        self.assertTrue(hasattr(self.estimator, 'y_min'))
        self.assertTrue(hasattr(self.estimator, 'y_max'))

    def test_evaluate_grid(self):
        xx, yy, density = self.estimator.evaluate_grid(grid_size=100)
        self.assertEqual(xx.shape, yy.shape)
        self.assertEqual(density.shape, xx.shape)
        self.assertTrue((density >= -1).all() and (density <= 1).all())

    def test_histogram2d(self):
        H, xedges, yedges = self.estimator.histogram2d(bins=50, density=False)
        self.assertEqual(H.shape, (50, 50))
        self.assertEqual(len(xedges), 51)
        self.assertEqual(len(yedges), 51)
        self.assertTrue((H >= 0).all())


if __name__ == '__main__':
    unittest.main()
