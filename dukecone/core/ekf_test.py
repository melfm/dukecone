#!/usr/bin/env python

import unittest
import numpy as np

from ekf_base import EKF
import pdb


class EKF_test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x0 = [0, 0, 0]
        mu = [0, 0, 0]
        S = 0.1 * np.identity(3)
        cls.ekf_loc = EKF(x0, mu, S)

    def test_basic_setup(self):
        state_vec = self.ekf_loc.bot.state
        self.assertEquals(len(state_vec),
                          3,
                          'state vector mismatch')

    def test_measurement_calc(self):
        mf = [3, 1]
        state = [3, 1, 1]

        self.ekf_loc.update_measurement(mf, state)
        bearing_m = self.ekf_loc.y[0]
        range_m = self.ekf_loc.y[1]
        self.assertEqual(bearing_m,
                         0.0010,
                         'Wrong bearing measure')
        self.assertEqual(range_m,
                         -1.0,
                         'Wrong range measure')

        mf = [5, 5]
        self.ekf_loc.update_measurement(mf, state)
        bearing_m = self.ekf_loc.y[0]
        range_m = self.ekf_loc.y[1]

        self.assertAlmostEqual(bearing_m,
                               4.472135955,
                               msg='Wrong bearing measure')
        self.assertAlmostEqual(range_m,
                               0.1071487,
                               msg='Wrong range measure')

    def test_get_bearing(self):

        state = [3, 1, 1]
        y = [3.0648, 0.3059]
        bearing_x = self.ekf_loc.get_bearing_x(state, y)
        self.assertAlmostEqual(bearing_x,
                               0.80239287,
                               msg='Wrong bearing x-component')
        bearing_y = self.ekf_loc.get_bearing_y(state, y)
        self.assertAlmostEqual(bearing_y,
                               2.957898699,
                               msg='Wrong bearing y-component')

    def test_state_update(self):
        # self.ekf_loc.run_simulation()
        pass

    def test_closest_feat(self):
        feature_map = np.matrix('5 5; 3 1 ;-4 5; -2 3; 0 4')
        state = [1, 1, 0]

        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [3, 1])

        state = [4, 5, 0]
        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [5, 5])

        state = [10, 8, 0]
        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [5, 5])

        state = [-3,  3, 0]
        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [-2, 3])

if __name__ == '__main__':
    unittest.main()
