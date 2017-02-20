#!/usr/bin/env python

import unittest
import numpy as np

from ekf_base import EKF

import pdb

class EKF_test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        x0 = np.array([0, 0, 0])
        mu = np.array([0, 0, 0])
        S = 0.1 * np.identity(3)
        cls.ekf_loc = EKF(x0, mu, S)


    def test_basic_setup(self):
        state_vec = self.ekf_loc.bot.state
        #print('State vec', state_vec)

        #print('Control input', self.ekf_loc.u)



    def test_state_update(self):
        self.ekf_loc.run_simulation()

    def test_closest_feat(self):
        feature_map =  np.matrix('5 5; 3 1 ;-4 5; -2 3; 0 4')
        state = [1, 1, 0]

        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [3 ,1])

        state = [4, 5 , 0]
        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [5 ,5])

        state = [10, 8 , 0]
        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [5 ,5])

        state = [-3,  3 , 0]
        res = self.ekf_loc.closest_feature(feature_map, state)
        np.testing.assert_array_equal(res, [-2 ,3])

if __name__ == '__main__':
    unittest.main()
