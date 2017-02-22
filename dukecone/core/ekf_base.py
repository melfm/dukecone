#!/usr/bin/env python

# Extended Kalman Filter implementation

import numpy as np
import math
from scipy import linalg as la
import copy


class TurtleBot:

    def __init__(self):
        self.state = []

        r = [1e-4, 1e-4, 1e-6]
        self.R = np.diag(r)

    def update_state(self, u, dt):
        self.state[0] = self.state[
            0] + u[0] * np.cos(self.state[2]) * dt
        self.state[1] = self.state[
            1] + u[0] * np.sin(self.state[2]) * dt
        self.state[2] = self.state[2] + u[1] * dt

        # Add Gaussian noise to motion
        add_noise()

    def add_noise(self):
        # Select a motion disturbance
        e = np.random.multivariate_normal([0, 0, 0], self.R, 1)
        self.state[0] += e[0][0]
        self.state[1] += e[0][1]
        self.state[2] += e[0][2]


class EKF():

    def __init__(self, x0, mu, S):

        self.bot = TurtleBot()
        # Simulation parameters
        self.x0 = x0                    # initial state
        self.n = len(self.x0)           # number of states
        self.Tf = 20
        self.dt = 0.1
        self.T = np.arange(0, (self.Tf + self.dt), 0.1)

        self.bot.state = self.x0

        self.mup = [0, 0, 0]
        self.y = []

        self.mu = mu                    # mean
        self.S = 0.1 * np.identity(3)   # Covariance matrix

        # Measurement noise
        q = [0.01, 0.01]
        self.Q = np.diag(q)

        # Control inputs
        self.u = [1, 0.3]

        # Feature location
        self.closest_feat_location = []

        self.Ht = None
        self.meas_updates = None

        # Simulation initializations
        self.mu_S = []
        self.mup_S = []
        self.mf = []
        self.Inn = []

        self.bot_states = []

    def update_input(self, new_input):
        """ Method to receive input from EKF ROS Wrapper"""
        self.u = new_input

    def update_estimate(self, u, dt):
        self.mup[0] = self.mu[0] + u[0] * np.cos(self.mu[2]) * dt
        self.mup[1] = self.mu[1] + u[0] * np.sin(self.mu[2]) * dt
        self.mup[2] = self.mu[2] + u[1] * dt

    def update_measurement(self, mf, state):
        # Determine measurement
        meas_range = np.maximum(0.001,
                                np.sqrt(np.power((mf[0] - state[0]), 2) +
                                        np.power((mf[1] - state[1]), 2)))
        meas_bearing = math.atan2(
                                (mf[1] - state[1]),
                                (mf[0] - state[0]))
        meas_bearing -= state[2]
        self.y = []
        self.y.append(meas_range)
        self.y.append(meas_bearing)

    def add_measurement_noise(self):
        # Select a motion disturbance
        dn = np.random.multivariate_normal([0, 0], self.Q, 1)

        self.y[0] += dn[0][0]
        self.y[1] += dn[0][1]

    def get_bearing_x(self, state, y):
        y_0 = y[0]
        y_1 = y[1]
        x_2 = state[2]

        bearing = y_0 * np.cos(y_1 + x_2)
        return bearing

    def get_bearing_y(self, state, y):
        y_0 = y[0]
        y_1 = y[1]
        x_2 = state[2]

        bearing = y_0 * np.sin(y_1 + x_2)
        return bearing

    def calc_predicted_range(self, mf, mup):
        rp = np.sqrt((np.power((mf[0] - mup[0]), 2)) +
                     (np.power((mf[1] - mup[1]), 2)))

        self.Ht = np.matrix([[-(mf[0] - mup[0]) / rp,
                              -(mf[1] - mup[1]) / rp,
                              0],
                             [(mf[1] - mup[1]) / np.power(rp, 2),
                              -(mf[0] - mup[0]) / np.power(rp, 2),
                              -1]])

    def calc_meas_update(self, mf, mup):
        update_0 = np.sqrt(np.power((mf[0] - mup[0]), 2) +
                           np.power((mf[1] - mup[1]), 2))
        update_1 = math.atan2(mf[1] - mup[1],
                              mf[0] - mup[0]) - mup[2]
        self.meas_updates = np.matrix([[update_0, update_1]])

    def do_estimation(self):

        for t in range(1, len(self.T)):
            # Keep storing these for plotting
            current_state = copy.copy(self.bot.state)
            self.bot_states.append(current_state)
            # Update state
            self.bot.update(self.u, self.dt)
            self.update_estimate(self.u, self.dt)

            current_mup = copy.copy(self.mup)
            self.mup_S.append(current_mup)

            self.mf = self.closest_feature(self.feat_map, self.bot.state)

            self.update_measurement(self.mf, self.bot.state)
            self.add_measurement_noise()

            # Extended Kalman Filter Estimation
            # Prediction update
            Gt = np.matrix(
                [[1, 0, -self.u[0] * np.sin(self.mu[2]) * self.dt],
                 [0, 1, self.u[0] * np.cos(self.mu[2]) * self.dt],
                 [0, 0, 1]])

            Sp = Gt * self.S * Gt.transpose() + self.bot.R

            # Linearization
            # Predicted range
            self.calc_predicted_range(self.mf, self.mup)

            # Measurement update
            K = Sp * np.transpose(self.Ht) * la.inv(
                self.Ht * Sp * np.transpose(self.Ht) + self.Q)

            self.calc_meas_update(self.mf, self.mup)

            meas_upd_arr = np.squeeze(np.asarray(self.meas_updates))
            I = self.y - meas_upd_arr
            I[1] = np.mod(I[1] + math.pi, 2 * math.pi) - math.pi
            self.Inn.append(I)

            K_I = K * np.matrix(I).T
            current_mu = np.matrix(self.mup).T + K_I
            self.mu = np.asarray(current_mu).flatten()

            self.S = (np.identity(self.n) - K * self.Ht) * Sp
            # Store results
            current_bot_mu = copy.copy(self.mu)
            self.mu_S.append(np.asarray(current_bot_mu))
