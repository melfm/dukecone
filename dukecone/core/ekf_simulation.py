#!/usr/bin/env python

import numpy as np
import math
from scipy import linalg as la
import matplotlib.pyplot as plt
import copy
from matplotlib.patches import Ellipse

import numpy.random as rnd
import pdb


class TurtleBot:

    def __init__(self):
        self.state = []

        r = [1e-4, 1e-4, 1e-6]
        self.R = np.diag(r)

    def update(self, u, dt):
        # Select a motion disturbance
        e = np.random.multivariate_normal([0, 0, 0], self.R, 1)
        self.state[0] = self.state[
            0] + u[0] * np.cos(self.state[2]) * dt + e[0][0]
        self.state[1] = self.state[
            1] + u[0] * np.sin(self.state[2]) * dt + e[0][1]
        self.state[2] = self.state[2] + u[1] * dt + e[0][2]


class EKF():

    def __init__(self, x0, mu, S):

        self.bot = TurtleBot()
        # Simulation parameters
        self.x0 = [0, 0, 0]             # initial state
        self.n = len(self.x0)           # number of states
        self.Tf = 20
        self.dt = 0.1
        self.T = np.arange(0, (self.Tf + self.dt), 0.1)

        self.bot.state = self.x0

        self.mup = [0, 0, 0]
        self.y = []

        self.mu = [0, 0, 0]             # mean
        self.S = 0.1 * np.identity(3)   # Covariance matrix

        # Measurement noise
        q = [0.01, 0.01]
        self.Q = np.diag(q)

        # Control inputs
        self.u = [1, 0.3]

        # Feature map
        self.feat_map = np.matrix('5 5; 3 1 ;-4 5; -2 3; 0 4')
        # More convenient for plotting
        self.feat_map_plot = [[5, 3, -4, -2, 0], [5, 1, 5, 3, 4]]

        self.Ht = None
        self.meas_updates = None

        # Simulation initializations
        self.mu_S = []
        self.mup_S = []
        self.mf = []
        self.Inn = []

        self.bot_states = []

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

    def closest_feature(self, feat_map, state):
        ind = 0
        mind = float('inf')
        for i in range(0, feat_map.shape[0]):
            for j in range(feat_map.shape[1]-1):
                dist = np.sqrt(np.power((feat_map.item(i, j) - state[0]), 2) +
                               np.power((feat_map.item(i, j+1) - state[1]), 2))
                if (dist < mind):
                    mind = dist
                    ind = i
        feat = feat_map[ind:ind+1]
        feat_array = []
        feat_array.append(feat.item(0))
        feat_array.append(feat.item(1))
        return feat_array

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

    def run_simulation(self):

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
            self.plot(t)

    def plot(self, t):
        # Plot
        plt.ion()

        plt.figure(1)
        plt.axis('equal')
        plt.axis([-4, 7, -3, 8])
        plt.plot(self.feat_map_plot[0], self.feat_map_plot[1], 'ro')
        plt.plot(self.mf[0], self.mf[1], 'bs')
        x_states = [state[0] for state in self.bot_states]
        y_states = [state[1] for state in self.bot_states]
        plt.plot(x_states, y_states, 'g^')
        plt.plot(
            [self.bot.state[0],
             self.bot.state[0] + self.get_bearing_x(self.bot.state, self.y)],
            [self.bot.state[1],
             self.bot.state[1] + self.get_bearing_y(self.bot.state, self.y)],
            'r--')
        mup_xs = [mup[0] for mup in self.mu_S]
        mup_ys = [mup[1] for mup in self.mu_S]
        plt.plot(mup_xs, mup_ys, 'r.')
        plt.show()
        print("timestep", t)
        plt.pause(0.0000001)
        plt.clf()

    def plot_ellipse(self):
        fig = plt.figure(2)
        ax = fig.add_subplot(111, aspect='equal')

        eig_val, v = np.linalg.eig(self.S)
        eigen_val = np.sqrt(eig_val)
        mup_xs = [mup[0] for mup in self.mu_S]
        mup_ys = [mup[1] for mup in self.mu_S]

        ell = Ellipse(xy=(self.mu[0], self.mu[1]),
                      width=eigen_val[0],
                      height=eigen_val[1],
                      angle=(self.mu[2]))

        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        plt.show()
