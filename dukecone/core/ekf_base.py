#!/usr/bin/env python

import numpy as np
import math
from scipy import linalg as la
import matplotlib.pyplot as plt

import pdb


class TurtleBot:

    def __init__(self):
        self.states = []
        self.mup = []
        self.y = []

        q = [0.01, 0.01]
        self.Q = np.diag(q)

        r = [1e-4, 1e-4, 1e-6]
        self.R = np.diag(r)

    def update(self, u, dt):
        # Select a motion disturbance
        e = np.random.multivariate_normal([0, 0, 0], self.R, 1)
        self.states[0] = self.states[0] + u[0] * np.cos(self.states[2]) * dt + e
        self.states[1] = self.states[1] + u[0] * np.sin(self.states[2]) * dt + e
        self.states[2] = self.states[2] + u[1] * dt + e

    def update_estimate(self, u, dt):
        self.mup[0] = self.mu[0] + (u[0] * np.cos(self.mu[2]) * dt)
        self.mup[1] = self.mu[1] + (u[0] * np.sin(self.mu[2]) * dt)
        self.mup[2] = self.mu[2] + (u[1] * self.dt)

    def update_measurement(self, mf):
        # Select a motion disturbance
        dn = np.random.multivariate_normal([0, 0], self.Q, 1)
        # Determine measurement
        meas_range = np.sqrt(np.power((self.mf[0] - self.x[0]), 2)) + dn[0]
        meas_bearing = math.atan2(
            (mf[1] - self.states[1]),
            (mf[0] - self.states[0]))
        meas_bearing -= self.states[2]
        meas_bearing += dn[1]

        self.y[0] = meas_range
        self.y[1] = meas_bearing


class EKF():

    def __init__(self, x0, mu, S):

        self.bot = TurtleBot()
        # Simulation parameters
        self.x0 = [0, 0, 0]             # initial state
        self.n = len(self.x0)           # number of states
        self.nS = 1000                  # number of samples
        self.Tf = 20
        self.dt = 0.1
        self.T = np.arange(0, (self.Tf + self.dt), 0.1)

        self.mu = [0, 0, 0]             # mean
        self.S = 0.1 * np.identity(3)   # Covariance matrix

        self.y = []
        self.bot.states[0] = self.x0

        # Control inputs
        self.u = np.ones([2, len(self.T)])
        self.u[1, :] = 0.3 * self.u[1, :]

        # Feature map
        self.feat_map = np.matrix('5 5; 3 1 ;-4 5; -2 3; 0 4')
        # More convenient for plotting
        self.feat_map_plot = [[5, 3, -4, -2, 0], [5, 1, 5, 3, 4]]

        self.Ht = None
        self.meas_updates = None

        # Simulation initializations
        self.mu_S = []
        self.mf = []
        self.mup = []
        self.Inn = []

        self.bot_states = []
        self.bot_mup_S = []

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
        return feat

    def get_bearing(self, t):
        y_0 = self.y[0, t]
        y_1 = self.y[1, t]
        x_2 = self.x[2, t]

        bearing = y_0 * np.cos(y_1 + x_2)
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
        self.meas_updates = np.matrix([[np.sqrt(
                                    np.power(
                                        (mf[0] - mup[0]),
                                        2) + np.power(
                                        (mf[1] - mup[1]),
                                        2))],
                                       [(math.atan2(
                                           self.mf[1] - self.mup[0],
                                           self.mf[0] - self.mup[0]))]])

    def run_simulation(self):

        for t in range(1, len(self.T)):
            self.bot.update(self.u, self.dt)
            self.bot_states.append(self.bot.states)

            self.bot.update_estimate(self.u, self.dt)
            self.bot_mup_S.append(self.bot.mup)

            nearest_feat = self.closest_feature(self.feat_map, self.x[:, t])

            self.mf.append(nearest_feat)

            self.bot.update_measurement(self.mf)

            # Extended Kalman Filter Estimation
            # Prediction update
            Gt = np.matrix(
                [[1, 0, -self.u[0, t] * np.sin(self.mu[2]) * self.dt],
                 [0, 1, self.u[0, t] * np.cos(self.mu[2]) * self.dt],
                 [0, 0, 1]])

            Sp = Gt * self.S * Gt.transpose() + self.R

            # Linearization
            # Predicted range
            self.calc_predicted_range(self.mf, self.mup)

            # Measurement update
            K = Sp * np.transpose(self.Ht) * la.inv(
                self.Ht * Sp * np.transpose(self.Ht) + self.Q)

            self.calc_meas_update(self.mf, self.mup)

            I = self.y[:, t].reshape(2, 1) - self.meas_updates
            I[1] = np.mod(I[1] + math.pi, 2 * math.pi) - math.pi
            self.Inn.append(I)

            self.mu = (self.mup.reshape(3, 1) + K * I).T

            # Store results
            self.mup_S[:, t] = self.mup
            self.mu_S[:, t] = self.mu

            self.plot(t)

    def plot(self, t):
        # Plot
        plt.ion()

        plt.figure(1)
        plt.axis([-4, 6, -1, 7])
        plt.plot(self.feat_map_plot[0], self.feat_map_plot[1], 'ro')
        plt.plot(self.mf[0, t], self.mf[1, t], 'bs')
        plt.plot(self.x[0, 0:t], self.x[1, 0:t], 'g^')
        plt.plot([self.x[0, t], self.x[0, t] + self.get_bearing(t)],
                 [self.x[1, t], self.x[1, t] + self.get_bearing(t)],
                 'r--')
        plt.plot(self.mup_S[0, 0:t], self.mup_S[1, 0:t], 'b--')
        plt.show()
        print("step", t)
        plt.pause(0.1)
        plt.clf()
