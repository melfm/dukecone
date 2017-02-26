#!/usr/bin/env python

# Extended Kalman Filter implementation

import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import sys

from scipy import linalg as la

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
        self.add_noise()

    def add_noise(self):
        # Select a motion disturbance
        e = np.random.multivariate_normal([0, 0, 0], self.R, 1)
        self.state[0] += e[0][0]
        self.state[1] += e[0][1]
        self.state[2] += e[0][2]


class EKF():

    def __init__(self, x0, mu, S, dt):

        self.bot = TurtleBot()
        # Simulation parameters
        self.x0 = x0                    # initial state
        self.n = len(self.x0)           # number of states
        self.dt = dt

        self.bot.state = self.x0

        self.mup = mu
        self.y = [0, 0]

        self.mu = mu                    # mean
        self.S = S                      # Covariance matrix

        # Measurement noise
        q = [0.01, 0.01]
        self.Q = np.diag(q)

        # Define measurement matrix
        self.Ht = None

        # Control inputs
        self.u = [0.1, 0]

        self.meas_updates = None

        self.measure_needs_update = False

        # Simulation initializations
        self.mu_S = []
        self.mup_S = []
        self.mf = [2.5, 0]
        self.Inn = []

        self.bot_states = []

    def update_input(self, new_input):
        # Make sure input makes sense
        assert(len(new_input) == 2)
        self.u = new_input

    def update_estimate(self, u, dt):
        self.mup[0] = self.mu[0] + u[0] * np.cos(self.mu[2]) * dt
        self.mup[1] = self.mu[1] + u[0] * np.sin(self.mu[2]) * dt
        self.mup[2] = self.mu[2] + u[1] * dt

    def set_measurement(self, feat_range, feat_bearing):
        self.y = [feat_range, feat_bearing]
        self.measure_needs_update = True

    def calc_Ht(self, mf, mup):
        # predicted range
        rp = np.sqrt((np.power((mf[0] - mup[0]), 2)) +
                     (np.power((mf[1] - mup[1]), 2)))

        self.Ht = np.matrix([[-(mf[0] - mup[0]) / rp,
                              -(mf[1] - mup[1]) / rp,
                              0],
                             [(mf[1] - mup[1]) / np.power(rp, 2),
                              -(mf[0] - mup[0]) / np.power(rp, 2),
                              -1]])

    def process_measurements(self, mf, mup):
        # Calculate range
        update_0 = np.sqrt(np.power((mf[0] - mup[0]), 2) +
                           np.power((mf[1] - mup[1]), 2))
        update_1 = math.atan2(mf[1] - mup[1],
                              mf[0] - mup[0]) - mup[2]
        self.meas_updates = np.matrix([[update_0, update_1]])

    def update_measurement(self, h, Sp):
        # Measurement update
        K = Sp * np.transpose(self.Ht) * la.inv(
            self.Ht * Sp * np.transpose(self.Ht) + self.Q)
        assert(K.shape == (3,2))

        I = self.y - h
        # wrap angle
        I[1] = np.mod(I[1] + math.pi, 2 * math.pi) - math.pi
        # store for plotting
        self.Inn.append(I)

        current_mu = np.matrix(self.mup).T + (K * np.matrix(I).T)
        assert(current_mu.shape == (3,1))
        self.mu = np.asarray(current_mu).flatten()
        self.S = (np.identity(self.n) - K * self.Ht) * Sp

    def do_estimation(self):

        # Keep storing these for plotting
        current_state = copy.copy(self.bot.state)
        self.bot_states.append(current_state)
        # Update state for EKF comparison
        self.bot.update_state(self.u, self.dt)

        # Extended Kalman Filter Estimation
        # ---------------------------------------------
        # Prediction update (mup)
        self.update_estimate(self.u, self.dt)

        # Linearization of motion model
        Gt = np.matrix(
                [[1, 0, -self.u[0] * np.sin(self.mu[2]) * self.dt],
                 [0, 1, self.u[0] * np.cos(self.mu[2]) * self.dt],
                 [0, 0, 1]])

        # Calculate predicted covariance
        Sp = Gt * self.S * Gt.transpose() + self.bot.R

        # Linearization of measurement model
        self.calc_Ht(self.mf, self.mup)

        # Measurement update
        # ---------------------------------------------
        # update if we have a new measurement
        if (self.measure_needs_update):
            self.process_measurements(self.mf, self.mup)
            h = np.squeeze(np.asarray(self.meas_updates))
            self.update_measurement(h, Sp)
            self.measure_needs_update = False
        else:
            # don't have measurement
            self.mu = self.mup
            self.S = Sp

        # Store results
        current_mup = copy.copy(self.mup)
        self.mup_S.append(current_mup)

        current_bot_mu = copy.copy(self.mu)
        self.mu_S.append(np.asarray(current_bot_mu))

    def plot(self):
        # Plot
        plt.ion()
        plt.figure(1)
        plt.axis('equal')
        plt.axis([-1, 1, -1, 3])
        plt.plot(self.mf[0], self.mf[1], 'bs')
        x_states = [state[0] for state in self.bot_states]
        y_states = [state[1] for state in self.bot_states]
        plt.plot(x_states, y_states, 'g^')
        plt.plot(
            self.bot.state[0],
            self.bot.state[1],
            'r--')
        mu_xs = [mu[0] for mu in self.mu_S]
        mu_ys = [mu[1] for mu in self.mu_S]

        mup_xs = [mup[0] for mup in self.mup_S]
        mup_ys = [mup[1] for mup in self.mup_S]

        plt.plot(mu_xs, mu_ys, 'r.')
        plt.plot(mup_xs, mup_ys, 'b--')
        plt.show()
        plt.pause(0.0000001)
        plt.clf()
