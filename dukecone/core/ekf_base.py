#!/usr/bin/env python

# Extended Kalman Filter implementation

from scipy import linalg as la
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import pdb

class EKF():

    def __init__(self, mu, dt):

        self.mu = mu                    # belief
        self.n = len(self.mu)           # number of states
        self.mup = mu                   # mu bar
        self.dt = dt
        # Covariance matrix
        self.S = 0.1*np.identity(self.n)
        self.y = [0, 0]

        q = [0.25, 0.25]               # Measurement noise
        self.Q = np.diag(q)

        r = [1e-2, 1e-2, 1e-3]          # Motion model noise
        self.R = np.diag(r)

        self.u = [0.0, 0.0]             # Initialize control inputs

        # Initial feature location
        self.mf = [1.2, 0.0]

        # Define measurement matrix
        self.Ht = None
        self.meas_updates = None
        self.measure_needs_update = False

        # For live plotting if neccessary
        self.mu_S = []
        self.mup_S = []
        self.Inn = []
        
        self.do_estimation()

    def update_cmd_input(self, new_input):
        # Make sure input makes sense
        assert(len(new_input) == 2)
        self.u = new_input

    def update_feat_mf(self, new_mf):
        assert(len(new_mf) == 2)
        self.mf = new_mf

    def update_estimate(self):
        self.mup[0] = self.mu[0] + self.u[0] * np.cos(self.mu[2]) * self.dt
        self.mup[1] = self.mu[1] + self.u[0] * np.sin(self.mu[2]) * self.dt
        self.mup[2] = self.mu[2] + self.u[1] * self.dt

    def set_measurement(self, feat_range, feat_bearing):
        self.y = [feat_range, feat_bearing]
        print('actual_measurements:', self.y)
        self.measure_needs_update = True

    def calc_Ht(self):
        # predicted range
        rp = np.sqrt((np.power((self.mf[0] - self.mup[0]), 2)) +
                     (np.power((self.mf[1] - self.mup[1]), 2)))

        self.Ht = np.matrix([[-(self.mf[0] - self.mup[0]) / rp,
                              -(self.mf[1] - self.mup[1]) / rp,
                              0],
                             [(self.mf[1] - self.mup[1]) / np.power(rp, 2),
                              -(self.mf[0] - self.mup[0]) / np.power(rp, 2),
                              -1]])

    def process_measurements(self, mf, mup):
        # Calculate range
        predicted_range = np.sqrt(np.power((mf[0] - mup[0]), 2) +
                                  np.power((mf[1] - mup[1]), 2))
        predicted_bearing = math.atan2(mf[1] - mup[1],
                                       mf[0] - mup[0]) - mup[2]
        predicted_bearing = np.mod(predicted_bearing + math.pi, 2 * math.pi)\
                            - math.pi
        
        print('meas_prediction:', predicted_range, predicted_bearing,)
        
        self.meas_updates = np.matrix([[predicted_range, predicted_bearing]])
        
        #print('Meas Update:', predicted_range, predicted_bearing)

    def update_measurement(self, h, Sp):
        # Measurement update
        K = Sp * np.transpose(self.Ht) * la.inv(
            self.Ht * Sp * np.transpose(self.Ht) + self.Q)
        assert(K.shape == (3, 2))

        I = self.y - h
        # wrap angle
        I[1] = np.mod(I[1] + math.pi, 2 * math.pi) - math.pi
        # store for plotting
        # take this out in the future
        self.Inn.append(I)

        current_mu = np.matrix(self.mup).T + (K * np.matrix(I).T)
        assert(current_mu.shape == (3, 1))
        self.mu = np.asarray(current_mu).flatten()
        self.S = (np.identity(self.n) - K * self.Ht) * Sp

    def do_estimation(self):

        # Extended Kalman Filter Estimation
        # ---------------------------------------------
        # Prediction update (mup)
        self.update_estimate()

        # Linearization of motion model
        Gt = np.matrix(
                [[1, 0, -self.u[0] * np.sin(self.mu[2]) * self.dt],
                 [0, 1, self.u[0] * np.cos(self.mu[2]) * self.dt],
                 [0, 0, 1]])

        # Calculate predicted covariance
        Sp = Gt * self.S * Gt.transpose() + self.R

        # Linearization of measurement model
        self.calc_Ht()

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
        # take this out in the future
        # since plotting happens outside
        # Live plotting is for debugging only
        current_mup = copy.copy(self.mup)
        self.mup_S.append(current_mup)

        current_bot_mu = copy.copy(self.mu)
        self.mu_S.append(np.asarray(current_bot_mu))
        
        #print('mu:', self.mu)
        
    # Live plotting, only use for debugging
    def plot(self):
        # Plot
        plt.axis('equal')
        fig1 = plt.figure(1)
        if(len(self.mf) > 0):
            plt.plot(self.mf[0], self.mf[1], 'bs')
        if((len(self.mu_S) > 0) & (len(self.mup_S) > 0)):
            mu_xs = [mu[0] for mu in self.mu_S]
            mu_ys = [mu[1] for mu in self.mu_S]

            mup_xs = [mup[0] for mup in self.mup_S]
            mup_ys = [mup[1] for mup in self.mup_S]

            plt.figure(1)
            plt.plot(mu_xs, mu_ys, 'b--')
            plt.plot(mup_xs, mup_ys, 'm--')
            fig1.savefig('EKF.png')
        if(len(self.Inn) > 0):
            # plot the innovations
            fig2 = plt.figure(2)
            plt.figure(2)
            innovation_r = [I[0] for I in self.Inn]
            innovation_b = [I[1] for mup in self.Inn]
            plt.plot(innovation_r, 'r')
            plt.plot(innovation_b, 'b')
            fig2.savefig('Innovations.png')
