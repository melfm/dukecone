#!/usr/bin/env python

import rospy
import numpy as np
import os
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3, Pose2D

import pdb


class Plotter:

    def __init__(self):

        # slightly hacky, replace with time
        # since we are plotting against time
        self.counter = 0
        self.obj_counter = 0
        incoming_measure_top = '/dukecone/estimates/meas'
        bot_mocap_top = '/turtlebot/ground_pose'
        obj_mocap_top = '/obj1/ground_pose'

        # EKF topics
        ekf_mu_top = '/dukecone/estimates/mu'

        # self.incoming_measure_sub = rospy.Subscriber(incoming_measure_top,
        #                                             Vector3,
        #                                             self.measure_callback)

        self.bot_mocap_pose = rospy.Subscriber(bot_mocap_top,
                                               Pose2D,
                                               self.bot_pose_callback)

        self.obj_mocap_pose = rospy.Subscriber(obj_mocap_top,
                                               Pose2D,
                                               self.object_pose_callback)

        self.ekf_mu_sub = rospy.Subscriber(ekf_mu_top,
                                           Vector3,
                                           self.ekf_mu_callback)

        self.bot_pose = []
        self.object_pose = []
        self.ekf_mu = []

        # change directory for saving plots
        os.chdir('../')

    def measure_callback(self, data):
        plt.ion()
        plt.figure(1)
        plt.axis([0, 500, 0, 3])
        print(data.x, data.y)
        plt.plot(self.counter, data.x, 'ro')
        plt.plot(data.y, 'b--')
        plt.show()
        plt.pause(0.0000001)
        self.counter += 1

    def bot_pose_callback(self, data):
        bot_x = data.x
        bot_y = data.y
        bot_theta = data.theta

        # apply rotation
        R = np.matrix('0 1 0; -1 0 0; 0 0 1')
        bot_state_enu = [bot_x, bot_y, bot_theta]
        bot_state_enu = (np.asarray(bot_state_enu)).reshape(3, 1)
        bot_state_nwu = R * bot_state_enu
        self.bot_pose = bot_state_nwu

        self.bot_pose = np.asarray(self.bot_pose).flatten()

        self.plot_mocap_data()
        #self.plot_robot_trajectory()

    def object_pose_callback(self, data):
        object_x = data.x
        object_y = data.y
        object_pose = [object_x, object_y]

        R = np.matrix('0 1; -1 0')
        object_pose_enu = (np.asarray(object_pose)).reshape(2, 1)
        object_pose_nwu = R * object_pose_enu

        self.object_pose = np.asarray(object_pose_nwu).flatten()

    def ekf_mu_callback(self, data):
        mu_x = data.x
        mu_y = data.y
        mu_theta = data.z

        self.ekf_mu = [mu_x, mu_y, mu_theta]

        # self.plot_ekf_mu()

    def plot_robot_trajectory(self):

        if (self.bot_pose is not None):
            fig = plt.figure(2)
            plt.plot(self.bot_pose[0],
                     self.bot_pose[1],
                     'g^')
            fig.savefig('test_plots/Turtlebot_Mocap_Pos.png')

    def plot_mocap_data(self):

        if (len(self.bot_pose) > 0 and
                len(self.object_pose) > 0 and
                len(self.ekf_mu) > 0):
            # plot both
            plt.ion()
            plt.figure(3)
            plt.axis([-2, 2, -1, 2])
            plt.plot(self.bot_pose[0],
                     self.bot_pose[1],
                     'g^')
            plt.plot(self.object_pose[0],
                     self.object_pose[1],
                     'ro')
            plt.plot(self.ekf_mu[1],
                     self.ekf_mu[0],
                     'bo')
            plt.show()
            plt.pause(0.00000001)

    def plot_ekf_mu(self):

        if(self.ekf_mu is not None):
            fig = plt.figure(1)
            plt.plot(self.ekf_mu[0],
                     self.ekf_mu[1],
                     'p--')
            fig.savefig('test_plots/EKF_mu.png')

    def plot_objmeas_overtime(self):
        pass

if __name__ == '__main__':
    plotternode = Plotter()
    rospy.init_node('Plotter', anonymous=True)

    r = rospy.Rate(20)

    while not rospy.is_shutdown():
        r.sleep()
