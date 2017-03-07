#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3, Pose2D


class Plotter:

    def __init__(self):

        # slightly hacky, replace with time
        # since we are plotting against time
        self.counter = 0
        incoming_measure_top = '/dukecone/estimates/meas'
        bot_mocap_top = '/turtlebot/ground_pose'
        obj_mocap_top = '/obj1/ground_pose'

        self.incoming_measure_sub = rospy.Subscriber(incoming_measure_top,
                                                     Vector3,
                                                     self.measure_callback)

        self.bot_mocap_pose = rospy.Subscriber(bot_mocap_top,
                                               Pose2D,
                                               self.bot_pose_callback)

        self.obj_mocap_pose = rospy.Subscriber(obj_mocap_top,
                                               Pose2D,
                                               self.object_pose_callback)

        self.bot_pose = []
        self.object_pose = []

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
        bot_state = [bot_x, bot_y, bot_theta]
        self.bot_pose = bot_state

        self.plot_mocap_data()

    def object_pose_callback(self, data):
        object_x = data.x
        object_y = data.y
        object_pose = [object_x, object_y]
        self.object_pose = object_pose

    def plot_mocap_data(self):

        if (self.bot_pose is not None and
                self.object_pose is not None):
            # plot both
            plt.ion()
            plt.figure(2)
            print(self.bot_pose)
            plt.plot(self.bot_pose[0],
                     self.bot_pose[1],
                     'g^')
            plt.plot(self.object_pose[0],
                     self.object_pose[1],
                     'ro')
            plt.show()
            plt.pause(0.001)

if __name__ == '__main__':
    plotternode = Plotter()
    rospy.init_node('Plotter', anonymous=True)

    r = rospy.Rate(80)

    while not rospy.is_shutdown():
        r.sleep()
