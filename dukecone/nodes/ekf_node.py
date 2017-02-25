#!/usr/bin/env python
import sys
sys.path.insert(0, '../core/')
import ekf_base as ekf
import rospy
import numpy as np

from geometry_msgs.msg import Twist
from dukecone.msg import ObjectLocation

class EKFNode():

    def __init__(self):
        rospy.init_node('EKFNode', anonymous=True)

        # subscribe to turtlebot input command
        bot_cmd_topic = '/cmd_vel_mux/input/navi'
        self.ekf_sub_input = rospy.Subscriber(
                                             bot_cmd_topic,
                                             Twist,
                                             self.bot_input_callback)
        # subscribe to object detector
        tf_obj_topic = '/tensorflow/object/location'
        self.ekf_sub_obj = rospy.Subscriber(tf_obj_topic,
                                            ObjectLocation,
                                            self.obj_callback)

        # For now we are using static commands
        self.bot_linear_x = 0.1
        self.feat_x_center = None
        self.feat_y_center = None
        self.feat_range = 0.0
        self.feat_bearing = 0.0

        # Define operating rates for ekf
        # TODO: Update EKF rate
        self.ekf_rate = 15
        dt = 1.0/self.ekf_rate

        # TODO: EKF init function

        # Define initial state and prior
        # Assume we have perfect knowledge of prior
        x0 = [0, 0, 0]
        mu = [0, 0, 0]

        # Define number of states
        n = len(x0)

        # Define initial covariance
        S = 0.1*np.identity(n)

        Tf = 30

        self.ekf = ekf.EKF(x0, mu, S, Tf, dt)

    def bot_input_callback(self, data):
        pass
        # Extract velocity data from Twist message
        # vel = data.linear.x
        # omega = data.angular.z
        # new_input = [vel, omega]
        # Send new input to EKF class
        # self.ekf.update_input(new_input)

    def obj_callback(self, data):
        """ Get position of closest detected object"""
        self.feat_x_center = data.x_center
        self.feat_y_center = data.y_center
        self.feat_range = data.distance/1000.0

    def calculate_bearing(self, x_center, y_center):
        # Calculate robot bearing based on object location

        # Temporarily assume we are facing object head on
        #self.feat_bearing = 0
        pass

    def run_estimator(self):

        # Update input
        self.ekf.update_input([self.bot_linear_x, 0])
        # Update measurement
        self.ekf.update_measurement(self.feat_range,
                                    self.feat_bearing)

        self.ekf.do_estimation()


if __name__ == '__main__':

    ekf_node = EKFNode()
    rate = rospy.Rate(ekf_node.ekf_rate)

    while not rospy.is_shutdown():
        ekf_node.run_estimator()
        rospy.spin()
        rate.sleep()
