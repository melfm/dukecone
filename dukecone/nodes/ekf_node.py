#!/usr/bin/env python
import sys
sys.path.insert(0, '../core/')
import ekf_base as ekf
import rospy
import numpy as np
import time

from geometry_msgs.msg import Twist
from dukecone.msg import ObjectLocation

class EKFNode():

    def __init__(self):

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
        self.ekf_rate = 100
        dt = 1.0/self.ekf_rate

        # Define initial state and prior
        # Assume we have perfect knowledge of prior
        x0 = [0, 0, 0]
        mu = [0, 0, 0]
        # number of states
        n = len(x0)
        # Define initial covariance
        S = 0.1*np.identity(n)
        self.ekf = ekf.EKF(x0, mu, S, dt)

    def bot_input_callback(self, data):
        # Extract velocity data from Twist message
        self.ekf.update_input([self.bot_linear_x, 0])

    def obj_callback(self, data):
        """ Get position of closest detected object"""
        self.feat_x_center = data.x_center
        self.feat_y_center = data.y_center
        self.feat_range = data.distance
        # Update measurement
        self.ekf.update_measurement(self.feat_range,
                                    self.feat_bearing)


    def calculate_bearing(self, x_center, y_center):
        pass

    def run_estimator(self):
        self.ekf.do_estimation()

if __name__ == '__main__':
    rospy.init_node('EKFNode', anonymous=True)
    ekf_node = EKFNode()
    rate = rospy.Rate(50)
    counter = 0

    while not rospy.is_shutdown():
        ekf_node.run_estimator()
        rate.sleep()
        counter +=1
        if(counter == 1000):
            ekf_node.ekf.plot()
            time.sleep(10)
            print('restarting')
            counter = 0

