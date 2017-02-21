#!/usr/bin/env python

# Import Modules
from core import ekf_base as ekf

# Import math library
from math import radians, pi
import numpy as np

# Import ROS modules and messages
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from dukecone.msg import ObjectLocation


# Define EKFNode class
class EKFNode():
    # Define initialization
    def __init__(self):
        # Initialize ROS Node
        rospy.init_node('EKFNode', anonymous=True)

        # Create instance of EKF() class
        self.ekf = ekf.EKF()

        # Define subscribers for input and measurements
        bot_cmd_topic = '/cmd_vel_mux/input/navi'
        self.ekf_sub_input = rospy.Subscriber(
                                             bot_cmd_topic,
                                             Twist,
                                             self.bot_input_callback)
        tf_obj_topic = '/tensorflow/object/location'
        self.ekf_sub_obj = rospy.Subscriber(tf_obj_topic,
                                            ObjectLocation,
                                            self.obj_callback)

    def bot_input_callback(self, data):
        pass

    def obj_callback(self, data):
        pass


if __name__ == '__main__':

    ekf_node = EKFNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
