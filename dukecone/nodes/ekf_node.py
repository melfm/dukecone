#!/usr/bin/env python

from core import EKF as ekf

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from dukecone.msg import ObjectLocation
from math import radians, pi

class EKFNode():

    def __init__(self):

        self.ekf = ekf.EKF()

        bot_cmd_topic = 'cmd_vel_mux/input/navi'
        self.ekf_sub_input = rospy.Subscriber(
                                             bot_cmd_topic,
                                             Twist,
                                             self.bot_input_callback)
        tf_obj_topic = 'tensorflow/object/location'
        self.ekf_sub_obj = rospy.Subscriber(tf_obj_topic,
                                            ObjectLocation,
                                            self.obj_callback)


    def bot_input_callback(self):
        pass


    def obj_callback(self):
        pass


if __name__ == '__main__':

    ekf_node = EKFNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("bye bye")
