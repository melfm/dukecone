#!/usr/bin/env python

import sys
import rospy
import numpy as np
import time
import tf

from geometry_msgs.msg import Pose2D, PoseStamped


class MocapTestNode():

    def __init__(self):

        # Define subscribers
        bot_topic = '/turtlebot/pose'
        self.turtlebot_pose_sub = rospy.Subscriber(
                                                  bot_topic,
                                                  PoseStamped,
                                                  self.turtlebot_pose_callback)

        obj1_topic = '/obj1/pose'
        self.obj1_position_sub = rospy.Subscriber(
                                              obj1_topic,
                                              PoseStamped,
                                              self.obj1_position_callback)

        # Create variables to store turtlebot pose and obj1 pose
        self.turtlebot_pose = []
        self.obj1_position = []

    def turtlebot_pose_callback(self, msg):
        # Extract position data from message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # Convert msg orientation to quaternion
        self.msg_to_quaternion(msg)

        # Convert quaternion to roll, pitch, yaw (radians)
        rpy = tf.transformations.euler_from_quaternion(self.q)

        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        self.turtlebot_pose = [x, y, yaw]

    def obj1_position_callback(msg):
        # Extract position data from message
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        self.obj1_position = [x, y]

    def msg_to_quaternion(self, msg):
        x = msg.pose.orientation.y
        y = msg.pose.orientation.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w

        self.q = [x, y, z, w]


if __name__ == '__main__':
    rospy.init_node('mocap_test', anonymous=True)
    mocap_node = MocapTestNode()

    rospy.spin()
