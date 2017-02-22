#!/usr/bin/env python
import sys
sys.path.insert(0, '../core/')
import ekf_base as ekf
import rospy

from geometry_msgs.msg import Twist
from dukecone.msg import ObjectLocation


class EKFNode():

    def __init__(self):
        self.ekf = ekf.EKF()
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
        self.bot_linear_x = 0.2
        self.feat_x_center = None
        self.feat_y_center = None
        self.feat_distance = None
        self.feat_bearing = None

    def bot_input_callback(self, data):
        pass
        # Extract velocity data from Twist message
        #vel = data.linear.x
        #omega = data.angular.z
        #new_input = [vel, omega]
        # Send new input to EKF class
        # self.ekf.update_input(new_input)

    def obj_callback(self, data):
        """ Get position of closest detected object"""
        self.feat_x_center = data.x_center
        self.feat_y_center = data.y_center
        self.feat_distance = data.distance

    def calculate_bearing(self):
        # Make assumptions and based on those
        # calculate the bearing from the detected
        # object location

        # for now though assume static
        self.feat_bearing = 0

    def run_estimator(self):

        # Update input
        self.ekf.update_input([self.bot_linear_x, 0])
        # Update measurement
        self.ekf.update_measurement_static(self.feat_bearing,
                                           self.feat_distance)

        self.ekf.do_estimation()


if __name__ == '__main__':

    ekf_node = EKFNode()
    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        ekf_node.run_estimator()
        rospy.spinOnce()
        rate.sleep()
