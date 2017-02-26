#!/usr/bin/env python
import sys
sys.path.insert(0, '../core/')
import ekf_base as ekf
import rospy
import numpy as np
import time

from geometry_msgs.msg import Twist, Vector3
from dukecone.msg import ObjectLocation


class EKFNode():

    def __init__(self):

        # Publishers and subscribers
        bot_cmd_topic = '/cmd_vel_mux/input/navi'
        self.ekf_sub_input = rospy.Subscriber(
                                             bot_cmd_topic,
                                             Twist,
                                             self.bot_input_callback)
        tf_obj_topic = '/tensorflow/object/location'
        self.ekf_sub_obj = rospy.Subscriber(tf_obj_topic,
                                            ObjectLocation,
                                            self.obj_callback)
        # inputs
        incoming_measure_top = '/dukecone/estimates/meas'

        # estimates
        mup_estimate_top = '/dukecone/estimates/mup'
        bot_state_top = '/dukecone/estimates/state'
        prior_mean_top = '/dukecone/estimates/mean'
        covariance_S_top = '/dukecone/estimates/covariance'
        obj_coords_top = '/dukecone/estimates/obj_2dcoord'

        self.pub_incoming_meas = rospy.Publisher(
                                    incoming_measure_top,
                                    Vector3,
                                    queue_size=1)

        self.pub_mup_est = rospy.Publisher(
                                mup_estimate_top,
                                Vector3,
                                queue_size=1)

        self.pub_state_est = rospy.Publisher(
                                bot_state_top,
                                Vector3,
                                queue_size=1)

        # For now we are using static commands
        self.bot_linear_x = 0.1
        self.feat_x_center = None
        self.feat_y_center = None
        self.feat_range = 0.0
        self.feat_bearing = 0.0

        # Define operating rates for ekf
        self.ekf_rate = 50
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
        # TODO:Extract velocity data from Twist message
        self.ekf.update_input([self.bot_linear_x, 0])

    def obj_callback(self, data):
        """ Get position of closest detected object"""
        self.feat_x_center = data.x_center
        self.feat_y_center = data.y_center
        self.feat_range = data.distance
        # Update measurement
        self.ekf.set_measurement(self.feat_range,
                                 self.feat_bearing)

        # TODO remove this from here, it should be
        # called inside the input callback which
        # atm is not being called
        self.ekf.update_input([0.1, 0])
        self.run_estimator()

    def make_measure_topic(self, input_y):
        measure_msg = Vector3()
        measure_msg.x = input_y[0]
        measure_msg.y = input_y[1]
        measure_msg.z = 0.0   # not using this, ignore

        return measure_msg

    def make_estimate_topics(self, states, mup):
        states_msg = Vector3()
        states_msg.x = states[0]
        states_msg.y = states[1]
        states_msg.z = states[2]

        mup_msg = Vector3()
        mup_msg.x = mup[0]
        mup_msg.y = mup[1]
        mup_msg.z = mup[2]

        return states_msg, mup_msg

    def run_estimator(self):
        # this needs to be called
        # when there is input
        self.ekf.do_estimation()

        # publish measurements
        # rospy.loginfo(self.pub_incoming_meas)
        pub_meas_msg = self.make_measure_topic(self.ekf.y)
        self.pub_incoming_meas.publish(pub_meas_msg)

        # publish estimates
        pub_state_msg, pub_mup_msg = self.make_estimate_topics(
            self.ekf.bot.state, self.ekf.mup)
        self.pub_state_est.publish(pub_state_msg)
        self.pub_mup_est.publish(pub_mup_msg)


if __name__ == '__main__':
    rospy.init_node('EKFNode', anonymous=True)
    ekf_node = EKFNode()
    rate = rospy.Rate(50)
    counter = 0

    while not rospy.is_shutdown():
        ekf_node.run_estimator()
        rate.sleep()
        counter += 1
        # after some time do some plotting
        if(counter == 1000):
            ekf_node.ekf.plot()
            time.sleep(20)
            sys.exit()
