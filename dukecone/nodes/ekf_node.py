#!/usr/bin/env python

import core.ekf_base as ekf
import rospy
import numpy as np

from geometry_msgs.msg import Twist, Vector3, Pose2D
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
        mu_estimate_top = '/dukecone/estimates/mu'
        bot_state_top = '/dukecone/estimates/state'

        # MOCAP subscribers
        bot_mocap_top = '/turtlebot/ground_pose'
        self.bot_mocap_sub = rospy.Subscriber(
                                              bot_mocap_top,
                                              Pose2D,
                                              self.bot_mocap_callback)

        obj1_mocap_top = '/obj1/ground_pose'
        self.obj1_mocap_sub = rospy.Subscriber(
                                              obj1_mocap_top,
                                              Pose2D,
                                              self.obj1_mocap_callback)

        self.pub_incoming_meas = rospy.Publisher(
                                    incoming_measure_top,
                                    Vector3,
                                    queue_size=1)

        self.pub_mup_est = rospy.Publisher(
                                mup_estimate_top,
                                Vector3,
                                queue_size=1)

        self.pub_mu_est = rospy.Publisher(
                                mu_estimate_top,
                                Vector3,
                                queue_size=1)

        self.pub_state_est = rospy.Publisher(
                                bot_state_top,
                                Vector3,
                                queue_size=1)

        self.feat_x_center = None
        self.feat_y_center = None
        self.feat_range = 0.0
        self.feat_bearing = 0.0

        # initial rate - update once the inputs
        # start coming in
        self.dt = 0
        self.input_timer = True
        self.dt0 = None
        self.dt1 = None

        # Define initial state and prior
        # Assume we have perfect knowledge of prior
        x0 = [0, 0, 0]
        mu = [0, 0, 0]
        # number of states
        n = len(x0)
        # Define initial covariance
        S = 0.1*np.identity(n)
        self.ekf = ekf.EKF(x0, mu, S, self.dt)

        # Define MOCAP variables
        self.bot_mocap_pose = []

    def bot_input_callback(self, data):
        bot_linear_x = data.linear.x
        bot_omega = data.angular.z
        self.ekf.update_input([bot_linear_x, bot_omega])
        if (self.input_timer):
            self.dt0 = rospy.get_rostime()
            self.input_timer = False
        else:
            self.dt1 = rospy.get_rostime()
            self.input_timer = True
            # update dt
            self.dt = (self.dt1 - self.dt0).to_sec()
            #print('Updated dt ->', self.dt)
            # update ekf dt
            self.ekf.dt = self.dt

        self.run_estimator()

    def obj_callback(self, data):
        """ Get position of closest detected object"""
        self.feat_x_center = data.x_center
        self.feat_y_center = data.y_center
        self.feat_range = data.distance
        # Update measurement
        self.ekf.set_measurement(self.feat_range,
                                 self.feat_bearing)

    def bot_mocap_callback(self, data):
        # Robot pose groundtruth
        self.bot_mocap_pose = [data.x, data.y, data.theta]
        # print('bot_mocap_pose:', self.bot_mocap_pose[0],
        #      self.bot_mocap_pose[1],
        #      self.bot_mocap_pose[2])

    def obj1_mocap_callback(self, data):
        new_mf = [data.x, data.y]
        R = np.matrix('0 1; -1 0')
        mf_enu = (np.asarray(new_mf)).reshape(2, 1)
        mf_nwu = R * mf_enu
        updated_mf = np.asarray(mf_nwu).flatten()

        self.ekf.update_feat_mf(updated_mf)

    def make_measure_topic(self, input_y):
        measure_msg = Vector3()
        measure_msg.x = input_y[0]
        measure_msg.y = input_y[1]
        measure_msg.z = 0.0   # not using this, ignore

        return measure_msg

    def make_estimate_topics(self, states, mu, mup):
        states_msg = Vector3()
        states_msg.x = states[0]
        states_msg.y = states[1]
        states_msg.z = states[2]

        mu_msg = Vector3()
        mu_msg.x = mu[0]
        mu_msg.y = mu[1]
        mu_msg.z = mu[2]

        mup_msg = Vector3()
        mup_msg.x = mup[0]
        mup_msg.y = mup[1]
        mup_msg.z = mup[2]

        return states_msg, mu_msg, mup_msg

    def run_estimator(self):
        # this needs to be called
        # when there is input
        self.ekf.do_estimation()

        # publish measurements
        # rospy.loginfo(self.pub_incoming_meas)
        pub_meas_msg = self.make_measure_topic(self.ekf.y)
        self.pub_incoming_meas.publish(pub_meas_msg)

        # publish estimates
        pub_state_msg, pub_state_mu, pub_mup_msg = self.make_estimate_topics(
            self.ekf.bot.state, self.ekf.mu, self.ekf.mup)
        self.pub_state_est.publish(pub_state_msg)
        self.pub_mu_est.publish(pub_state_mu)
        self.pub_mup_est.publish(pub_mup_msg)


if __name__ == '__main__':
    rospy.init_node('EKFNode', anonymous=True)
    ekf_node = EKFNode()
    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        ekf_node.run_estimator()
        rate.sleep()
