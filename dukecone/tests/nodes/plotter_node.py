#!/usr/bin/env python

import rosbag
import numpy as np
import os
import matplotlib.pyplot as plt

from rosbag_parser import rosbag_parser

bag = rosbag.Bag(
    "/home/melissafm/Workspace/ME780/turtlydata/plot_bags/ekf_plot_bag.bag")


class Vec3Data:

    def __init__(self):
        self.vec3_time = []
        self.x = []
        self.y = []
        self.z = []

    def parse_msg(self, ros_msg, ros_time):
        self.vec3_time.append(ros_time)
        self.x.append(ros_msg.x)
        self.y.append(ros_msg.y)
        self.z.append(ros_msg.z)


class Pos2Data:

    def __init__(self):
        self.vec3_time = []
        self.x = []
        self.y = []
        self.z = []

    def parse_msg(self, ros_msg, ros_time):
        self.vec3_time.append(ros_time)

        # Because pose data is in ENU coordinates
        # need to convert to NWU
        # This is only applicable to Mocap data
        ros_msg_enu = [ros_msg.x, ros_msg.y, ros_msg.theta]
        ros_msg_nwu = self.rotation_helper(3, ros_msg_enu)
        self.x.append(ros_msg_nwu[0])
        self.y.append(ros_msg_nwu[1])
        self.z.append(ros_msg_nwu[2])

    def rotation_helper(self, pose_dim, input_vec_enu):

        if (pose_dim == 2):
            R = np.matrix('0 1; -1 0')
            input_array_enu = (np.asarray(input_vec_enu)).reshape(2, 1)
            input_vec_nwu = R * input_array_enu
            input_vec_nwu = np.asarray(input_vec_nwu).flatten()
            return input_vec_nwu

        elif (pose_dim == 3):
            # apply rotation
            R = np.matrix('0 1 0; -1 0 0; 0 0 1')
            input_array_enu = (np.asarray(input_vec_enu)).reshape(3, 1)
            input_vec_nwu = R * input_array_enu
            input_vec_nwu = np.asarray(input_vec_nwu).flatten()
            return input_vec_nwu
        else:
            print('Wrong input. Dimension has to be 2D or 3D')
            return


class EstimateData:

    def __init__(self):
        # "/dukecone/estimates/state"
        self.estimate_bot_pose = Vec3Data()

        # "/dukecone/estimates/mu"
        self.estimate_mu = Vec3Data()

        # "/dukecone/estimates/mup"
        self.estimate_mup = Vec3Data()

        # "/dukecone/estimates/meas"
        self.incoming_measure = Vec3Data()

    def parse_incoming_measures(self, ros_msg, ros_time):
        self.incoming_measure.parse_msg(ros_msg, ros_time)

    def parse_position_estimate(self, ros_msg, ros_time):
        self.estimate_pose_pose.parse_msg(ros_msg, ros_time)

    def parse_mu_estimate(self, ros_msg, ros_time):
        self.estimate_mu.parse_msg(ros_msg, ros_time)

    def parse_mup_estimate(self, ros_msg, ros_time):
        self.estimate_mup.parse_msg(ros_msg, ros_time)


class MocapData:

    def __init__(self):
        # "/turtlebot/ground_pose"
        self.position_turtle = Pos2Data()

        # "/ob1/ground_pose"
        self.position_object = Pos2Data()

    def parse_position_body(self, ros_msg, ros_time):
        self.position_turtle.parse_msg(ros_msg, ros_time)

    def parse_position_object(self, ros_msg, ros_time):
        self.position_object.parse_msg(ros_msg, ros_time)


def plot_ekf_estimates(est_data):

    # change directory to store in the plot dir
    os.chdir('../')

    #########################
    # Plot incoming measures
    #########################
    plt.figure()
    plt.subplot(211)
    plt.plot(est_data.incoming_measure.vec3_time,
             est_data.incoming_measure.x,
             label="Incoming measure y - Range")

    plt.title("Measurement Y-Input to EKF")
    plt.xlabel("Time (s)")
    plt.ylabel("Range (m)")

    plt.subplot(212)
    plt.plot(est_data.incoming_measure.vec3_time,
             est_data.incoming_measure.y,
             label="Incoming measure y - Bearing")
    plt.xlabel("Time (s)")
    plt.ylabel("Bearing (rad)")
    plt.savefig("test_plots/measure-y.png")

    plt.subplots_adjust(bottom=0.08, hspace=0.9)

    #########################
    # Plot ekf estimations
    #########################

    plt.figure(figsize=(7.0, 7.0))
    plt.subplot(211)
    plt.plot(est_data.estimate_mu.x,
             est_data.estimate_mu.y,
             'r--',
             label="Ekf estimates")
    plt.plot(est_data.estimate_mup.x,
             est_data.estimate_mup.y,
             'b--')

    plt.title("Estimated Target mu 2D")
    plt.xlabel("X-position (m)")
    plt.ylabel("Y-position (m)")

    plt.subplot(212)
    plt.plot(est_data.estimate_mu.vec3_time,
             est_data.estimate_mu.z,
             'b-')
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (rad)")

    plt.savefig("test_plots/estimation.png")


def plot_mocap_data(mocap_data):

    #########################
    # Plot Mocap data
    #########################
    plt.figure()

    plt.axis([mocap_data.position_turtle.x[0],
              mocap_data.position_turtle.x[-1]+2,
              mocap_data.position_turtle.y[0]-0.2,
              mocap_data.position_turtle.y[-1]+0.2])

    plt.plot(mocap_data.position_turtle.x,
             mocap_data.position_turtle.y,
             label="Turtlebot position gound truth")

    plt.plot(mocap_data.position_object.x,
             mocap_data.position_object.y,
             'bo',
             label="Object position gound truth")

    plt.title("Mocap Object Pose")
    plt.xlabel("X-pose (m)")
    plt.ylabel("Y-pose (m)")
    plt.savefig("test_plots/mocap_data.png")


if __name__ == '__main__':

    mocap_data = MocapData()
    est_data = EstimateData()

    params = [
        {"topic": "/turtlebot/ground_pose",
         "callback": mocap_data.parse_position_body},
        {"topic": "/obj1/ground_pose",
         "callback": mocap_data.parse_position_object},
        {"topic": "/dukecone/estimates/meas",
         "callback": est_data.parse_incoming_measures},
        {"topic": "/dukecone/estimates/mu",
         "callback": est_data.parse_mu_estimate},
        {"topic": "/dukecone/estimates/mup",
         "callback": est_data.parse_mup_estimate}
    ]

    rosbag_parser(bag, params)
    print('Plotting...')
    plot_ekf_estimates(est_data)
    plot_mocap_data(mocap_data)
