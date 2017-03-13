#!/usr/bin/env python

# Import modules
import math
import numpy as np
import rosbag
from rosbag_parser import rosbag_parser
# Import ROS messages
from geometry_msgs.msg import Pose2D

# Import custom messages
from dukecone.msg import ObjectLocation

# Define rosbag (all options)
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/0p85m_yolo.bag")
bag = rosbag.Bag(
    "/home/pganti/rosbags/dukecone/1p3m_yolo.bag")
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/static_2p6m_yolo.bag")
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/static_12deg_yolo.bag")
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/static_22deg_yolo.bag")
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/static_27deg_yolo.bag")
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/static_n9deg_yolo.bag")
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/static_n17deg_yolo.bag")
# bag = rosbag.Bag(
#    "/home/pganti/rosbags/dukecone/static_n25deg_yolo.bag")

class ObjLocationData:

    def __init__(self):
        self.time = []
        self.range = []
        self.bearing = []

    def parse_msg(self, ros_msg, ros_time):
        self.time.append(ros_time)
        self.range.append(ros_msg.distance)
        self.bearing.append(ros_msg.bearing)

class PoseData:

    def __init__(self):
        self.time = []
        self.x = []
        self.y = []
        self.theta = []

    def parse_msg(self, ros_msg, ros_time):
        self.time.append(ros_time)

        # Because pose data is in ENU coordinates
        # need to convert to NWU
        # This is only applicable to Mocap data
        ros_msg_enu = [ros_msg.x, ros_msg.y, ros_msg.theta]
        ros_msg_nwu = self.rotation_helper(3, ros_msg_enu)

        self.x.append(ros_msg_nwu[0])
        self.y.append(ros_msg_nwu[1])
        self.theta.append(ros_msg_nwu[2])

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


class TensorflowData:
    
    def __init__(self):
        # "/tensorflow/object/location"
        self.obj1_location = ObjLocationData()
        
    def parse_measurements(self, ros_msg, ros_time):
        self.obj1_location.parse_msg(ros_msg, ros_time)


class MocapData:

    def __init__(self):
        # "/turtlebot/ground_pose"
        self.turtlebot_pose = PoseData()
        
        # "/obj1/ground_pose"
        self.obj1_position = PoseData()
    
    def parse_pose_turtlebot(self, ros_msg, ros_time):
        self.turtlebot_pose.parse_msg(ros_msg, ros_time)
        
    def parse_position_obj(self, ros_msg, ros_time):
        self.obj1_position.parse_msg(ros_msg, ros_time)


def run_tests(mocap_data, tensorflow_data):
    # Average all info
    turtlebot_x_avg = np.mean(mocap_data.turtlebot_pose.x)
    turtlebot_y_avg = np.mean(mocap_data.turtlebot_pose.y)
    turtlebot_theta_avg = np.mean(mocap_data.turtlebot_pose.y)

    obj1_x_avg = np.mean(mocap_data.obj1_position.x)
    obj1_y_avg = np.mean(mocap_data.obj1_position.y)

    obj1_range_avg = np.mean(tensorflow_data.obj1_location.range)
    obj1_bearing_avg = np.mean(tensorflow_data.obj1_location.bearing)

    # Calculate comparison
    calc_range = np.sqrt(np.power((obj1_x_avg - turtlebot_x_avg), 2)
                         + np.power((obj1_y_avg - turtlebot_y_avg), 2))
    calc_bearing = math.atan2(obj1_y_avg - turtlebot_y_avg,
                              obj1_x_avg - turtlebot_x_avg) \
                   - turtlebot_theta_avg
    calc_bearing = calc_bearing*180.0/math.pi
    
    print('Average Values')
    print('------------------------------------')
    print('Calculated Range: ', calc_range)
    print('Measured Range: ', obj1_range_avg)
    print('Calculated Bearing: ', calc_bearing)
    print('Measured Bearing:', obj1_bearing_avg)
    
    # Calculate Standard Deviation and variance
    turtlebot_x_var = np.var(mocap_data.turtlebot_pose.x)
    turtlebot_y_var = np.var(mocap_data.turtlebot_pose.y)
    turtlebot_theta_var = np.var(mocap_data.turtlebot_pose.theta)
    
    obj1_x_var = np.var(mocap_data.obj1_position.x)
    obj1_y_var = np.var(mocap_data.obj1_position.y)
    
    obj1_range_var = np.var(tensorflow_data.obj1_location.range)
    obj1_bearing_var = np.var(tensorflow_data.obj1_location.bearing)

    print('Variance')
    print('------------------------------------')
    print('Turtlebot X Variance: ', turtlebot_x_var)
    print('Turtlebot Y Variance: ', turtlebot_y_var)
    print('Turtlebot Theta Variance: ', turtlebot_theta_var)
    print('Object 1 X Variance: ', obj1_x_var)
    print('Object 1 Y Variance: ', obj1_y_var)
    print('Object 1 Range Variance: ', obj1_range_var)
    print('Object 1 Bearing Variance: ', obj1_bearing_var)

if __name__ == '__main__':
    # Create MOCAP and Tensorflow
    mocap_data = MocapData()
    tensorflow_data = TensorflowData()
    
    # Define parameters for rosbag_parser
    params = [
        {"topic": "/turtlebot/ground_pose",
         "callback": mocap_data.parse_pose_turtlebot},
        {"topic": "/obj1/ground_pose",
         "callback": mocap_data.parse_position_obj},
        {"topic": "/tensorflow/object/location",
         "callback": tensorflow_data.parse_measurements},
    ]
    
    # Run ROSBAG parser
    rosbag_parser(bag, params)
    
    # Run tests on extracted data
    run_tests(mocap_data, tensorflow_data)
