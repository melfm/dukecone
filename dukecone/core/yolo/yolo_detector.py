#!/usr/bin/env python
import os
from copy import deepcopy
import argparse

import tensorflow as tf
from yolo_cnn_net import Yolo_tf

import rospy
from sensor_msgs.msg import Image
from dukecone.msg import *
from cv_bridge import CvBridge, CvBridgeError

# Model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('threshold', 0.2, 'YOLO sensitivity threshold')
flags.DEFINE_integer('alpha', 0.1, 'YOLO alpha')
flags.DEFINE_integer('iou_threshold', 0.5, 'YOLO iou threshold')
flags.DEFINE_integer('num_class', 20, 'Number of classes')
flags.DEFINE_integer('num_box', 2, 'Number of boxes')
flags.DEFINE_integer('grid_size', 7, 'Number of boxes')
flags.DEFINE_integer('image_width', 480, 'Width of image')
flags.DEFINE_integer('image_height', 640, 'Height of image')


class YoloNode(object):
    def __init__(self, yolo):
        self.bridge = CvBridge()
        self.image_rgb_topic = "/camera/rgb/image_color"
        self.image_depth_topic  = "/camera/depth_registered/image"
        self.rgb_image_sub = rospy.Subscriber(self.image_rgb_topic,
                                              Image,
                                              self.image_callback)
        self.image_depth = None
        self.tf_topic = 'tensorflow/object/location'
        self.pub_img_pos = rospy.Publisher(self.tf_topic, ObjectLocation, queue_size=1)

    def image_callback(self, data):
        try:
            image_depth_copy = deepcopy.copy(self.image_depth)
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Do detection
            results = yolo.detect_from_kinect(cv_image)
            # Get the distance
            self.calculate_distance(results, image_depth_copy)
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.image_depth = cv_depth_image
        except CvBridgeError as e:
            print(e)

    def calculate_distance(self, results, image_depth):
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2

            x_center = (x + w) / 2
            y_center = (y + h) / 2
            print("X center, y_center", x_center, " " , y_center)

            center_pixel_depth = image_depth[x_center, y_center]
            distance = float(center_pixel_depth)
            print("Distance from target: ", distance)
            # Publish the distance and bounding box
            object_topic = construct_topic(results, distance)
            rospy.loginfo(self.pub_img_pos)
            self.pub_img_pos.publish(object_topic)

    def construct_topic(self, bounding_box, distance):
            pass



if __name__ == '__main__':
    current_dir = os.getcwd()
    image_dir = current_dir + '/images/'
    weight_dir = current_dir + '/weights/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type = str, default= weight_dir + 'YOLO_small.ckpt')
    parser.add_argument('--load_test_file', type = str, default = image_dir + 'puppy.jpg')
    parser.add_argument('--save_test_file', type = str, default = image_dir + 'puppy_out.jpg')
    parser.add_argument('--alpha', type =int, default= 0.1)
    parser.add_argument('--threshold', type =int, default= 0.2)
    parser.add_argument('--iou_threshold', type =int, default= 0.5)
    parser.add_argument('--num_class', type =int, default= 20)
    parser.add_argument('--num_box', type =int, default= 2)
    parser.add_argument('--grid_size', type =int, default= 7)
    parser.add_argument('--image_width', type =int, default= 480)
    parser.add_argument('--image_height', type =int, default= 640)

    FLAGS = parser.parse_args()

    # Create yolo detector instance
    yolo = Yolo_tf(FLAGS)
    # Run the Yolo ros node
    yolonode = YoloNode(yolo)
    rospy.init_node('YoloNode', anonymous=True)
    r = rospy.Rate(50)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
