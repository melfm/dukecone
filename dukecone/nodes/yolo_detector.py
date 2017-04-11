#!/usr/bin/env python2
import sys
import argparse
import numpy as np
import cv2
import math
import pickle

import tensorflow as tf
import rospy

from copy import deepcopy
from sensor_msgs.msg import Image
from dukecone.msg import ObjectLocation
from cv_bridge import CvBridge, CvBridgeError

from core.yolo.yolo_cnn_net import Yolo_tf

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
        self.image_depth_topic = "/camera/depth_registered/image_raw"
        self.rgb_image_sub = rospy.Subscriber(self.image_rgb_topic,
                                              Image,
                                              self.image_callback)
        self.depth_image_sub = rospy.Subscriber(self.image_depth_topic,
                                                Image,
                                                self.depth_callback)

        self.image_depth = None
        self.tf_topic = 'tensorflow/object/location'
        self.pub_img_pos = rospy.Publisher(
            self.tf_topic, ObjectLocation, queue_size=1)

        self.test_mode = False
        self.img_width = None
        self.img_height = None
        self.depth_width = None
        self.depth_height = None

    def image_callback(self, data):
        try:
            image_depth_copy = deepcopy(self.image_depth)
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.img_width, self.img_height, _ = cv_image.shape
            # Do detection
            results = yolo.detect_from_kinect(cv_image)
            # Get the distance
            if(image_depth_copy is not None):
                self.calculate_distance(results, image_depth_copy)
                if (self.test_mode):
                    cv2.imwrite('image_test2_rgb.jpg', cv_image)
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            if (self.test_mode):
                with open('depth_test2.data', 'w') as f:
                    pickle.dump(cv_depth_image, f)
            self.depth_width, self.depth_height, _ = cv_depth_image.shape
            self.image_depth = cv_depth_image
        except CvBridgeError as e:
            print(e)

    def calculate_distance(self, results, image_depth):
        # Only publish if you see an object and the closest
        # Loop through all the bounding boxes and find min
        object_depth = sys.maxsize
        detected = False
        #bounding_box = None
        for i in range(len(results)):
            # check for objects pre-defined in mocap
            current_feat = results[i][0]
            if(current_feat == 'car' or
                    current_feat == 'tvmonitor' or
                    current_feat == 'dog'):
                detected = True
                location = self.get_object_2dlocation(i, results)
                #x = location[0]
                #y = location[1]
                w = location[2]
                h = location[3]
                x_center = location[4]
                y_center = location[5]

                # sanity check
                if(x_center > 640 or y_center > 480):
                    break

                center_pixel_depth = image_depth[y_center, x_center]
                distance_avg = self.depth_region(
                    image_depth, y_center, x_center, w, h)
                # convert to mm
                distance = float(center_pixel_depth) * 0.001
                # print("Distance of object {} from target : \
                #      {}".format(i, distance))

                # print("Averaged distance of object {} : "
                #      .format(distance_avg))

                # self.draw_bounding_box(results, i)
                if distance < object_depth:
                    object_depth = distance_avg
                    #bounding_box = [x, y, w, h]
                    object_name = results[i][0]

        if(detected):
            # Publish the distance and bounding box
            object_loc = [x_center, y_center]
            measurements = self.calculate_bearing(object_loc, object_depth)

            bearing = measurements[0]
            object_range = measurements[1]

            object_topic = self.construct_topic(
                            object_depth,
                            x_center,
                            y_center,
                            bearing,
                            object_range,
                            object_name)

            # rospy.loginfo(self.pub_img_pos)
            self.pub_img_pos.publish(object_topic)

    def depth_region(self, depth_map, y_center, x_center, w, h):
        # grab depths along a strip and take average
        # go half way
        starting_width = w/4
        end_width = w - starting_width
        x_center = x_center - starting_width
        pixie_avg = 0.0

        for i in range(starting_width, end_width):
            assert (depth_map.shape[1] > end_width)
            assert (depth_map.shape[1] > x_center)
            pixel_depth = depth_map[y_center, x_center]
            pixie_avg += pixel_depth
            x_center += 1

        pixie_avg = (pixie_avg/(end_width - starting_width)) * 0.001
        return float(pixie_avg)

    def calculate_bearing(self, object_loc, object_depth):
        # only consider horizontal FOV.
        # Bearing is only in 2D
        horiz_fov = 57.0  # degrees

        # Define Kinect image params
        image_width = 640  # Pixels

        # Calculate Horizontal Resolution
        horiz_res = horiz_fov/image_width

        # location of object in pixels.
        # Measured from center of image.
        # Positive x is to the left, positive y is upwards
        obj_x = image_width/2.0 - object_loc[0]

        # Calculate angle of object in relation to center of image
        bearing = obj_x*horiz_res        # degrees
        bearing = bearing*math.pi/180.0  # radians

        # Calculate true range, using measured bearing value.
        # Defined as depth divided by cosine of bearing angle
        if np.cos(bearing) != 0.0:
            object_range = object_depth/np.cos(bearing)
        else:
            object_range = object_depth

        measurements = [bearing, object_range]

        return measurements

    def get_object_2dlocation(self, index, results):
        x = int(results[index][1])
        y = int(results[index][2])
        w = int(results[index][3])//2
        h = int(results[index][4])//2

        x1 = x - w
        y1 = y - h
        x2 = x + w
        y2 = y + h
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        return [x, y, w, h, x_center, y_center]

    def construct_topic(self, distance, x_center, y_center,
                        bearing, object_range, object_name):
        obj_loc = ObjectLocation()
        obj_loc.x_center = x_center
        obj_loc.y_center = y_center
        obj_loc.distance = distance
        obj_loc.bearing = bearing
        obj_loc.true_range = object_range
        obj_loc.tag = object_name
        return obj_loc

    # use this function to draw the bounding box
    # of the detected object. For testing purposes
    def draw_bounding_box(self, boxes, index):
        img = np.zeros((self.img_width, self.img_height, 3), np.uint8)
        img[:, :] = (255, 0, 0)
        location = self.get_object_2dlocation(index, boxes)
        x = location[0]
        y = location[1]
        w = location[2]
        h = location[3]
        x_center = location[4]
        y_center = location[5]

        cv2.rectangle(img, (x-w, y-h), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(img, (x_center, y_center), 5, (0, 0, 255), -1)
        bb_name = 'bounding_box_{0}.jpg'.format(index)
        cv2.imwrite(bb_name, img)

if __name__ == '__main__':
    parent_dir = sys.path[0]
    image_dir = parent_dir + 'core/yolo/images/'
    weight_dir = parent_dir + 'core/yolo/weights/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--weights',
                        type=str,
                        default=weight_dir + 'YOLO_small.ckpt')
    parser.add_argument(
                        '--load_test_file',
                        type=str,
                        default=image_dir + 'puppy.jpg')
    parser.add_argument(
                        '--save_test_file',
                        type=str,
                        default=image_dir + 'puppy_out.jpg')
    parser.add_argument('--alpha', type=int, default=0.1)
    parser.add_argument('--threshold', type=int, default=0.2)
    parser.add_argument('--iou_threshold', type=int, default=0.5)
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--num_box', type=int, default=2)
    parser.add_argument('--grid_size', type=int, default=7)
    parser.add_argument('--image_width', type=int, default=480)
    parser.add_argument('--image_height', type=int, default=640)

    FLAGS = parser.parse_args(rospy.myargv()[1:])

    # Create yolo detector instance
    yolo = Yolo_tf(FLAGS)
    # Run the Yolo ros node
    yolonode = YoloNode(yolo)
    rospy.init_node('YoloNode', anonymous=True)
    r = rospy.Rate(30)

    try:
        rospy.spin()
        r.sleep()
    except KeyboardInterrupt:
        print("Shutting down")
