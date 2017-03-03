#!/usr/bin/env python

from __future__ import print_function

import unittest
import os
import sys
import argparse
import cv2
import numpy as np

import tensorflow as tf

sys.path.insert(0, '../core/yolo')
from yolo_cnn_net import Yolo_tf
from yolo_detector import YoloNode

# -----------------------------------
# Model parameters as external flags.
# -----------------------------------


import pdb
class MeasurementTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.chdir('../testing')
        current_dir = os.getcwd()
        cls.file_dir = current_dir
        os.chdir('../core')
        current_dir = os.getcwd()
        image_dir = current_dir + '/yolo/images/'
        weight_dir = current_dir + '/yolo/weights/'

        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--weights',
            type=str,
            default=weight_dir +
            'YOLO_small.ckpt')
        parser.add_argument(
            '--load_test_file',
            type=str,
            default=image_dir +
            'puppy.jpg')
        parser.add_argument(
            '--save_test_file',
            type=str,
            default=image_dir +
            'seen_puppy_out.jpg')
        parser.add_argument('--alpha', type=int, default=0.1)
        parser.add_argument('--threshold', type=int, default=0.2)
        parser.add_argument('--iou_threshold', type=int, default=0.5)
        parser.add_argument('--num_class', type=int, default=20)
        parser.add_argument('--num_box', type=int, default=2)
        parser.add_argument('--grid_size', type=int, default=7)
        parser.add_argument('--image_width', type=int, default=480)
        parser.add_argument('--image_height', type=int, default=640)

        FLAGS = parser.parse_args()

        # Create yolo detector instance
        cls.yolo = Yolo_tf(FLAGS)
        cls.detector_node = YoloNode(cls.yolo)
        cls.yolo.disp_console = False

    def test_bearing(self):
        test_img_file = self.file_dir + '/image_test1.jpg'
        detection_res = self.yolo.detect_from_file(test_img_file)

        # result should contain one object
        pdb.set_trace()
        x_center, y_center = self.detector_node.get_object_center(0, detection_res)
        car_obj_center = [x_center, y_center]

        self.detector_node.calculate_bearing(car_obj_center)




if __name__ == '__main__':
    unittest.main()

