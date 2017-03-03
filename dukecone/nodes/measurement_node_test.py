#!/usr/bin/env python

from __future__ import print_function

import unittest
import os
import sys
import argparse
import pickle

sys.path.insert(0, '../core/yolo')
from yolo_cnn_net import Yolo_tf
from yolo_detector import YoloNode

# -----------------------------------
# Model parameters as external flags.
# -----------------------------------


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
        detection_res, _ = self.yolo.detect_from_file(test_img_file)

        # result should contain one object
        self.assertTrue(detection_res)
        self.assertEqual(len(detection_res), 1, 'wrong number of objects')
        res = self.detector_node.get_object_2dlocation(0, detection_res)
        car_obj_center = [res[4], res[5]]

        bearing_angle = self.detector_node.calculate_bearing(car_obj_center)

        self.assertAlmostEqual(bearing_angle, 6.543126,
                               places=6,
                               msg='Wrong bearing angle')

    def test_depth(self):
        test_rgb_file = self.file_dir + '/image_test2_rgb.jpg'
        test_depth_file = self.file_dir + '/depth_test2.data'

        detection_res, _ = self.yolo.detect_from_file(test_rgb_file)
        with open(test_depth_file) as f:
            depth_image = pickle.load(f)

        self.assertTrue(detection_res)
        self.assertEqual(len(detection_res), 1, 'wrong number of objects')
        res = self.detector_node.get_object_2dlocation(0, detection_res)
        w = res[2]
        h = res[3]
        x_center = res[4]
        y_center = res[5]

        distance_avg = self.detector_node.depth_region(
            depth_image, y_center, x_center, w, h)

        self.assertAlmostEquals(distance_avg,
                                1.09225,
                                msg='Wrong averaged distance')

if __name__ == '__main__':
    unittest.main()
