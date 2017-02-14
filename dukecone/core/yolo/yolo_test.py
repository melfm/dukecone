#!/usr/bin/env python
""" Testing the object detector accuracy

Unit tests testing the object detector component
Testing coverage :
- Object bounding boxes
- Number of objects
- Maybe depth?

"""
from __future__ import print_function

import unittest
import os
import argparse

import cv2
import numpy as np

import tensorflow as tf
from yolo_cnn_net import Yolo_tf

import pdb
# -----------------------------------
# Model parameters as external flags.
# -----------------------------------
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


class YoloDetectionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        current_dir = os.getcwd()
        image_dir = current_dir + '/images/'
        weight_dir = current_dir + '/weights/'

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
        cls.image_dir = image_dir + 'seen_puppy.jpg'

    def test_bounding_box(self):
        results, img_size = self.yolo.detect_from_file(self.image_dir)
        print(results)
        obj_class = results[0][0]
        self.assertEqual('dog', obj_class)
        x = int(results[0][1])
        y = int(results[0][2])
        print("weidth and height before ",int(results[0][3]) \
              , int(results[0][4]))
        w = int(results[0][3])//2
        h = int(results[0][4])//2
        print("weidth and height before ", w, h)
        #print(x,y,w,h)
        img = np.zeros((img_size[0], img_size[1], 3), np.uint8)
        img[:,:] = (255, 0, 0)
        cv2.rectangle(img, (x-w, y-h), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('bounding_box.jpg', img)




    def test_object_depth(self):
        # Q - How to test this?
        pass

    def number_of_detected_objects(self):
        pass


if __name__ == '__main__':
    unittest.main()
