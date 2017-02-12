import tensorflow as tf
import argparse
import os
from yolo_cnn_net import Yolo_tf

#import pdb

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

def main(_):
    yolo = Yolo_tf(FLAGS)

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
    tf.app.run()
