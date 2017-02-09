#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageViewer(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_rgb = "/camera/rgb/image_color"
        self.image_depth = "/camera/depth_registered/image"
        self.rgb_image_sub = rospy.Subscriber(self.image_rgb,
                                              Image,
                                              self.rgb_callback)
        self.depth_image_sub = rospy.Subscriber(self.image_depth,
                                                Image,
                                                self.depth_callback)

    def rgb_callback(self, data):
        try:
            cv_rgb_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("RgbImage", cv_rgb_image)
        cv2.imwrite("rgb_image.jpg", cv_rgb_image)
        cv2.waitKey(1)

    def depth_callback(self, data):
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, "8UC1")
        except CvBridgeError as e:
            print(e)

        normalized = np.copy(cv_depth_image)
        cv2.normalize(normalized, cv_depth_image, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("Depth Image", cv_depth_image)
        cv2.imwrite("gray_image.jpg", cv_depth_image)
        cv2.waitKey(1)



if __name__ == "__main__":
    iv = ImageViewer()
    rospy.init_node('image_viewer', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
