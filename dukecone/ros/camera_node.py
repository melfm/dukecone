#!/usr/bin/env python
import cv2

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageViewer(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = "/camera/rgb/image_color"
        self.image_sub = rospy.Subscriber(self.image_topic,
                                          Image,
                                          self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        print cv_image
        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    iv = ImageViewer()
    rospy.init_node('image_viewer', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
