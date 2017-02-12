#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageViewer(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = "/camera/depth_registered/image_raw"
        self.image_sub = rospy.Subscriber(self.image_topic,
                                          Image,
                                          self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
            depth_image = cv_image.astype(np.uint8)

        except CvBridgeError as e:
            print(e)

        print cv_image
        #cv2.imshow("Image", cv_image)
        cv2.imwrite("gray_image.jpg", depth_image)
        print(np.argmax(depth_image))
        print depth_image.dtype
        cv2.waitKey(1)


if __name__ == "__main__":
    iv = ImageViewer()
    rospy.init_node('image_viewer', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
