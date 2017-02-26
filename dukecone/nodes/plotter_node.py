#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import Vector3


class Plotter:

    def __init__(self):

        # slightly hacky, replace with time
        # since we are plotting against time
        self.counter = 0
        incoming_measure_top = '/dukecone/estimates/meas'
        self.incoming_measure_sub = rospy.Subscriber(incoming_measure_top,
                                                     Vector3,
                                                     self.measure_callback)

    def measure_callback(self, data):
        plt.ion()
        plt.figure(2)
        plt.axis([0, 500, 0, 3])
        print(data.x, data.y)
        plt.plot(self.counter, data.x, 'ro')
        plt.plot(data.y, 'b--')
        plt.show()
        plt.pause(0.0000001)
        self.counter += 1


if __name__ == '__main__':
    plotternode = Plotter()
    rospy.init_node('Plotter', anonymous=True)

    r = rospy.Rate(50)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
