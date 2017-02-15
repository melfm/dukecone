#!/usr/bin/env python

# Turtlebot(indigo) drawing shapes

import rospy
from geometry_msgs.msg import Twist, Vector3
from math import radians, pi


class DrawCircle():

    def __init__(self):
        # initiliaze
        rospy.init_node('drawcircle', anonymous=False)
        # Set the callback to shutdown the node
        rospy.on_shutdown(self.shutdown)

        self.cmd_vel = rospy.Publisher(
            'cmd_vel_mux/input/navi', Twist, queue_size=10)

        # 5 HZ
        r = rospy.Rate(5)
        T = 10
        radius = 0.1

        velocity_x = 2*pi*(radius/T)
        angular_z = 2*(pi/T)

        while not rospy.is_shutdown():
            rospy.loginfo("Going around circles now")
            velocities = Twist(
                            Vector3((velocity_x), 0, 0),
                            Vector3(0, 0, (angular_z)))
            rospy.loginfo(velocities)
            self.cmd_vel.publish(velocities)
            r.sleep()

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Drawing ...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


class DrawSquare():

    def __init__(self):
        # initiliaze
        rospy.init_node('drawsquare', anonymous=False)
        rospy.on_shutdown(self.shutdown)

        self.cmd_vel = rospy.Publisher(
            'cmd_vel_mux/input/navi', Twist, queue_size=10)
        # 5 HZ
        r = rospy.Rate(5)

        # let's go forward at 0.2 m/s
        move_cmd = Twist()
        move_cmd.linear.x = 0.2
        # by default angular.z is 0 so setting this isn't required

        # let's turn at 45 deg/s
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.angular.z = radians(45)  # 45 deg/s in radians/s

        # two keep drawing squares.  Go forward for 2 seconds (10 x 5 HZ) then
        # turn for 2 second
        count = 0
        while not rospy.is_shutdown():
            # go forward 0.4 m (2 seconds * 0.2 m / seconds)
            rospy.loginfo("Going Straight")
            for x in range(0, 10):
                self.cmd_vel.publish(move_cmd)
                r.sleep()
            # turn 90 degrees
            rospy.loginfo("Turning")
            for x in range(0, 10):
                self.cmd_vel.publish(turn_cmd)
                r.sleep()
            count = count + 1
            if(count == 4):
                count = 0
            if(count == 0):
                rospy.loginfo(
                    "TurtleBot should be close to the original starting position (but it's probably way off)")

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Drawing Squares")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

if __name__ == '__main__':
    try:
        DrawCircle()
    except:
        rospy.loginfo("node terminated.")
