#!/bin/bash

roscore &

rosrun dukecone yolo_detector.py & 

sleep 5

rosrun dukecone ekf_node.py &

rosbag play ~/rosbags/dukecone/moveforward2.bag

killall -9 roscore
killall -9 rosmaster
