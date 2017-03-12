#!/usr/bin/env python2


def rosbag_parser(bag, params, time_end=None, time_start=None):
    topics = []
    # parse topics
    for param in params:
        topics.append(param["topic"])

    # parse
    time_init = None
    for ros_topic, ros_msg, ros_time in bag.read_messages(topics=topics):
        time_init = float(ros_time.to_sec())
        break

    for ros_topic, ros_msg, ros_time in bag.read_messages(topics=topics):
        time = float(ros_time.to_sec()) - time_init

        # end early
        if time_end is not None and time > time_end:
            return

        # only parse messages if start time condition is reached
        if time_start is not None and time > time_start:
            for param in params:
                if ros_topic == param["topic"]:
                    param["callback"](ros_msg, time)

        elif time_start is None:
            for param in params:
                if ros_topic == param["topic"]:
                    param["callback"](ros_msg, time)
