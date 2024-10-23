#!/usr/bin/env python3
import os
import cv2
import sys
import time
import argparse
import numpy as np

# ROS
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from torobo_control_msgs.srv import CheckCollision
from torobo_sensor_msgs.msg import ToroboState

class RightHandConfig:
    def __init__(self):
        self.pinch_close_pose = np.deg2rad( [42.5, 42.5, 48.5, 48.5, 89.0, 89.0, 75.9, 4.4, 12.8, 12.8] )
        self.pinch_open_pose = self.pinch_close_pose - np.deg2rad([10, 0, 10, 0, 0, 0, 0, 0, 10, 0])

class LeftHandConfig:
    def __init__(self):
        pass

class HandClient:
    def __init__(self, namespace, side):

        TOPIC_NAME = namespace + "/" + side + '_hand_controller/command'
        self.publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)
        while self.publisher.get_num_connections() == 0:
            rospy.sleep(1)

        if side == "right":
            self.config = RightHandConfig()
        elif side == "left":
            self.config = LeftHandConfig()
        else:
            raise NotImplementedError

        # Create JointTrajectory message
        self.trajectory = self.create_joint_trajectoy_base()
        self.trajectory.joint_names = [
                side + '_hand_first_finger/joint_1',
                side + '_hand_first_finger/joint_2',
                side + '_hand_second_finger/joint_1',
                side + '_hand_second_finger/joint_2',
                side + '_hand_third_finger/joint_1',
                side + '_hand_third_finger/joint_2',
                side + '_hand_thumb/joint_1',
                side + '_hand_thumb/joint_2',
                side + '_hand_thumb/joint_3',
                side + '_hand_thumb/joint_4']
        
        
    def command(self, pose):
        self.trajectory.header.stamp = rospy.Time.now()
        self.trajectory.points[0].positions = pose.tolist()
        self.publisher.publish(self.trajectory)

    def create_joint_trajectoy_base(self):
        # Creates a message.
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Time(0)
        trajectory.points.append(point)

        return trajectory
    
    def openHand(self):
        self.command(self.config.hand_open_pose)
    
    def openPinch(self):
        self.command(self.config.pinch_open_pose)

    def closePinch(self):
        self.command(self.config.pinch_close_pose)

if __name__ == "__main__":    
    # args
    rospy.init_node('hand_controller', anonymous=True, disable_signals=True)

    # call class
    # left_hand = HandClient("/torobo", "left")
    right_hand = HandClient("/torobo", "right")

    print("#######################")
    print("Command list")
    print("e: Esc")
    print("o: Open right pinch")
    print("c: Close right pinch")
    
    while True:
        print("Please input key command")
        key = input()
        if key == "" or key == "e":
            break
        if key == "o":
            right_hand.openPinch()
        if key == "c":
            right_hand.closePinch()