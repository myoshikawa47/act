#!/usr/bin/env python3
import os
import cv2
import sys
import time
import datetime
import argparse
import numpy as np
import pickle

# ROS
import rospy
from rqt_bag.recorder import Recorder   
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from torobo_control_msgs.srv import CheckCollision
from torobo_sensor_msgs.msg import ToroboState

from hand_controller import HandClient


"""
How to use
$ cd /work/Yoshikawa/robot
$ python3 move_to_initial_pose.py
"""



class MoveToInitPose:
    def __init__(self, freq=100):
        # parameter
        self.current_left_arm = None
        self.current_right_arm = None
        
        # Joint states for subtask 1
        self.head_t0 = np.array([0.0, 0.56, -0.024])  
        self.left_arm_t0 = np.array([ -0.39913, -0.35334, -0.07564, 1.13114, 0.02698, 0.04563, 0.01495], dtype=np.float32) # down position
        self.right_arm_t0 = np.array([0.75296, -0.8325, -0.23391, 1.90076, -0.6311, 0.65651, 0.10761], dtype=np.float32) # data on 0702
        self.torso_t0 = np.array([-0.17453, 0.52345, 0.000], dtype=np.float32)
                
        # initialize ros node
        self.freq = freq
        self.r = rospy.Rate(freq)

        # service
        # self.check_collision = rospy.ServiceProxy('/torobo/check_collision', CheckCollision)

        # publisher
        self.cmd_pub = rospy.Publisher("/torobo/online_joint_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.hand_pub = rospy.Publisher('/hand_states', JointState, queue_size=1)

        # subscriber
        rospy.Subscriber("/torobo/joint_states", JointState, self.joint_callback)
        rospy.wait_for_message('/torobo/joint_states', JointState)

        # hand
        # self.left_hand = HandClient("/torobo", "left")
        self.right_hand = HandClient("/torobo", "right")

        # setup the trajectory message
        trajectory = JointTrajectory()
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Time(0)
        trajectory.points.append(point)
        trajectory.joint_names = [
            "head/joint_1",
            "head/joint_2",
            "head/joint_3",            
            "left_arm/joint_1",
            "left_arm/joint_2",
            "left_arm/joint_3",
            "left_arm/joint_4",
            "left_arm/joint_5",
            "left_arm/joint_6",
            "left_arm/joint_7",
            "right_arm/joint_1",
            "right_arm/joint_2",
            "right_arm/joint_3",
            "right_arm/joint_4",
            "right_arm/joint_5",
            "right_arm/joint_6",
            "right_arm/joint_7",
            "torso/joint_1",
            "torso/joint_2",
            "torso/joint_3",
        ]

        trajectory.header.stamp = rospy.Time.now()
        self.trajectory = trajectory

        
        self.hand_msg = JointState()
        self.hand_msg.name = ["right/state"]
        
        # initial hand state
        # 0 is open, 1 is close
        # warning: This hand states for subtask 1
        self.hand_states = [0]
        self.hand_msg.position = self.hand_states        
        
    def joint_callback(self, msg):
        self.current_head = np.array(msg.position[7:10])
        self.current_left_arm = np.array(msg.position[10:17])
        self.current_right_arm = np.array(msg.position[27:34])
        self.current_torso = np.array(msg.position[-3:])

    def initialization(self, exptime):
        self.generate_trajectory([self.current_head, self.current_left_arm, self.current_right_arm, self.current_torso], [self.head_t0, self.left_arm_t0, self.right_arm_t0, self.torso_t0], exptime)
        
        # hand initialization for subtask 1
        # input("close")
        # self.right_hand.closePinch()
        # input("open")
        self.right_hand.openPinch()
        
        rospy.logwarn("Finish initializing")


    def command(self, cmd):
        if not np.isnan(cmd).any():
            self.trajectory.header.stamp = rospy.Time.now()
            self.trajectory.points[0].positions = cmd.tolist()
            # print(self.trajectory)
            self.cmd_pub.publish(self.trajectory)

            
    def generate_trajectory(self, start_position, goal_position, time=5):
        nloop = self.freq * time
        head_trajectory = np.linspace(start_position[0], goal_position[0], nloop)
        left_arm_trajectory = np.linspace(start_position[1], goal_position[1], nloop)
        right_arm_trajectory = np.linspace(start_position[2], goal_position[2], nloop)
        torso_trajectory = np.linspace(start_position[3], goal_position[3], nloop)

        for loop_ct in range(nloop):
            cmd = np.concatenate([head_trajectory[loop_ct], left_arm_trajectory[loop_ct], right_arm_trajectory[loop_ct], torso_trajectory[loop_ct]],-1)
            self.command(cmd)
            self.hand_pub.publish(self.hand_msg)

            self.r.sleep()
        rospy.sleep(0.5)
        
    
    def control_hand(self, key):
        if key == "open":
            self.left_hand.openPinch()
            self.hand_states[0] = 0
        if key == "close":
            self.right_hand.closePinch()
            self.hand_states[0] = 1
    
    def main(self, exptime):
        self.initialization(exptime)
        # record_list = self.collect_way_point()            
        # self.playback(record_list)
        
        
if __name__ == "__main__":    
    rospy.init_node('move_to_init_pose', anonymous=True, disable_signals=True)

    # call class
    m = MoveToInitPose()
    m.main(3)
    