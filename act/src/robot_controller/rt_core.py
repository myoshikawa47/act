#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import argparse
import time
import numpy as np
import rospy

# ROS
import rospy
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, TransformStamped

from .constants import TOPICS, MODALITY_TO_TOPIC, CAMERA_HEIGHT, CAMERA_WIDTH, CROP_PARAMS


class RTCore:
    def __init__(self):
        # load initial data
        self.img_head_left = None
        self.img_head_right = None
        self.img_arm_left = None
        self.img_arm_right = None
        self.head_state = None
        self.left_arm_state = None
        self.left_arm_cmd_state = None
        self.right_arm_state = None
        self.right_arm_cmd_state = None
        # self.pose_larm = None
        # self.obj_pos = None
        # self.task_index = None # to be overwritten
        # self.bridge = CvBridge()
        
        self.head_msg = JointState()
        self.left_arm_msg = JointState()
        self.right_arm_msg = JointState()
        
        # self.head_pub = rospy.Publisher("/maharo/head/command_states", JointState, queue_size=1)
        self.head_pub = rospy.Publisher(MODALITY_TO_TOPIC['joint']['head_ol'], JointState, queue_size=1)
        # self.left_arm_pub = rospy.Publisher(MODALITY_TO_TOPIC['joint']['larm_ol'], JointState, queue_size=1)
        self.left_arm_pub = rospy.Publisher(MODALITY_TO_TOPIC['joint']['larm_cmd'], JointState, queue_size=1)
        # self.right_arm_pub = rospy.Publisher(MODALITY_TO_TOPIC['joint']['rarm_ol'], JointState, queue_size=1)
        self.right_arm_pub = rospy.Publisher(MODALITY_TO_TOPIC['joint']['rarm_cmd'], JointState, queue_size=1)

        rospy.Subscriber(MODALITY_TO_TOPIC['joint']['head'], JointState, self.head_callback)
        rospy.Subscriber(MODALITY_TO_TOPIC['joint']['larm'], JointState, self.left_arm_state_callback)
        rospy.Subscriber(MODALITY_TO_TOPIC['joint']['larm_cmd'], JointState, self.left_arm_cmd_state_callback)
        rospy.Subscriber(MODALITY_TO_TOPIC['joint']['rarm'], JointState, self.right_arm_state_callback)
        rospy.Subscriber(MODALITY_TO_TOPIC['joint']['rarm_cmd'], JointState, self.right_arm_cmd_state_callback)
        # rospy.Subscriber(MODALITY_TO_TOPIC['image']['front'], CompressedImage, self.front_img_callback)
        rospy.Subscriber(MODALITY_TO_TOPIC['image']['stereo'], CompressedImage, self.stereo_img_callback)
        rospy.Subscriber(MODALITY_TO_TOPIC['image']['lhand'], CompressedImage, self.left_hand_img_callback)
        rospy.Subscriber(MODALITY_TO_TOPIC['image']['rhand'], CompressedImage, self.right_hand_img_callback)
        # rospy.Subscriber(MODALITY_TO_TOPIC['pose']['obj'], PoseStamped, self.obj_pos_callback)
        # rospy.Subscriber(MODALITY_TO_TOPIC['pose']['larm'], PoseStamped, self.pose_larm_callback)

        print("Waiting for message...")
        for topic in TOPICS['image']:
            rospy.wait_for_message(MODALITY_TO_TOPIC['image'][topic], CompressedImage, timeout=None)
    
    def decode_img(self, msg):
        img_arr = np.frombuffer(msg.data, np.uint8)
        np_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return np_img
    
    def crop_and_resize(self, np_img, name):
        param = CROP_PARAMS[name]
        up, left, height, width = param['up'], param['left'], param['height'], param['width']
        if not(None in [up, left, height, width]):
            np_img = np_img[up:up+height, left:left+width]

        return cv2.resize(np_img, (CAMERA_WIDTH, CAMERA_HEIGHT))

    # def front_img_callback(self, msg):
    #     np_img = self.decode_img(msg)
    #     self.img_front = np_img.astype(np.float32)

    def stereo_img_callback(self, msg):
        np_img = self.decode_img(msg)
        self.img_head_right = self.crop_and_resize(np_img, 'rhead').astype(np.float32)

    def left_hand_img_callback(self, msg):
        np_img = self.decode_img(msg)
        self.img_arm_left = self.crop_and_resize(np_img, 'lhand').astype(np.float32)
        
    def right_hand_img_callback(self, msg):
        np_img = self.decode_img(msg)
        self.img_arm_left = self.crop_and_resize(np_img, 'rhand').astype(np.float32)
        
    def head_callback(self, msg):
        if len(self.head_msg.name) == 0:
            self.head_msg.name = msg.name
            self.head_msg.position = msg.position
        self.head_state = np.array(msg.position)
        
    def left_arm_state_callback(self, msg):
        if len(self.left_arm_msg.name) == 0:
            self.left_arm_msg.name = msg.name
            self.left_arm_msg.position = msg.position
        self.left_arm_state = np.array(msg.position)

    def left_arm_cmd_state_callback(self, msg):
        if len(self.left_arm_msg.name) == 0:
            self.left_arm_msg.name = msg.name
            self.left_arm_msg.position = msg.position
        self.left_arm_cmd_state = np.array(msg.position)
    
    def right_arm_state_callback(self, msg):
        if len(self.right_arm_msg.name) == 0:
            self.right_arm_msg.name = msg.name
            self.right_arm_msg.position = msg.position
        self.right_arm_state = np.array(msg.position)

    def right_arm_cmd_state_callback(self, msg):
        if len(self.right_arm_msg.name) == 0:
            self.right_arm_msg.name = msg.name
            self.right_arm_msg.position = msg.position
        self.right_arm_cmd_state = np.array(msg.position)

    # def obj_pos_callback(self, msg):
    #     self.obj_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    # def pose_larm_callback(self, msg):
    #     self.pose_larm = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
    def custom_bridge(self, msg):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        height, width = msg.height, msg.width
        
        # 画像のデコード（msg.encoding に応じて変換方法を変更する）
        if msg.encoding == "rgb8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV は BGR 形式
            return img
        elif msg.encoding == "bgr8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        elif msg.encoding == "mono8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return img
        elif msg.encoding == "bgra8":
            img = np_arr.reshape((height, width, 4))  # BGRA形式
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # αチャンネルを削除してBGRに変換
            return img
        else:
            rospy.logerr(f"Unsupported encoding: {msg.encoding}")
            return
