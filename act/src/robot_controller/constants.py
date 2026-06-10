import numpy as np


# topics to be used
TOPICS = {
    'joint': [
        # 'head',
        'larm',
        'larm_cmd',
        'rarm',
        'rarm_cmd',
    ],
    # 'pose': [
    #     'larm',
    #     'obj'
    # ],
    'image': [
        # 'front',
        # 'aruco',
        'stereo',
        'lhand',
        # 'rhand'
    ],
}


CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640
CROP_PARAMS = {
    'lhead': {
        'up': None,
        'left': None,
        'height': None,
        'width': None
    },
    'rhead': {
        'up': 120,
        'left': 120,
        'height': 195,
        'width': 420
    },
    'lhand': {
        'up': None,
        'left': None,
        'height': None,
        'width': None
    },
    'rhand': {
        'up': None,
        'left': None,
        'height': None,
        'width': None
    },
}


# reference for all topics
MODALITY_TO_TOPIC = {
    'joint':{
        'head': "/maharo/head/joint_states",
        'head_ol': "/maharo/head/online_joint_states",
        'larm': "/maharo/left_arm/upperbody/joint_states",
        'larm_ol': "/maharo/left_arm/upperbody/online_joint_states",
        'larm_cmd': "/maharo/left_arm/upperbody/command_states",
        'rarm': "/maharo/right_arm/upperbody/joint_states",
        'rarm_ol': "/maharo/right_arm/upperbody/online_joint_states",
        'rarm_cmd': "/maharo/right_arm/upperbody/command_states",
        'rviz': "/joint_states",
    },
    'pose':{
        'larm': "/left_tip_pose",
        'obj': "/obj_pose",
    },
    'image':{
        # 'front': "/camera/color/image_raw/compressed", # tum
        'front': "/usb_cam/image_raw/compressed", 
        'aruco': "/aruco_image/compressed",
        'stereo': "/zed2i/zed_node/stereo_raw/image_raw_color/compressed",
        'lhand': "/left_hand_camera/image_raw/compressed",
        'rhand': "/right_hand_camera/image_raw/compressed",
    }
}
