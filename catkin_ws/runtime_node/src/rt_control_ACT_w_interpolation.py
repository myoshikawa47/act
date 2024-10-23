#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import yaml
import torch

# ROS
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState, CompressedImage
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState, Joy

from hand_controller import HandClient
from robot import Torobo
from move_to_initial_pose import MoveToInitPose


# added for ACT
from imitate_episodes import make_policy
from utils import compute_dict_mean, set_seed, detach_dict # helper functions

import IPython
e = IPython.embed


class RTControl(Torobo):
    def __init__(self, ckpt_path, control_freq, inference_freq, device):
        movegroup="right_arm"
        tooltip="right_arm/link_tip"
        super().__init__(movegroup=movegroup, tooltip=tooltip)
        
        # parameter
        with open(os.path.join(os.path.split(ckpt_path)[0], 'config.yaml')) as f:
            self.config = yaml.safe_load(f)
        self.config['device'] = device
        self.device = device
        self.current_image = None
        self.current_left_arm = None
        self.current_right_arm = None

        # Joint states for subtask 1
        self.head_t0 = np.array([0.0, 0.56, -0.024])  
        self.left_arm_t0 = np.array([ 0.192, -0.699, 0.513, 2.138, 0.609, 0.048, -0.173], dtype=np.float32)
        self.right_arm_t0 = np.array([0.75296, -0.8325, -0.23391, 1.90076, -0.6311, 0.65651, 0.10761], dtype=np.float32) # data on 0702
        self.torso_t0 = np.array([-0.17453, 0.52345, 0.000], dtype=np.float32)
                
        # initialize ros node
        self.inference_freq = inference_freq
        self.control_freq = control_freq
        self.exptime = 45 / int(inference_freq/ 10)
        self.r = rospy.Rate(control_freq)
        
        # initialization
        self.move_init_pose = MoveToInitPose(freq=self.control_freq)
        
        # load model
        set_seed(1)
        self.policy = make_policy(self.config['policy_class'], self.config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
        print(loading_status)
        self.policy.to(device)
        self.policy.eval()

        # load stats
        stats_path = os.path.join(os.path.split(ckpt_path)[0], 'dataset_stats.npz')
        stats = np.load(stats_path)

        # pre, post process
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        self.query_frequency = self.config['num_queries']
        if self.config['temporal_agg']:
            self.query_frequency = 1
        
        self._rpy = np.load("/home/ogata/work/Yoshikawa/rosbag/0704/data/test/hand_pose.npy")[0,0,3:6]
        
        # publisher
        self.bridge = CvBridge()
        self.cmd_pub = rospy.Publisher("/torobo/online_joint_trajectory_controller/command", JointTrajectory, queue_size=1)    

        # subscriber
        rospy.Subscriber('/torobo/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/torobo/head/see3cam_left/camera/color/image_raw/calibration/compressed', CompressedImage, self.cam_left_calib_callback)
        print("wait for message: /torobo/joint_states")
        rospy.wait_for_message('/torobo/joint_states', JointState)
        print("wait for message: /torobo/head/see3cam_left/camera/color/image_raw/calibration/compressed")
        rospy.wait_for_message('/torobo/head/see3cam_left/camera/color/image_raw/calibration/compressed', CompressedImage)
                                
        # hand
        self.right_hand = HandClient("/torobo", "right")

        # setup the trajectory message
        self.joint_msg = JointState()
        self.joint_msg.name = [
            "right/joint_1",
            "right/joint_2",
            "right/joint_3",
            "right/joint_4",
            "right/joint_5",
            "right/joint_6",
            "right/joint_7",
        ]

        # setup the hand message
        # 0 is open, 1 is close
        self.hand_states = [0]
        
    
    def joint_callback(self, msg):
        self.current_head = np.array(msg.position[7:10])
        self.current_left_arm = np.array(msg.position[10:17])
        self.current_right_arm = np.array(msg.position[27:34])
        self.current_torso = np.array(msg.position[-3:])
    

    def cam_left_calib_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        np_img = np_img[200:650,680:] # 600 * 450
        # np_img = cv2.resize(np_img,(60, 45))
        # import ipdb;ipdb.set_trace()
        self.current_image = np_img
    

    def initialization(self):
        self.generate_trajectory([self.current_right_arm], [self.right_arm_t0], 5)
        
        # hand initialization for subtask 1
        self.right_hand.openPinch()
        
        rospy.logwarn("Finish initializing")


    def generate_trajectory(self, start_position, goal_position, time=5):
        nloop = self.control_freq * time
        right_arm_trajectory = np.linspace(start_position[0], goal_position[0], nloop)

        for loop_ct in range(nloop):
            _position = right_arm_trajectory[loop_ct]
            self.joint_msg.position = _position
            self.joint_pub.publish(self.joint_msg)
            self.r.sleep()
            
        rospy.sleep(0.5)  


    def hand_sleep(self):
        if self.inference_freq == 10:
            # rospy.sleep(0.1)
            rospy.sleep(0.2) # for D
        elif self.inference_freq == 20:
            rospy.sleep(0.5)
        elif self.inference_freq == 30:
            rospy.sleep(0.5)
        elif self.inference_freq == 50:
            rospy.sleep(0.7)
        elif self.inference_freq == 70:
            rospy.sleep(0.8)
        else:
            rospy.sleep(0.8)
            print("freq is ", self.inference_freq)
            exit()            
    

    def hand_open(self):
        self.right_hand.openPinch()
        self.hand_sleep()
        

    def hand_close(self):
        self.right_hand.closePinch()
        self.hand_sleep()


    def predict_10Hz(self, infer_id):
        if infer_id % self.query_frequency == 0:
            img_t = self.current_image.transpose(2, 0, 1)
            img_t = torch.from_numpy(img_t / 255.0).float().unsqueeze(0).unsqueeze(0).to(self.device) # (batch(=1), camera_num(=1), channel, height, width)
            
            _pose_t = self.compute_fk(self.current_right_arm)           
            _pose_t = np.concatenate( (list(_pose_t)[:3], self._hand_state), axis=-1)
            pose_t = self.pre_process(_pose_t)
            pose_t = torch.Tensor(np.expand_dims(pose_t, axis=0)).to(self.device)
            
            # prediction
            self.all_actions = self.policy(pose_t, img_t)  # [1, query_num, state_dim]
            
        if self.args['temporal_agg']:
            self.all_time_actions[[infer_id], infer_id: infer_id + self.config['num_queries']] = self.all_actions
            actions_for_curr_step = self.all_time_actions[:, infer_id] # [inferloop, statedim]
            
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True) # [1, statedim]
        else:
            raw_action = self.all_actions[:, infer_id % self.query_frequency]

        # denormalization
        raw_action = raw_action.squeeze(0).cpu().detach().numpy()
        action = self.post_process(raw_action)
        pred_pose = action

        _position = pred_pose[:-1]
        self._hand_state = np.array([pred_pose[-1]])

        if self.hand_states[0] == 0 and self._hand_state[0] > 0.8:
            print("hand_close")
            sleep_time = 1. / self.control_freq - (time.perf_counter() - self.prev_time)
            print(f'sleep time: {sleep_time}')
            rospy.sleep(max(sleep_time, 0.))            

            self.hand_close()
            self.hand_states[0] = 1

        if self.hand_states[0] == 1 and self._hand_state[0] < 0.2:
            print("hand_open")            
            sleep_time = 1. / self.control_freq - (time.perf_counter() - self.prev_time)
            print(f'sleep time: {sleep_time}')
            rospy.sleep(max(sleep_time, 0.))
            
            self.hand_open()
            self.hand_states[0] = 0
        
        # calc ik
        res = self.compute_ik(x=_position[0], y=_position[1], z=_position[2],
                            roll=self._rpy[0], pitch=self._rpy[1], yaw=self._rpy[2],
                            joint_angles=self.current_right_arm)
        right_arm_cmd = np.array(res.position[26:33])

        return right_arm_cmd

        
    def main(self):
        nloop = int(self.control_freq * self.exptime)
        self.inferloop = int(self.inference_freq * self.exptime)
        self._hand_state = np.array([0.0])
        target_joint = np.array([])
        self.prev_time = None

        if self.config['temporal_agg']:
            self.all_time_actions = torch.zeros([self.inferloop, self.inferloop + self.config['num_queries'], self.config['state_dim']]).to(self.device)
            self.action_chunk = torch.zeros([self.inferloop, self.config['num_queries'], self.config['state_dim']]).to(device)
            
        input("Are you ready??")
        with torch.inference_mode():
            for loop_ct in range(nloop):
                if loop_ct % int(self.control_freq / self.inference_freq) == 0:
                    infer_id = loop_ct // int(self.control_freq / self.inference_freq)
                    _target_joint = self.predict_10Hz(infer_id)
                    n_sample = int(self.control_freq/self.inference_freq) + 1
                    target_joint = np.linspace(self.current_right_arm, _target_joint, n_sample)[1:]

                if len(target_joint) > 0:
                    cmd = target_joint[0]
                    target_joint = np.delete(target_joint, 0, 0)

                    point = JointTrajectoryPoint()
                    point.time_from_start = rospy.Time(0)
                    trajectory = JointTrajectory()
                    trajectory.points.append(point)
                    trajectory.header.stamp = rospy.Time.now()
                    trajectory.points[0].positions = cmd.tolist()
                    trajectory.joint_names = [ "right_arm/joint_1",
                                                "right_arm/joint_2",
                                                "right_arm/joint_3",
                                                "right_arm/joint_4",
                                                "right_arm/joint_5",
                                                "right_arm/joint_6",
                                                "right_arm/joint_7"]
                    
                    if self.prev_time is not None:
                        sleep_time = 1. / self.control_freq - (time.perf_counter() - self.prev_time)
                        print(f'loop {loop_ct - 1} sleep time: {sleep_time}')
                        rospy.sleep(max(sleep_time, 0.))
                    self.cmd_pub.publish(trajectory)
                    self.prev_time = time.perf_counter()

        print("Finished !!!")
        print("Return to Initial Pose")
        self.move_init_pose.main(3)
        

        
if __name__ == "__main__":    
    rospy.init_node('rt_control', anonymous=True, disable_signals=True)

    ckpt_path = rospy.get_param("~ckpt_path")
    control_freq = rospy.get_param('~control_freq')
    inference_freq = rospy.get_param('~inference_freq')
    device = rospy.get_param('~device')
    if device >= 0:
        device = f'cuda:{device}'
    else:
        device = 'cpu'

    t = RTControl(ckpt_path, control_freq, inference_freq, device)
    t.main()