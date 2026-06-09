import os
import sys
import yaml
import glob
import argparse
import time
from pathlib import Path
import numpy as np
import torch

# ROS
import rospy

# import ipdb;ipdb.set_trace()
from src.robot_controller.rt_core import RTCore
from src.utils import set_seed
from src.policy import ACTPolicy


def normalization(x, bounds, minmax):
    return (x - bounds[0]) / (bounds[1] - bounds[0]) * (minmax[1] - minmax[0]) + minmax[0]

class Deploy(RTCore):
    def __init__(self, args: argparse.Namespace):
        super(Deploy, self).__init__()
        self.args = args

        # control freq and loop length
        freq = args.freq
        self.r = rospy.Rate(freq)
        self.nloop = args.exp_time * args.freq

        # config
        with open(os.path.join(os.path.split(args.ckpt_path)[0], 'config.yaml')) as f:
            self.config = yaml.safe_load(f)

        # gt
        files = sorted(glob.glob(str(Path(args.dataset_dir) / '*.npz')))
        print(files[args.idx])
        self.gt = np.load(files[args.idx])
        self.init_pos = {'left_arm': self.gt['left_arm_command'][0],'right_arm': self.gt['right_arm_command'][0] }

        # device
        self.device = 'cpu' if args.device < 0 else f"cuda:{args.device}"
        self.config['device'] = self.device

        # model
        set_seed(1)
        self.policy = ACTPolicy(self.config)
        loading_status = self.policy.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device(self.device)))
        print(loading_status)
        self.policy.to(self.device)
        self.policy.eval()

        # pre, post process
        stats_path = Path(args.ckpt_path).parent / 'dataset_stats.npz'
        stats = np.load(stats_path)
        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # temporal ensembling logic
        self.query_frequency = self.config['num_queries']
        if self.config['temporal_agg']:
            self.query_frequency = 1
            self.all_time_actions = torch.zeros([self.nloop, self.nloop + self.config['num_queries'], self.config['state_dim']]).to(self.device)

    def initialization(self, exp_time: int = 5, freq: int = 100,) -> None:
        rospy.logwarn("Start initialization")
        nloop = exp_time * freq

        traj = dict()
        for k, v in self.init_pos.items():
            curr_pos = getattr(self, k + '_msg').position
            traj[k] = np.linspace(curr_pos, v, nloop)
        for i in range(1, nloop):
            target_pos = {k: v[i] for k, v in traj.items()}
            self.pub_msg(target_pos)
            time.sleep(1./freq)

        rospy.logwarn("Finished initialization")

    def pub_msg(self, target_pos: dict) -> None:
        for k, v in target_pos.items():
            msg = getattr(self, k + '_msg')
            pub = getattr(self, k + '_pub')
            msg.header.stamp = rospy.Time.now()
            msg.position = v
            pub.publish(msg)

    def get_observation(self, prev_obs=None, loop_ct=None):
        lrarm_np = np.concatenate([self.left_arm_cmd_state, self.right_arm_cmd_state], -1)
        nlrarm_np = self.pre_process(lrarm_np)
        nimgs_list = [] 
        for imgname in ['img_head_right', 'img_arm_left']:
            img = getattr(self, imgname)
            nimg = np.transpose(getattr(self, imgname), (2, 0, 1)) / 255.0
            nimgs_list.append(nimg)
        nimgs_np = np.np.vstack(nimgs_list)

        nlrarm = torch.from_numpy(nlrarm_np).unsqueeze(0)
        nimgs = torch.from_numpy(nimgs_np).unsqueeze(0)

        return nlrarm, nimgs

    @torch.inference_mode()
    def get_action(self, loop_ct):
        
        if self.args.playback:
            return {'left_arm': self.gt['left_arm_command'][loop_ct],'right_arm': self.gt['right_arm_command'][loop_ct] }
        
        # prediction
        if loop_ct % self.query_frequency == 0:
            self.all_actions = self.policy(self.get_observation())  # [1, query_num, state_dim]
            
        # temporal ensembling
        if self.config['temporal_agg']:
            self.all_time_actions[[loop_ct], loop_ct: loop_ct + self.config['num_queries']] = self.all_actions
            actions_for_curr_step = self.all_time_actions[:, loop_ct] # [inferloop, statedim]
            
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            naction = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True) # [1, statedim]
        else:
            naction = self.all_actions[:, loop_ct % self.query_frequency]

        # denormalization
        naction = naction.squeeze(0).cpu().detach().numpy()
        action = self.post_process(naction)
        
        return {'left_arm': action[:9], 'right_arm': action[9:18]}
        

    def run(self):
        input("Playback.run(): Press enter to move to init pos")
        self.initialization()

        input("Are you ready?")
        for loop_ct in range(self.nloop):

            target_pos = self.get_action(loop_ct)

            print(target_pos)
            self.pub_msg(target_pos)
            
            if self.args.playback and loop_ct == len(self.gt['left_arm_command']) - 1:
                break

            # Sleep
            self.r.sleep()

        rospy.logwarn("Playback.run(): Finished execution")

        # self.initialization()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--playback", action='store_true')
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--exp_time", type=int, default=30)
    parser.add_argument("--freq", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    rospy.init_node("deploy_node", anonymous=True)
    task = Deploy(args)
    time.sleep(1)
    task.run()
    sys.exit()
