#-*- coding:utf-8 -*-

from torch.utils.data.dataset import Dataset
from mani_skill.utils import common, io_utils
from typing import List, Optional
from gymnasium import spaces
from tqdm import tqdm
import numpy as np
import torch
import h5py
import cv2

DATA_STAT = {
    'state_min': [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0, 0.0], 
    'state_max': [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033, 0.03999999910593033], 
    'action_min': [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0], 
    'action_max': [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0], 
    'action_std': [0.2199309915304184, 0.18780815601348877, 0.13044124841690063, 0.30669933557510376, 0.1340624988079071, 0.24968451261520386, 0.9589747190475464, 0.9827960729598999], 
    'action_mean': [-0.00885344110429287, 0.5523102879524231, -0.007564723491668701, -2.0108158588409424, 0.004714342765510082, 2.615924596786499, 0.08461848646402359, -0.19301606714725494]
}

def read_video_frames(video_path:str):
    """
        video_path: mp4 file path
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame[:, :, ::-1])
    cap.release()
    return frames

def normalize_action(joints: np.ndarray):
    action_min = np.asarray(DATA_STAT['action_min'])
    action_max = np.asarray(DATA_STAT['action_max'])
    joints = (joints - action_min) / (action_max - action_min) * 2 - 1
    return joints

def sequence_sample(
    action_datas:np.ndarray, # Shape: (episode_length, 8), Range: (-2, 2)
    obs_datas:List[np.ndarray], # (0, 255)
    obs_horizon:int = 2,
    pred_horizon:int = 16,
) -> List[dict]:  
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    action = action_datas[0]
    episode_length = action_datas.shape[0]
    pad_before = obs_horizon - 1
    pad_after = pred_horizon - obs_horizon
    slices = [
        (start, start + pred_horizon) for start in range(-pad_before, episode_length - pred_horizon + pad_after)
    ]
    sequence_datas = []
    for slice in slices:
        start_idx, end_idx = slice
        obs_seq = obs_datas[max(0, start_idx):start_idx+obs_horizon]
        action_seq = list(action_datas[max(0, start_idx):end_idx])
        
        if start_idx < 0: # pad before the trajectory
            obs = obs_seq[0]
            obs_seq.append(obs)
            
            action = action_seq[0]
            action_seq.append(action)
        if end_idx > episode_length: # pad after the trajectory
            gripper_action = action_seq[-1]
            for _ in range(end_idx - episode_length):
                action_seq.append(gripper_action)
        
        obs_seq = np.asarray(obs_seq) 
        action_seq = np.asarray(action_seq)   

        sequence_datas.append({
            'obs': obs_seq,
            'action': action_seq,
            'timestep': max(0, start_idx+obs_horizon),
        })
    return sequence_datas 

def sequence_slices(
    episode_length:int,
    obs_horizon:int = 2,
    pred_horizon:int = 16,
):
    # |o|o|                             observations: 2
    # | |a|a|a|a|a|a|a|a|               actions executed: 8
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    pad_before = obs_horizon - 1
    pad_after = pred_horizon - obs_horizon
    slices = [
        (start, start + pred_horizon) for start in range(-pad_before, episode_length - pred_horizon + pad_after)
    ]
    return slices

def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

class ModifiedSmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(
            self, 
            env_id,
            data_path, 
            obs_process_fn, 
            obs_space, 
            include_rgb, 
            include_depth, 
            device, 
            num_traj,
            control_mode: str,
            obs_horizon: int = 2,
            pred_horizon: int = 16,
            video_base_path: Optional[str] = None,
        ):
        self.control_mode = control_mode
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.video_base_path = video_base_path
        
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        from diffusion_policy.utils import load_demo_dataset
        
        h5_file = h5py.File(data_path, "r")
        json_path = data_path.replace(".h5", ".json")
        json_data = io_utils.load_json(json_path)
        
        # Train Episodes
        slices = []
        episodes = json_data["episodes"][:num_traj]
        for episode_idx, episode in enumerate(episodes):
            episode_id = episode["episode_id"]
            episode_length = episode["elapsed_steps"]
            
            # Get slices
            traj_slices = sequence_slices(
                episode_length = episode_length,
                obs_horizon = self.obs_horizon,
                pred_horizon = self.pred_horizon,
            )
            for slice in traj_slices:
                data = {
                    'slice':slice,
                    'traj_idx':episode_idx,
                }
                slices.append(data)
        self.slices = slices
        self.h5_file = h5_file
        self.device = device
        self.env_id = env_id


    def __getitem__(self, index):
        slice = self.slices[index]['slice']
        traj_idx = self.slices[index]['traj_idx']
        start, end = slice
        
        # Get Actions
        traj_key = f"traj_{traj_idx}"
        actions = self.h5_file[traj_key]['actions']
        actions = normalize_action(actions)
        L, act_dim = actions.shape
        
        pad_action_arm = torch.zeros(
            (act_dim - 1,), device=self.device
        )
        
        act_seq = actions[max(0, start) : end]
        act_seq = torch.tensor(act_seq).float().to(self.device)
        
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        
        # Get Obs
        if self.video_base_path is not None:
            video_path = f"{self.video_base_path}/{self.env_id}/motionplanning/{traj_idx}.mp4"
        else:
            video_path = f"/data/{self.env_id}/motionplanning/{traj_idx}.mp4"
        frames = read_video_frames(video_path)
        
        obs_traj = {
            "rgb": frames
        }
        obs_seq = {}
        for k, v in obs_traj.items():
            v_tensor = torch.tensor(v[
                max(0, start) : start + self.obs_horizon
            ]).float().permute(0, 3, 1, 2).to(self.device) # start+self.obs_horizon is at least 1
            obs_seq[k] = v_tensor
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)
    
if __name__ == '__main__':
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils import common, gym_utils
    import gymnasium as gym
    
    env_id = "PickCube-v1"
    control_mode = "pd_joint_pos"
    obs_mode = "rgb"
    env_kwargs = dict(
        control_mode=control_mode,
        reward_mode="sparse",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    tmp_env = gym.make(env_id, **env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    
    data_path = "/data/PickCube-v1/motionplanning/trajectory.h5"
    obs_process_fn = lambda x:x
    obs_space = orignal_obs_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    device = 'cuda'
    num_demos = 100
    
    dataset = ModifiedSmallDemoDataset_DiffusionPolicy(
        data_path=data_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=device,
        num_traj=num_demos,
        control_mode=control_mode,
        obs_horizon=2,
        pred_horizon=16
    )
    print("Dataset Size >>>", len(dataset))
    d = dataset[10]
    print(">>>", d['actions'].shape, d['observations']['state'].shape)
    print(">>>", torch.min(d['actions']), torch.max(d['actions']))
    print("options:", include_rgb, include_depth)