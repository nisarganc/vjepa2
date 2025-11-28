import os
import sys
sys.path.insert(0, "..")

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from app.vjepa_droid.transforms import make_transforms
from utils.mpc_utils import (
compute_new_pose,
poses_to_diff
)

# Compute the optimal action using MPC
from utils.world_model_wrapper import WorldModel
from utils.datasets_utils import \
        load_parquet_episode, \
        load_droid_episode, \
        plot_start_goal_frames
        
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# make tensor printing human-readable (no scientific notation, 4 decimals)
torch.set_printoptions(precision=4, sci_mode=False)

def count_params(model, trainable_only=False):
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


if __name__ == "__main__":

    # Load UTN parquet episode
    np_clips, np_states, np_actions = load_parquet_episode()
    H = 256
    W = 256

    # # Load DROID dataset
    # np_clips, np_states, np_actions = load_droid_episode()
    # H = 180
    # W = 320

    np_clips = np_clips[np.newaxis, ...]
    np_states = np_states[np.newaxis, ...]
    np_actions = np_actions[np.newaxis, ...]

    print(f"np_clips shape: {np_clips.shape}, np_states shape: {np_states.shape}, np_actions shape: {np_actions.shape}")


    # VJEPA 2-AC model initialization
    encoder, predictor = torch.hub.load("../", # root of the source code 
                                        "vjepa2_ac_vit_giant", 
                                        source="local",
                                        pretrained=True) 

    # check if model weights are loaded on cuda
    encoder.to("cuda")
    predictor.to("cuda") 
    # print("encoder params:", count_params(encoder))
    # print("decoder params:", count_params(predictor))


    # for reproducibility
    rollout_horizon = 2  
        
    # CURRENT OBSERVATION
    current = 2
    current_image = np_clips[0, current] # [256, 256, 3]

    # CURRENT STATE
    current_state = np_states[:, current, :] # [1, 7]
    s_n = current_state[np.newaxis, ...] # [1, 1, 7]
    s_n = torch.tensor(s_n, dtype=torch.float32) 

    # GROUND TRUTH ACTION
    intermediate_ = current + rollout_horizon
    gt_action = torch.tensor(np_actions[0, current:intermediate_, :], dtype=torch.float32) # [rollout_horizon, 7]

    # GROUND TRUTH NEXT STATE
    ground_truth_states = torch.tensor(np_states[0, current:intermediate_+1, :], dtype=torch.float32) # [rollout_horizon, 7]

    # Subgoals from 8, 7, 6, 5, 4, 3, 2
    for k in range(8, 1, -1):

        goal = current + k
        print(f"Goal distance: {k}")

        # Initialize transform (random-resize-crop augmentations)
        crop_size = 256
        tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
        transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.),
            random_resize_scale=(1., 1.),
            reprob=0.,
            auto_augment=False,
            motion_shift=False,
            crop_size=crop_size,
        )

        # World model wrapper initialization
        world_model = WorldModel(
            encoder=encoder,
            predictor=predictor,
            tokens_per_frame=tokens_per_frame,
            transform=transform,
            mpc_args={
                "rollout": rollout_horizon, 
                "samples": 25,
                "topk": 10,
                "cem_steps": 1,
                "momentum_mean": 0.15,
                "momentum_mean_gripper": 0.15,
                "momentum_std": 0.75,
                "momentum_std_gripper": 0.15,
                "maxnorm": 0.075,
                "verbose": True
            },
            normalize_reps=True,
            device="cuda"
        )

        # GOAL IMAGE
        goal_image = np_clips[0, goal] # [256, 256, 3]

        
        # Visualize start and goal video frames from traj
        # plot_start_goal_frames(np_clips, current, goal, H, W)

        with torch.no_grad():

            # Pre-trained VJEPA 2 ENCODER representation of current frame 
            z_goal = world_model.encode(goal_image) # [1, 256, 1408]
            
            # of goal frame
            z_n = world_model.encode(current_image) # [1, 256, 1408]

            # Action conditioned predictor and zero-shot action inference with CEM
            actions = world_model.infer_next_action(z_n, s_n.to(world_model.device), z_goal) # [4, 7] 

            # compute predicted next states
            predicted_states = s_n.clone()  # [1, 1, 7]
            for i in range(rollout_horizon): 
                a_n = actions[i].unsqueeze(0).unsqueeze(1)  # [1, 1, 7]
                s_next = compute_new_pose(s_n, a_n)  # [1, 1, 7]
                predicted_states = torch.cat((predicted_states, s_next), dim=1)  # [1, i+2, 7]
                s_n = s_next  

            print(f"Predicted new state: {predicted_states.to("cpu")}") # [1, rollout_horizon+1, 7]
            print(f"Ground truth new state: {ground_truth_states.unsqueeze(0)}") # [1, rollout_horizon+1, 7]
