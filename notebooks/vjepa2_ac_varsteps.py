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
from utils.datasets_utils import load_parquet_episode, load_droid_episode

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# make tensor printing human-readable (no scientific notation, 4 decimals)
torch.set_printoptions(precision=4, sci_mode=False)


if __name__ == "__main__":

    # load trajectory backwards to see how the energy landscape changes
    play_in_reverse = False  

    # Load UTN parquet episode
    np_clips, np_states, np_actions = load_parquet_episode()
    H = 256
    W = 256

    # # Load DROID dataset
    # np_clips, np_states, np_actions = load_droid_episode()
    # H = 180
    # W = 320

    if play_in_reverse:
        np_clips = np_clips[:, ::-1].copy()
        np_states = np_states[:, ::-1].copy()
        np_actions = np_actions[:, ::-1].copy()

    np_clips = np_clips[np.newaxis, ...]
    np_states = np_states[np.newaxis, ...]
    np_actions = np_actions[np.newaxis, ...]

    print(f"np_clips shape: {np_clips.shape}, np_states shape: {np_states.shape}, np_actions shape: {np_actions.shape}")
    
    # # randomly sample a state from last 1/4 of the trajectory
    # random_index = np.random.randint(
    #                         max(0, len(np_clips[0]) - len(np_clips[0]) // 4), 
    #                         len(np_clips[0]))
    # print(f"Randomly sampled index: {random_index}")

    # for reproducibility
    goal = 24
    current = goal - 4
    print(f"Goal index: {goal}")
    print(f"Current index: {current}")
        
    # Visualize start and goal video frames from traj
    plt.figure(figsize=(20, 3))
    _ = plt.imshow(
        np.transpose(np_clips[0, [current, goal], :, :, :], 
        (1, 0, 2, 3)).reshape(H, W * 2, 3))
    plt.savefig("start_goal_frames.png")
    plt.close() 

    # VJEPA 2-AC model initialization
    encoder, predictor = torch.hub.load("../", # root of the source code 
                                        "vjepa2_ac_vit_giant", 
                                        source="local",
                                        pretrained=True) 

    # check if model weights are loaded on cuda
    encoder.to("cuda")
    predictor.to("cuda")                             

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
        # Doing very few CEM iterations with very few samples just to run efficiently on CPU...
        # ... increase cem_steps and samples for more accurate optimization of energy landscape
        mpc_args={
            "rollout": 4, # ROLL-OUT HORIZON
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

    # GOAL OBSERVATION
    goal_image = np_clips[0, goal] # [256, 256, 3]
    

    # CURRENT OBSERVATION AND STATE
    current_img = np_clips[0, current] # [256, 256, 3]
    current_state = np_states[:, current, :] # [1, 7]

    # GROUND TRUTH ACTIONS FROM current TO goal
    gt_actions = torch.tensor(np_actions[0, current:goal, :], dtype=torch.float32) # [1, rollout, 7]

    with torch.no_grad():

        # Pre-trained VJEPA 2 ENCODER representation of current frame and goal frame
        z_goal = world_model.encode(goal_image) # [1, 256, 1408]
        
        z_n = world_model.encode(current_img) # [1, 256, 1408]

        # current observed state and to tensor
        s_n = current_state[np.newaxis, ...] # [1, 1, 7]
        s_n = torch.tensor(s_n, dtype=torch.float32)

        # to device
        s_n = s_n.to(world_model.device)

        # Action conditioned predictor and zero-shot action inference with CEM
        actions = world_model.infer_next_action(z_n, s_n, z_goal)

        print(f"Predicted action: {actions.cpu()}")
        print(f"Ground truth action: {gt_actions}")

        

