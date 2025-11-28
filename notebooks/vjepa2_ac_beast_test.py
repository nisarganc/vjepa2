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
    goal = 20
    current = goal - 20
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
            "rollout": goal - current, # ROLL-OUT HORIZON
            "samples": 25,
            "topk": 10,
            "cem_steps": 3,
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
    # Save current_img as a PNG file
    plt.imsave("current_frame.png", current_img)
    current_state = np_states[:, current, :] # [1, 7]

    # GROUND TRUTH ACTIONS AND STATES FOR COMPARISON
    gt_actions = torch.tensor(np_actions[0, current:goal, :], dtype=torch.float32) # [rollout, 7]
    gt_state = torch.tensor(np_states[0, current:goal, :], dtype=torch.float32) # [rollout, 7]

    print("current state:", current_state)

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

        # print(f"Predicted action: {actions.cpu()}")
        # print(f"Ground truth action: {gt_actions}")

        # Compare predicted pose trajectory with ground truth in a 3D plot (just positions)
        predicted_poses = [s_n[0, 0].cpu().numpy()]
        for t in range(actions.shape[0]):
            new_pose = compute_new_pose(
                torch.tensor(predicted_poses[-1]).unsqueeze(0).unsqueeze(0), 
                actions[t:t+1, :].unsqueeze(1)
            )
            predicted_poses.append(new_pose[0, 0].cpu().numpy())
        predicted_poses = np.array(predicted_poses) # [rollout+1, 7]
        gt_poses = gt_state.cpu().numpy() # [rollout, 7]
        gt_poses = np.concatenate([current_state, gt_poses], axis=0)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(predicted_poses[:, 0], predicted_poses[:, 1], predicted_poses[:, 2], label='Predicted Trajectory', marker='o')
        ax.plot(gt_poses[:, 0], gt_poses[:, 1], gt_poses[:, 2], label='Ground Truth Trajectory', marker='x')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title('Predicted vs Ground Truth Trajectory')
        ax.legend()
        plt.savefig(f"predicted_vs_gt_trajectory_{world_model.mpc_args['cem_steps']}_cem_steps.png")
        plt.close(fig)

        print(f"Predicted new pose: {compute_new_pose(s_n, actions[0:1, :].unsqueeze(1))}")
        # print(f"Ground truth new pose: {compute_new_pose(s_n, gt_actions[0:1, :].unsqueeze(1))}")
        print(f"Ground truth next state: {gt_state[1:2, :]}")
