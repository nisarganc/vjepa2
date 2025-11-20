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

# LeRobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import pyarrow.dataset as ds
from PIL import Image
import io

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def loss_fn(z, z_bar):

    loss = torch.abs(z[0] - z_bar[0])  # [patches, D] i.e., [256, 1408]
    loss_mean = torch.mean(loss, dim=1, keepdim=True)  # [patches, 1] i.e., [256, 1]
    breakpoint()

    # plot 256-dim loss as 16x16 image
    plt.imshow(loss_mean.squeeze().cpu().numpy().reshape(16, 16))
    plt.colorbar()
    plt.title("Per-token absolute difference loss")
    plt.savefig("per_token_loss_2_step.png")
    plt.close()
    loss = torch.mean(loss, dim=[1, 2])
    return loss.tolist() # [n] i.e., 125


if __name__ == "__main__":

    # load trajectory backwards to see how the energy landscape changes
    play_in_reverse = False  

    # Load LeRobot format dataset
    DATASET_PATH = '/home/abut37yx/RCSToolBox/extracted_dataset/data/chunk-000'
    PARTITIONING = ds.partitioning(
                    schema=pa.schema([pa.field("uuid", pa.binary(36))]), 
                    flavor="filename")

    EPISODE = ds.dataset(DATASET_PATH)
    print(EPISODE.schema)

    df = pd.read_parquet(
            DATASET_PATH,
            engine="pyarrow",
            
            # partitioning=PARTITIONING,
            # columns=["uuid", "reward", 
            #           "observation.frames.wrist.rgb", 
            #           "step"],
            # columns={"uuid": "uuid", 
                        # "reward": "reward", 
                        # "observation.frames.wrist.rgb": "observation.frames.wrist.rgb", 
                        # "step": "step"},
            columns={
                "image": pc.field("image"),
                "state": pc.field("state"),
                "actions": pc.field("actions"),
                "frame_index": pc.field("frame_index"),
                "index": pc.field("index"),
                "task_index": pc.field("task_index"),
            },
            dtype_backend="pyarrow",
        )
    # print df keys
    print(df.columns.values)
    
    # load "df["image"]" into numpy array
    np_clips = np.array([np.array(
                        Image.open(io.BytesIO(df["image"][i]['bytes'])
                        )) for i in range(len(df))])
    np_clips = np_clips[np.newaxis, ...] # (1, N, 256, 256, 3)

    np_states = np.array([np.array(
                        df["state"][i]) for i in range(len(df))])
    np_states = np_states[np.newaxis, ...] # (1, N, 7)

    np_actions = np.array([np.array(
                        df["actions"][i]) for i in range(len(df))])
    np_actions = np_actions[np.newaxis, ...] # (1, N, 7)

    if play_in_reverse:
        np_clips = np_clips[:, ::-1].copy()
        np_states = np_states[:, ::-1].copy()
        np_actions = np_actions[:, ::-1].copy()

    print(f"np_clips shape: {np_clips.shape}, np_states shape: {np_states.shape}, np_actions shape: {np_actions.shape}")
    
    # randomly sample a state from last 1/4 of the trajectory
    random_index = np.random.randint(
                            max(0, len(np_states[0]) - len(np_states[0]) // 4), 
                            len(np_states[0]))
    print(f"Randomly sampled index: {random_index}")
    random_index = 100
        
    # Visualize start and goal video frames from traj
    plt.figure(figsize=(20, 3))
    _ = plt.imshow(
        np.transpose(np_clips[0, [random_index-2,random_index], :, :, :], 
        (1, 0, 2, 3)).reshape(256, 256 * 2, 3))
    plt.savefig("start_goal_frames.png")
    plt.close() 
    exit()

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
            "rollout": 1, # ROLL-OUT HORIZON
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

    # GOAL
    goal_image = np_clips[0, random_index] # [256, 256, 3]

    # CURRENT OBSERVATION AND STATE
    current_img = np_clips[0, random_index-2] # [256, 256, 3]
    current_state = np_states[0, random_index-2] # [7,]

    # convert to tensors
    current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(1) # [1, 1, 7]


    with torch.no_grad():

        # Pre-trained VJEPA 2 ENCODER representation of current frame and goal frame
        z_goal = world_model.encode(goal_image) # [1, 256, 1408]
        
        z_n = world_model.encode(current_img) # [1, 256, 1408]

        # abs diff loss
        loss_fn(z_n, z_goal)

        # current observed state
        s_n = current_state # [1, 1, 7]

        # to device
        s_n = s_n.to(world_model.device)

        # Action conditioned predictor and zero-shot action inference with CEM
        # actions = world_model.infer_next_action(z_n, s_n, z_goal)
