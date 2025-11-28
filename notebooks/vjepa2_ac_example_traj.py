import os
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from app.vjepa_droid.transforms import make_transforms
from utils.mpc_utils import (
    compute_new_pose,
    poses_to_diff
)

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def forward_target(c, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h

def forward_actions(z, nsamples, grid_size=0.075, normalize_reps=True, action_repeat=1):

    def make_action_grid(grid_size=grid_size):
        action_samples = []
        for da in np.linspace(-grid_size, grid_size, nsamples):
            for db in np.linspace(-grid_size, grid_size, nsamples):
                for dc in np.linspace(-grid_size, grid_size, nsamples):
                    action_samples += [torch.tensor([da, db, dc, 0, 0, 0, 0], device=z.device, dtype=z.dtype)]
        
        # [125, 1, 7] 125 randomly sampled actions of 7.5, 3,7, and 0 cms in x, y, z directions
        return torch.stack(action_samples, dim=0).unsqueeze(1) 

    def step_predictor(_z, _a, _s):
        # predict new representation and new state given current rep, action, and state
        # [125, 256, 1408]
        _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        # compute new state given current state and sampled random action
        # [125, 1, 7]
        _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
        return _z, _s

    # Sample grid of actions
    action_samples = make_action_grid()
    print(f"Sampled grid of actions; num actions = {len(action_samples)}")

    # Context frame rep and context pose
    # first obs_z repeated [1, 512, 1408] -> [1, 256, 1408] -> [S, N, D] i.e., [125, 256, 1408]
    z_hat = z[:, :tokens_per_frame].repeat(int(nsamples**3), 1, 1)  
    # first state repeated  [S, 1, 7] i.e., [125, 1, 7]
    s_hat = states[:, :1].repeat((int(nsamples**3), 1, 1))  
    # random sampled actions [S, 1, 7] i.e., [125, 1, 7]
    a_hat = action_samples 

    for _ in range(action_repeat):
        # predicted new_z and computed new_s: [125, 256, 1408], [125, 1, 7]
        _z, _s = step_predictor(z_hat, a_hat, s_hat)
        z_hat = torch.cat([z_hat, _z], dim=1)
        s_hat = torch.cat([s_hat, _s], dim=1)
        a_hat = torch.cat([a_hat, action_samples], dim=1)

    print(f"a_hat shape after repeat:", z_hat.shape, s_hat.shape, a_hat.shape)
    return z_hat, s_hat, a_hat # [125, 512, 1408], [125, 2, 7], [125, 2, 7]

def loss_fn(z, h):
    z, h = z[:, -tokens_per_frame:], h[:, -tokens_per_frame:]
    loss = torch.abs(z - h)  # [B, Patches, D] i.e., [125, 256, 1408]
    loss = torch.mean(loss, dim=[1, 2]) # B, 1
    return loss.tolist() # [n] i.e., 125

if __name__ == "__main__":
    
    # Load robot trajectory
    play_in_reverse = False  # load trajectory backwards to see how the energy landscape changes
    trajectory = np.load("franka_example_traj.npz")
    np_clips = trajectory["observations"] # 2 images: [1, 2, 256, 256, 3]
    np_states = trajectory["states"] # 2 robot end-effector poses: [1, 2, 7]

    if play_in_reverse:
        np_clips = trajectory["observations"][:, ::-1].copy()
        np_states = trajectory["states"][:, ::-1].copy()

    np_actions = np.expand_dims(poses_to_diff(np_states[0, 0], 
                                            np_states[0, 1]), 
                                            axis=(0, 1)) # action = state2 - state1

    # Visualize loaded video frames from traj
    T = len(np_clips[0]) # 2
    plt.figure(figsize=(20, 3))
    _ = plt.imshow(np.transpose(np_clips[0], (1, 0, 2, 3)).reshape(256, 256 * T, 3))
    # plt.savefig("loaded_trajectory_frames.png")


    # Initialize VJEPA 2-AC model
    encoder, predictor = torch.hub.load("../", # root of the source code 
                                        "vjepa2_ac_vit_giant", 
                                        source="local",
                                        pretrained=True) 
    # encoder -> VisionTransformer
    # predictor -> VisionTransformerPredictorAC

    # Initialize transform -> app.vjepa_droid.transforms
    # random-resize-crop augmentations
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

    # Convert to torch tensors
    clips = transform(np_clips[0]).unsqueeze(0) # [1, 3, 2, 256, 256]
    states = torch.tensor(np_states) # [1, 2, 7]
    actions = torch.tensor(np_actions) # [1, 1, 7]
    print(f"clips: {clips.shape}; states: {states.shape}; actions: {actions.shape}")

    # Compute energy for cartesian action grid of size (nsample x nsamples x nsamples)
    nsamples = 5
    grid_size = 0.075
    with torch.no_grad():
        # pass through frozen encoder: [1, 3, 2, 256, 256] -> [1, 2*16*16, D] i.e, [1, 512, 1408]
        h = forward_target(clips) 
        # new state prediction from predictor
        z_hat, s_hat, a_hat = forward_actions(h, nsamples=nsamples, grid_size=grid_size)
        # jepa_ac transformer predictor loss
        loss = loss_fn(z_hat, h)

    # Plot the energy
    plot_data = []
    for b, v in enumerate(loss):
        plot_data.append((
            a_hat[b, :-1, 0].sum(),
            a_hat[b, :-1, 1].sum(),
            a_hat[b, :-1, 2].sum(),
            v,
        ))

    delta_x = [d[0] for d in plot_data]
    delta_y = [d[1] for d in plot_data]
    delta_z = [d[2] for d in plot_data]
    energy = [d[3] for d in plot_data]

    gt_x = actions[0, 0, 0]
    gt_y = actions[0, 0, 1]
    gt_z = actions[0, 0, 2]

    # Create the 2D histogram
    heatmap, xedges, yedges = np.histogram2d(delta_x, delta_z, weights=energy, bins=nsamples)

    # Set axis labels
    plt.xlabel("Action Delta x")
    plt.ylabel("Action Delta z")
    plt.title(f"Energy Landscape")

    # Display the heatmap
    print(f"Ground truth action (x,y,z) = ({gt_x:.2f},{gt_y:.2f},{gt_z:.2f})")
    _ = plt.imshow(heatmap.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="viridis")
    _ = plt.colorbar()
    # plt.savefig("energy_landscape_vjepa2_ac.png")
    # plt.close()

    # Compute the optimal action using MPC
    from utils.world_model_wrapper import WorldModel

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        # Doing very few CEM iterations with very few samples just to run efficiently on CPU...
        # ... increase cem_steps and samples for more accurate optimization of energy landscape
        mpc_args={
            "rollout": 2, # ROLL-OUT HORIZON
            "samples": 25,
            "topk": 10,
            "cem_steps": 5,
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.075,
            "verbose": True
        },
        normalize_reps=True,
        device="cpu"
    )

    with torch.no_grad():
        # Pre-trained VJEPA 2 ENCODER representation of current frame and goal frame
        h = forward_target(clips) # [1, 512, 1408]
        z_n, z_goal = h[:, :tokens_per_frame], h[:, -tokens_per_frame:] # [1, 256, 1408], [1, 256, 1408]

        # current observed state
        s_n = states[:, :1] # [1, 1, 7]
        print(f"Starting planning using Cross-Entropy Method...")

        # Action conditioned predictor and zero-shot action inference with CEM
        actions = world_model.infer_next_action(z_n, s_n, z_goal).cpu().numpy()
    
    print(f"Actions returned by planning with CEM: {actions}")
    # print(f"Actions returned by planning with CEM (x,y,z) = ({actions[0, 0]:.2f},{actions[0, 1]:.2f} {actions[0, 2]:.2f})")
