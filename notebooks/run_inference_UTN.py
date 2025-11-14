import os
import sys
sys.path.insert(0, "..")

import cv2  
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# robot imports
from net_franky import setup_net_franky
setup_net_franky("multihead.utn-ai.de", 18813) #localhost
from simple_move.robot.hw_bot.fr3_hw import Fr3Hw
# from simple_move.robot.twin_bot.fr3_sim import FR3Twin # simulation
from simple_move_fixtures.grippers import FinrayGripper, TactoGripper

# VJEPA imports
from app.vjepa_droid.transforms import make_transforms
from utils.world_model_wrapper import WorldModel

# MPC utils
from utils.mpc_utils import (
    compute_new_pose,
    poses_to_diff
)
# suppress warnings
import warnings
warnings.filterwarnings("ignore")



# INITIALIZATIONS
# read current observation at 4fps with resolution 256 * 256
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FPS, 4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

# robot initialization
gripper = FinrayGripper()
robot = Fr3Hw(gripper=gripper, 
              ip="192.168.100.1", 
              open_viewer=False) # Use your robot's IP
# robot = FR3Twin() # simulation

# reset robot to a position
# pose = np.array([
#     [1, 0, 0, 0.3],
#     [0, -1, 0, 0.0],
#     [0, 0, -1, 0.5],
#     [0, 0, 0, 1],
# ])
# robot.move(pose, cartesian=False)
# exit()

# # gen goal observation
# _, frame = cap.read()
# frame = cv2.resize(frame, (256, 256))
# cv2.imwrite("./UTN_GoalImages/goal_observation_air_kinect.png", frame)
# exit()

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
        "rollout": 2, # ROLL-OUT HORIZON
        "samples": 25,
        "topk": 10,
        "cem_steps": 2,
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

# read goal image from "./UTN_GoalImages/goal_observation.png" and pass it to world model
goal_img = cv2.imread("./UTN_GoalImages/goal_observation_air_kinect.png")
goal_img = cv2.cvtColor(goal_img, cv2.COLOR_BGR2RGB)
z_goal = world_model.encode(goal_img) # [1, 256, 1408]

i = 0
# INFERENCE LOOP
while i < 10:    
    # read current observation
    _, frame = cap.read()
    frame = cv2.resize(frame, (256, 256))

    current_obs = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get current state from the robot
    state = robot.get_cartesian_state()
    state = np.hstack((state[0:3, 3], # position
                    R.from_matrix(state[0:3, 0:3]).as_euler('xyz'), # orientation
                    robot.gripper.width # gripper width
                    )).reshape(1, 7)
    # convert to tensors
    current_state = torch.tensor(state, dtype=torch.float32).unsqueeze(1) # [1, 1, 7]

    # run inference to get next action
    with torch.no_grad():
        # Pre-trained VJEPA 2 ENCODER representation of current frame and goal frame
        z_n = world_model.encode(current_obs) # [1, 256, 1408]

        # current observed state
        s_n = current_state # [1, 1, 7]

        # to device
        s_n = s_n.to(world_model.device)

        # Action conditioned predictor and zero-shot action inference with CEM
        actions = world_model.infer_next_action(z_n, s_n, z_goal)

    # EXECUTE THE ACTIONS RETURNED BY CEM
    # convert to state changes
    next_pose = current_state
    for i in range(actions.shape[0]):
        action = actions[i].unsqueeze(0).unsqueeze(0) # 1 x 1 x 7
        next_pose = compute_new_pose(next_pose[:, -1:], action[:, -1:])

    next_pose = next_pose[0].cpu().numpy() # 1 x 7
    pose_without_gripper = next_pose[0, :-1]  # 6,
    gripper_width = next_pose[0, -1]  # 1,

    # convert to pose matrix
    rotation = R.from_euler('xyz', pose_without_gripper[3:]).as_matrix()
    trans_rot = np.hstack((rotation, 
                    np.array(pose_without_gripper[:3]).reshape(3, 1)))
    pose = np.vstack((trans_rot, 
                    np.array([0, 0, 0, 1])))

    # execute action
    robot.move(pose, cartesian=False)  # can switch to cartesian=True 
    robot.set_gripper(0.08 * gripper_width)  # TODO: set gripper width
    i += 1
