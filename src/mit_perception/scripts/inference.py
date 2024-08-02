"""
Python script for running Diffusion Policy inference on UR3E robot. Part of the script has been taken from
the original implementation of the Push-T experiment in Diffusion policy paper: 
https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing
"""
# Import libraries
import numpy as np
import torch
import torch.nn as nn
import collections
import pickle
#from skvideo.io import vwrite
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
import pupil_apriltags as apriltag

# Import from utils
from mit_perception.env_T import PushTEnv
from mit_perception.network import ConditionalUnet1D
from mit_perception.inference_utils import normalize_data, unnormalize_data

import mit_perception.move_utils as arm
from mit_perception.state_estimator_T import TAG_SIZES
from mit_perception.apriltag_utils import RealsenseCam, AprilTag

import gdown

device = torch.device('cpu')

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 16
action_dim = 2
obs_dim = 5
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
optimizer = torch.optim.AdamW(params=noise_pred_net.parameters(),lr=1e-4, weight_decay=1e-6)


ckpt_path = "test.ckpt"
#id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
#gdown.download(id=id, output=ckpt_path, quiet=False)

state_dict = torch.load(ckpt_path, map_location='cpu')
noise_pred_net = noise_pred_net
noise_pred_net.load_state_dict(state_dict)
print('Pretrained weights loaded.')

# example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))
print('noised_action',noised_action)
print('noised_action_shape',noised_action.shape)

# the noise prediction network
# takes noisy action, diffusion iteration and observation as input
# predicts the noise added to action
noise = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    global_cond=obs.flatten(start_dim=1))

#checkpoint = torch.load('train_cpu.pt')
#noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# This command is not working due to the model being trained on GPU and the version of torch being CPU. Fix it
#noise_pred_net.load_state_dict(torch.load('push_T_diffusion_model'),map_location=torch.device('cpu')) #Add to-do


#noise_pred_net = noise_pred_net.cuda()

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# limit enviornment interaction to 200 steps before termination
max_steps = 100

#Initialize the PushT environment. 
# The env class has been modified to include (x0,y0), which are the init coord of the T-block
env = PushTEnv(x0=256,y0=256,mass=0.1,friction=1,length=4)

# get first observation
obs = env.reset()

# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)

# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0

# setup real arm

#Have flags for what it should do
move_group_arm, move_group_hand = arm.setup()
arm.move_to_home_pose(move_group_arm)
pose = move_group_arm.get_current_pose()
pose.pose.position.z = 0.265
arm.move_to_pose(move_group_arm, pose.pose)
#arm.open_gripper(move_group_hand)
#arm.close_gripper(move_group_hand)

# Initialize the camera and AprilTag detector
cam = RealsenseCam(
    "317422075665", # cam serial?
    (1280, 720), # color img size
    (1280, 720), # depth img size
    30 # frame rate
)
# tag size in meters, or dict of tag_size: tag_ids
april_tag = AprilTag(tag_size=TAG_SIZES) 

#Manually encode the stats for min and max values of the obs and action
obs_max = np.array([496.14618  , 510.9579   , 439.9153   , 485.6641   ,   6.2830877])
obs_min= np.array([1.3456424e+01, 3.2938293e+01, 5.7471767e+01, 1.0827995e+02, 2.1559125e-04])
action_min= np.array([12.,25.])
action_max = np.array([511.,511.])
stats_obs = {'max':obs_max,'min':obs_min}
stats_action = {'max':action_max,'min':action_min}

#Hard-coded sequence of Gaussian noise for de-noising
#file = open('noise.pkl','rb')
#data = pickle.load(file)
#file.close()
#noisy_action_list = data['noise']
i_idx=0

seed=1000
torch.manual_seed(seed=seed)
with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats_obs) 
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            #noisy_action  = noisy_action_list[i_idx]
            noisy_action = torch.randn(
               (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # initialize scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
           
                
                if rewards==[]:
                    reward = torch.tensor(0)
                else:
                    reward = rewards.pop()
                
        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats_action)

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps without replanning
        N_idx=1
        for i in range(0,len(action),N_idx):
            # stepping env
            #uncomment below to use simulated environment
            #obs, coverage, reward, done, info = env.step(action[i])

            """
            Uncomment below to use real step (currently very untested )
            Execution: action[i] -[X,Y] --> scaled end effector pose in the PoseStamped object -->move_pose() 
            Observation collection: April_Tag1, April_Tag2 (new location)--> obs vector [5X1] , [x_end, y_end, x_ob, y_ob, theta_ob]
            """
            obs, coverage, reward, done, info = env.step_real(action[i:i+N_idx].reshape((1,2)), move_group_arm, april_tag, cam)
            
            # save observations
            info = env._get_info()
            shape = info['block_pose']
            
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
  
            print('reward',reward)
            #imgs.append(env.render(mode='rgb_array'))
            print(step_idx)
            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break
        i_idx+=1
# print out the maximum target coverage
print('Score: ', max(rewards))

# visualize
#from IPython.display import Video
#vwrite('vis_1_01_5.gif', imgs)
#Video('vis__1_01_5.mp4', embed=True, width=1024*4, height=1024*4)