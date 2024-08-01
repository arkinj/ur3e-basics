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
from skvideo.io import vwrite
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
import pupil_apriltags as apriltag

# Import from utils
from mit_perception.env_T import PushTEnv
from mit_perception.network import ConditionalUnet1D
from mit_perception.inference_utils import normalize_data, unnormalize_data

import mit_perception.move_utils as move_utils
import mit_perception.perception_utils as perception_utils

device = torch.device('cuda')

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
noise_pred_net.load_state_dict(torch.load('push_T_diffusion_model')) #Add to-do

noise_pred_net = noise_pred_net.cuda()

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
max_steps = 200

#Initialize the PushT environment. 
# The env class has been modified to include (x0,y0), which are the init coord of the T-block
env = PushTEnv(x0=500,y0=100,mass=0.1,friction=1,length=4)

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
move_group_arm, move_group_hand = move_utils.setup()
move_utils.move_to_home_pose(move_group_arm)
move_utils.close_gripper(move_group_hand)

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
file = open('noise.pkl','rb')
data = pickle.load(file)
file.close()
noisy_action_list = data['noise']
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
            noisy_action  = noisy_action_list[i_idx]
            #noisy_action = torch.randn(
            #   (B, pred_horizon, action_dim), device=device)
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

        for i in range(len(action)):
            # stepping env
            #uncomment below to use simulated environment
            #obs, coverage, reward, done, info = env.step(action[i])

            """
            Uncomment below to use real step (currently very untested )
            Execution: action[i] -[X,Y] --> scaled end effector pose in the PoseStamped object -->move_pose() 
            Observation collection: April_Tag1, April_Tag2 (new location)--> obs vector [5X1] , [x_end, y_end, x_ob, y_ob, theta_ob]
            """
            obs, coverage, reward, done, info = env.step_real(action[i], move_group_arm, april_tag, cam)
            
            # save observations
            info = env._get_info()
            shape = info['block_pose']
            
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
  
            print('reward',reward)
            imgs.append(env.render(mode='rgb_array'))

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
from IPython.display import Video
vwrite('vis_1_01_5.gif', imgs)
#Video('vis__1_01_5.mp4', embed=True, width=1024*4, height=1024*4)