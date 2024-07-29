#@markdown ### **Inference**
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm

# env import
from skvideo.io import vwrite
from mit_perception.env_T import PushTEnv
from mit_perception.network import ConditionalUnet1D
from mit_perception.inference_utils import normalize_data, unnormalize_data
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


# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
noise_pred_net.load_state_dict(torch.load('/home/anjali/push_T_diffusion_model'))

noise_pred_net = noise_pred_net.cuda()


num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# limit enviornment interaction to 200 steps before termination
max_steps = 200
env = PushTEnv(x0=500,y0=100,mass=0.1,friction=1,length=4)
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(x0=100,y0=100)

# get first observation
obs = env.reset()
print(obs)
# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0
alpha = 0.1

#def cost_grad(nmean):
seed=42
torch.manual_seed(seed=seed)
with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats['obs']) ##Need to make stats into a dictionary
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
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
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            ## Replace this with the data we get from april_tags. This can be a function, which takes end
            # effector pose (x,y) as an input, scales it, and then observes the april tag response and scales it
            obs, coverage, reward, done, info = env.step(action[i])
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

# print out the maximum target coverage
print('Score: ', max(rewards))

# visualize
from IPython.display import Video
vwrite('vis_1_01_5.gif', imgs)
#Video('vis__1_01_5.mp4', embed=True, width=1024*4, height=1024*4)