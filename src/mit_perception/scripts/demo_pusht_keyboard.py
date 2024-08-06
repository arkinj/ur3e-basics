import numpy as np
import click
from mit_perception.replay_buffer import ReplayBuffer
from mit_perception.env_T import PushTEnv
import pygame

import mit_perception.move_utils as arm
from mit_perception.state_estimator_T import TAG_SIZES
from mit_perception.apriltag_utils import RealsenseCam, AprilTag

import cv2

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-rs', '--render_size', default=96, type=int)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('-s', '--simulation', is_flag=True, default=False)
@click.option('-mg', '--move-gripper', is_flag=True, default=False)

def main(output, render_size, control_hz, simulation, move_gripper):
    """
    Collect demonstration for the Push-T task.
    
    Usage: python demo_pusht.py -o data/pusht_demo.zarr
    
    This script is compatible with both Linux and MacOS.
    Hover mouse close to the blue circle to start.
    Push the T block into the green area. 
    The episode will automatically terminate if the task is succeeded.
    Press "Q" to exit.
    Press "R" to retry.
    Hold "Space" to pause.
    """

    if not simulation:
        move_group_arm, move_group_hand = arm.setup()
        arm.move_to_home_pose(move_group_arm)
        pose = move_group_arm.get_current_pose()
        pose.pose.position.z = 0.265
        arm.move_to_pose(move_group_arm, pose.pose)
        if move_gripper:
            arm.open_gripper(move_group_hand)
            arm.close_gripper(move_group_hand)
            
        # Initialize the camera and AprilTag detector
        cams = [
            RealsenseCam(
                "317422075665",
                (1280, 720), # color img size
                (1280, 720), # depth img size
                30), # frame rate
            RealsenseCam(
                "243222071097",
                (1280, 720),
                (1280, 720),
                30)]
        # tag size in meters, or dict of tag_size: tag_ids
        april_tag = AprilTag(tag_size=TAG_SIZES) 
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # create PushT env with keypoints
    env = PushTEnv(x0=256,y0=256,mass=0.1,friction=1,length=4, render_size=render_size)
    agent = env.teleop_agent()
    # pygame.init()
    # clock = pygame.time.Clock()

    obs = env.reset()
    img = env.render(mode='human')
    
    # episode-level while loop
    while True:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f'starting seed {seed}')
        
        # reset env and get observations (including info and render for recording)
        obs = env.reset()
        info = env._get_info()
        img = env.render(mode='rgb_array')
        
        # loop state
        retry = False
        pause = False
        done = False
        plan_idx = 0
        # pygame.display.set_caption(f'plan_idx:{plan_idx}')
        # step-level while loop
        while not done:
            # process keypress events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # hold Space to pause
                        plan_idx += 1
                        pygame.display.set_caption(f'plan_idx:{plan_idx}')
                        pause = True
                    elif event.key == pygame.K_r:
                        # press "R" to retry
                        retry=True
                    elif event.key == pygame.K_q:
                        # press "Q" to exit
                        exit(0)
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        pause = False

            # handle control flow
            if retry:
                break
            if pause:
                continue
            
            # get action from mouse
            # None if mouse is not close to the agent
            act = agent.act(obs)
            print(act)
            # print(act, obs)
            # act = np.random.rand((2))*512
            # print(pygame.mouse.get_pos())
            if not act is None:
                # teleop started
                # state dim 2+3
                state = np.concatenate([info['pos_agent'], info['block_pose']])
                data = {
                    'img': img,
                    'state': np.float32(state),
                    # 'keypoint': np.float32(keypoint),
                    'action': np.float32(act),
                    'n_contacts': np.float32([info['n_contacts']])
                }
                episode.append(data)
                
            # step env and render
            if simulation:
                obs, coverage, reward, done, info = env.step(act)
            else:
                action = np.array(act).reshape((1,2)) if act is not None else None
                obs, coverage, reward, done, info = env.step_real(action, move_group_arm, april_tag, cams)
            img = env.render(mode='human')

            # print(obs, coverage, reward, done, info)
            
            # regulate control frequency
            env.clock.tick(control_hz)
            

        # after done
        if not retry:
            # save episode buffer to replay buffer (on disk)
            data_dict = dict()
            for key in episode[0].keys():
                data_dict[key] = np.stack(
                    [x[key] for x in episode])
            replay_buffer.add_episode(data_dict, compressors='disk')
            print(f'saved seed {seed}')
        else:
            print(f'retry seed {seed}')


if __name__ == "__main__":
    main()