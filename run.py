
# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path
import airsim
import logging
from ActorCritic import Episode, Experience, Environment
# import numpy as np
# import time
# import os
# import pprint
# import tempfile
# import math
# from math import *
# # from scipy.misc import imsave
# import imageio
# import cv2
#
# from abc import ABC, abstractmethod
# from path_planner import *

from utils import *
from vehicles import Car, Drone


def play_game(pos=[0, 5, -1], goal=(120, 0), uav_size=[0.29*3, 0.98*2], hfov = radians(90), coll_thres=5, yaw=0,
              limit_yaw=5, step=0.1):
    # create vehicle objects
    drone = Drone()
    car = Car()

    # tmp_dir = "airsim_drone"
    #
    # responses = drone.client.simGetImages([
    #     airsim.ImageRequest("1", airsim.ImageType.DepthPlanner, True)], vehicle_name="Drone1")
    # response = responses[0]

    # initial position
    # --- Load models -------------------------------------------------------------------------------------------
    #todo actor, critic = load_a2c_models()
    # --- Plan path to goal -------------------------------------------------------------------------------------
    topview, filename = get_topview_image(190, drone=drone)
    obstacles = PathPlanningObstacles(filename, proportion_x=1, proportion_y=1)
    goals, planner = path_planning(obstacles=obstacles, topview=topview, x_goal=goal)

    # --- send vehicle and drone to initial positions (random at each game/episode) ------------------------------
    drone.move(pos, yaw)
    car.move(pos, yaw)

    # ------------------------------------------------------------------------------------------------------------
    env = Environment(drone=drone, car=car, planner=planner, goals=goals)

    current_state_vec = env.reset(pos, yaw)
    print('Init state:', current_state_vec)

    while len(env.goals):
        for dt in range(10000):

            # GET ACTION
            # todo:
            # ADD DISCRITIZATION OF STATES (i.e. 0:0.1:1)
            # value, policy_dist = actor_critic.forward(state)
            #
            # value = value.detach().numpy()[0, 0]
            # dist = policy_dist.detach().numpy()
            #
            # action = np.random.choice(num_outputs, p=np.squeeze(dist))
            # log_prob = torch.log(policy_dist.squeeze(0)[action])
            # entropy = -np.sum(np.mean(dist) * np.log(dist))

            value, log_probs, action = 0.5, 0.5, 0.2 #todo remove
            # todo - add entropy accumulator

            # EXECUTE ACTION
            new_state, reward, done, _ = env.step(action)

            episode.add_experience(Experience(current_state_vec, action, reward, new_state, value, log_probs, episode.uuid))

            if done:
                print('Target reached.')
                episode.logger.info('target_reached')
                break

            current_state_vec = new_state
            print('New State Vector:', current_state_vec)

    print('Drone Reached Final Target')
    episode.logger.info('final_target_reached')

    ''' NEED TO DECIDE WHERE WE WANT THIS:
    Qval_final, _ = actor_critic.forward(state)
    Qval_final = Qval_final.detach().numpy()[0, 0]
    Qvals = episode.compute_Qvals(Qval_final)

    # update actor critic
    values = torch.FloatTensor(episode.values)
    Qvals = torch.FloatTensor(Qvals)
    log_probs = torch.stack(episdoe.log_probs)

    advantage = Qvals - values
    actor_loss = (-log_probs * advantage).mean()
    critic_loss = 0.5 * advantage.pow(2).mean()
    ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

    ac_optimizer.zero_grad()
    ac_loss.backward()
    ac_optimizer.step()
    
    ADD A WAY TO SAVE THIS (WILL RELOAD ON EACH EPISODE....)
    
    '''
    # currently reset() doesn't work in CV mode. Below is the workaround
    # drone.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)


# ---- MAIN ---------------------------------------------------------

if __name__ == '__main__':
    episode_uuid = round(time.time())
    logger = setup_logger('logger', 'episodes', f'episode_{episode_uuid}_{time.strftime("%a,%d_%b_%Y_%H_%M_%S")}.txt')
    logger.info(f'Episode: {episode_uuid} - START -')
    episode = Episode(episode_uuid, logger)
    play_game()
