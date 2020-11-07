# Original author: Roma Sokolkov
# Edited by Antonin Raffin
# Edited by Sheelabhadra Dey

import random
import time
import os
import warnings

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from config import INPUT_DIM, MIN_STEERING, MAX_STEERING, JERK_REWARD_WEIGHT, MAX_STEERING_DIFF
from config import ROI, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE, REWARD_CRASH, CRASH_SPEED_WEIGHT


class JetVAEEnv(object):
    def __init__(self, vae=None, jet_racer=None, min_throttle=0.45, max_throttle=0.6, 
                 n_command_history=0, frame_skip=1, n_stack=1, 
                 action_lambda=0.5):
        # JetRacer object
        self.jet = jet_racer

        # save last n commands
        self.n_commands = 2
        self.n_command_history = n_command_history
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_stack = n_stack
        self.stacked_obs = None

        # assumes that we are using VAE input
        self.vae = vae
        self.z_size = None
        if vae is not None:
            self.z_size = vae.z_size
        
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, self.z_size),
            dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32)

        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.frame_skip = frame_skip
        self.action_lambda = action_lambda
        self.last_throttle = 0.0

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.command_history[0, -2 * (i + 1)]
                prev_steering = self.command_history[0, -2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (MAX_STEERING - MIN_STEERING)

                if abs(steering_diff) > MAX_STEERING_DIFF:
                    error = abs(steering_diff) - MAX_STEERING_DIFF
                    jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def postprocessing_step(self, action, observation, reward, done, info):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).
        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        jerk_penalty = 0 # self.jerk_penalty()
        # Cancel reward if the continuity constrain is violated
        if jerk_penalty > 0 and reward > 0:
            reward = 0
        reward -= jerk_penalty

        if self.n_stack > 1:
            self.stacked_obs = np.roll(self.stacked_obs, shift=-observation.shape[-1], axis=-1)
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs, reward, done, info

        return observation, reward, done, info

    def step(self, action):
        # Convert from [-1, 1] to [0, 1]
        t = (action[1] + 1) / 2
        # Convert from [0, 1] to [min, max]
        action[1] = (1 - t) * self.min_throttle + self.max_throttle * t

        # Clip steering angle rate to enforce continuity
        if self.n_command_history > 0:
            prev_steering = self.command_history[0, -2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff

        self.jet.apply_throttle(action[1])
        self.jet.apply_steering(action[0])

        # Repeat action if using frame_skip
        for _ in range(self.frame_skip):
            self.jet.apply_throttle(action[1])
            self.jet.apply_steering(action[0])
            im = self.jet.get_image()
            observation = self.vae.encode_from_raw_image(im)
            reward, done = self.reward()

        self.last_throttle = action[1]

        return observation, reward, done, {}

    def reset(self):
        print("Start to reset env")
        im = self.jet.get_image()
        observation = self.vae.encode_from_raw_image(im)

        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))

        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)

#         if self.n_stack > 1:
#             self.stacked_obs[...] = 0
#             self.stacked_obs[..., -observation.shape[-1]:] = observation
#             return self.stacked_obs
        
        self.jet.apply_throttle(0)
        self.jet.apply_steering(0)
        
        print('reset finished')
        return observation


    def reward(self):
        """
        :param measurements:
        :return: reward, done
        """
        done = False

        """distance"""

        """speed"""
        # # In the wayve.ai paper, speed has been used as reward
        # SPEED_REWARD_WEIGHT = 0.1
        # speed_reward = SPEED_REWARD_WEIGHT*measurements.player_measurements.forward_speed

        """crash/off-road"""
        #### TO-DO ###
        # if there is an intervention using the joystick
        # register a negative reward
#         if intervention:
#             norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
#             return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle, done

        """staying on road"""
        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        return 1 + throttle_reward, done
