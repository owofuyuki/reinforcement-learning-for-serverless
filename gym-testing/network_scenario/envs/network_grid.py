from enum import Enum
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class NetworkGridEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, size=5):
        '''
        Define environment parameters
        '''
        self.size = size  # The size of the square grid
        self.num_bs = self.size ** 2
        self.num_actions = self.num_bs * 17
        self.current_time = 0  # Initial time

        '''
        - Observations (state space) are dictionaries with 2 traffic matrices "traffic_demand" and "traffic_state".
        - There are 17 actions for each BS, load can be split equally in up to 4 directions.
        - 
        - 
        '''
        self.observation_space = spaces.Dict(
            {
                "traffic_demand": spaces.Box(low=0, high=1, shape=(self.size, self.size)),
                "traffic_state": spaces.Box(low=0, high=1, shape=(self.size, self.size))
            }
        )
        
        self.action_space = spaces.Discrete(self.num_actions)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_obs(self):
        return {"traffic_demand": self._demand_matrix, "traffic_state": self._state_matrix}
    
    def _get_info(self):
        pass
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Need the this line to seed self.np_random
        
        '''
        Initialize the environment
        '''
        self.current_time = 0
        self.active_bs = np.ones((self.size, self.size), dtype=bool)
        self._demand_matrix = np.full((self.size, self.size), 0.1)
        self._state_matrix = np.full((self.size, self.size), 0.1)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        pass
    
    def render(self):
        pass
    
    def close(self):
        pass