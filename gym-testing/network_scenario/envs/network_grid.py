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
        '''
        self.observation_space = spaces.Dict(
            {
                "traffic_demand": spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.float32),
                "traffic_state": spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.float32)
            }
        )
        
        self.action_space = spaces.Discrete(self.num_actions)
        
        '''
        Initialize the state and other variables
        '''
        self.current_time = 0
        self.active_bs = np.ones((self.size, self.size), dtype=bool)
        self.prev_bs = (-1, -1)  # Set an initial value
        self._demand_matrix = np.full((self.size, self.size), 0.1)
        self._state_matrix = np.full((self.size, self.size), 0.1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_reward(self):
        energy_csm = []
        
        
        traffic_loss
        return reward
        
    def _get_obs(self):
        return {"traffic_demand": self._demand_matrix, "traffic_state": self._state_matrix}
    
    def _get_info(self):
        return {
            "traffic_coverage",
            "energy_saving"
        }
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        self.current_time = 0
        self.active_bs = np.ones((self.size, self.size), dtype=bool)
        self.prev_bs = -1, -1  # Set an initial value
        self._demand_matrix = np.full((self.size, self.size), 0.1)
        self._state_matrix = np.full((self.size, self.size), 0.1)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        '''
        Check if action is within the valid range [0, 424]
        '''
        if action < 0 or action > 424:
            raise ValueError("Invalid action. Action must be in the range [0, 424].")
        
        '''
        Determine the action and the BS location to perform the action
        '''
        bs_row, bs_col, bs_action = action // 17, action % 17, action % 17
        bs_row, bs_col = bs_row // self.size, bs_row % self.size
        
        '''
        Implement action masking to ensure valid actions and update traffic state based on load shifts
        '''
        if ((bs_row, bs_col) == self.prev_bs):
        # Check if the current BS is the same as the previous BS
            return  ###
        
        if (bs_action == 0):
            # Action 0: BS turning ON
            self.active_bs[bs_row, bs_col] = 1
        elif (bs_action >= 1 and bs_action <= 15):
            # Action 1-15: Bs turning OFF and shifting loads
            if (self.active_bs[bs_row, bs_col] == 0):
                # Do not turn off an already deactivated BS
                return ###
            else:
                self.active_bs[bs_row, bs_col] = 0
                self._state_matrix[bs_row, bs_col] = 0
                bs_left, bs_top = (bs_action & 0b1000) >> 3, (bs_action & 0b0100) >> 2
                bs_right, bs_bottom = (bs_action & 0b0010) >> 1, (bs_action & 0b0001)
                corner_idx = self.size - 1
                
                # BSs at edge may have less than 4 neighbors
                if ((bs_row, bs_col) == (0, 0)):
                    shift_load = bs_right + bs_bottom
                    self._state_matrix[0, 1] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    self._state_matrix[1, 0] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif ((bs_row, bs_col) == (0, corner_idx)):
                    shift_load = bs_left + bs_bottom
                    self._state_matrix[0, corner_idx - 1] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    self._state_matrix[1, corner_idx] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif ((bs_row, bs_col) == (corner_idx, 0)):
                    shift_load = bs_top + bs_right
                    self._state_matrix[corner_idx - 1, 0] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    self._state_matrix[1, corner_idx] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif ((bs_row, bs_col) == (corner_idx, corner_idx)):
                    shift_load = bs_top + bs_left
                    self._state_matrix[corner_idx - 1, corner_idx] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    self._state_matrix[corner_idx, corner_idx - 1] += (self._state_matrix[bs_row, bs_col] / shift_load)
                    
                elif (bs_row == 0):
                    shift_load = bs_left + bs_right + bs_bottom
                    for dr, dc in [(0, -1), (0, 1), (1, 0)]:
                        self._state_matrix[dr, bs_col + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                elif (bs_row == corner_idx):
                    shift_load = bs_left + bs_right + bs_top
                    for dr, dc in [(0, -1), (0, 1), (-1, 0)]:
                        self._state_matrix[corner_idx + dr, bs_col + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                elif (bs_col == 0):
                    shift_load = bs_right + bs_top + bs_bottom
                    for dr, dc in [(-1, 0), (1, 0), (0, 1)]:
                        self._state_matrix[bs_row + dr, dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                elif (bs_col == corner_idx):
                    shift_load = bs_left + bs_top + bs_bottom
                    for dr, dc in [(-1, 0), (1, 0), (0, -1)]:
                        self._state_matrix[bs_row + dr, corner_idx + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)

                else:
                    shift_load = bs_left + bs_top + bs_right + bs_bottom
                    for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        self._state_matrix[bs_row + dr, bs_col + dc] += (self._state_matrix[bs_row, bs_col] / shift_load)
               
        else: pass  # Action 16: Do nothing
        
        '''
        Increment the time step
        '''
        self.current_time += 1
        
        '''
        Calculate the reward
        '''
        
        
        '''
        An episode is done if ...
        '''
        observation = self._get_obs()
        info = self._get_info()
        '''
        Update the previous BS
        '''
        self.prev_bs = bs_row, bs_col
        
        return observation, info
        
    
    def render(self):
        '''
        Implement a visualization method if needed
        '''
        pass
    
    def close(self):
        pass