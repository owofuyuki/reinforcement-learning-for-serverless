import numpy as np

import gymnasium as gym
from gymnasium import spaces

'''
Defines symbols in a state machine:
    N  = Null
    L0 = Cold
    L1 = Warm Disk
    L2 = Warm CPU
    A  = Active
'''

class ServerlessEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, size=5):
        super(ServerlessEnv, self).__init__()
        """
        Define environment parameters
        """
        self.size = size
        self.num_states = 5  # The number of states in a container's lifecycle (N, L0, L1, L2, A)
        self.num_resources = 3  # The number of resource parameters (GPU. CPU, RAM)

        """
        - Observations (state space) are dictionaries with 3 matrices "container_quantity",
          "request_quantity" and "remained_resource".
        - .
        """
        self.observation_space = spaces.Dict({
            "container_traffic": spaces.Box(low=0, high=256, shape=(self.size, self.num_states), dtype=np.int),
            "request_quantity":  spaces.Box(low=0, high=256, shape=(self.size, 1), dtype=np.int),
            "remained_resource": spaces.Box(low=0, high=256, shape=(self.num_resources, 1), dtype=np.int),
        })

        """
        Initialize the state and other variables
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def _get_reward(self):
        reward = 123456
        return reward
        
        
    def _get_obs(self):
        return {
            "container_traffic": self._container_matrix,
            "request_quantity": self._request_matrix,
            "remained_resource": self._resource_matrix,
        }


    def _get_info(self):
        
        return {
            "system_profit": 100,
            "energy_consumption": 200,
            "penalty_delay": 300,
            "penalty_abandone": 400
        }
    
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    
    def render(self):
        '''
        Implement a visualization method (if needed)
        '''
        info = self._get_info()
        reward = self._get_reward()
        print("SYSTEM EVALUATION PARAMETERS:")
        print("- Energy Consumption: {:.2f}%".format(info["energy_consumption"]))
        print("- Delay Penalty:      {:.2f}%".format(info["penalty_delay"]))
        print("- Abandone Penalty:   {:.2f}%".format(info["penalty_abandone"]))
        print("- Profit:             {:.2f}%".format(info["system_profit"]))
        print("- Reward:             {:.2f}%".format(reward))


    def close(self):
        """
        Implement the close function to clean up (if needed)
        """
        pass



if __name__ == "__main__":
    # Create the network grid environment
    rlss_env = ServerlessEnv()
    
    # Reset the environment to the initial state
    observation, info = rlss_env.reset()
