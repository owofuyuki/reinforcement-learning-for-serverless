import enum
import numpy as np

import gymnasium as gym
from gymnasium import spaces


'''
Define an index corresponding to the action that changes the container's state:
    - Return to previous state: -1
    - Maintain current status: 0
    - Move to the next state: 1
'''
class Actions(enum.Enum):
    previous_state = -1   
    current_state = 0
    next_state = 1


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
        '''
        Define environment parameters
        '''
        self.size = size  # The number of services
        self.num_states = 5  # The number of states in a container's lifecycle (N, L0, L1, L2, A)
        self.num_resources = 3  # The number of resource parameters (RAM, GPU, CPU)
        self.num_actions = len(Actions)
        self.max_container = 256

        '''
        Define observations (state space)
        '''
        self.observation_space = spaces.Dict({
            "request_quantity":  spaces.Box(low=0, high=self.max_container, shape=(self.size, 1), dtype=np.int8),
            "remained_resource": spaces.Box(low=0, high=self.max_container, shape=(self.num_resources, 1), dtype=np.int8),
            "container_traffic": spaces.Box(low=0, high=self.max_container, shape=(self.size, self.num_states), dtype=np.int8),
        })
        
        '''
        Define action space containing two matrices
        '''
        # Define the diagonal matrix 
        # and set the main diagonal elements to be in the range [0, self.max_container] 
        self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int8)
        np.fill_diagonal(self._action_coefficient.low, 0)
        np.fill_diagonal(self._action_coefficient.high, self.max_container)
        
        # Define the matrix with elements in the set {-1, 0, 1} 
        # and set the last column of the custom matrix space to be always zero
        self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, 5), dtype=np.int8)
        self._action_unit.low[:, -1] = 0
        self._action_unit.high[:, -1] = 0
        
        self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))  # Combine the two action matrix into a Tuple space
        
        '''
        Initialize the state and other variables
        '''
        self.current_time = 0  # Start at time 0
        self.timeout = 1000  # Set timeout value = 1000s
        self.limited_ram = 64  # Set limited amount of RAM = 64GB/service
        self._request_matrix = np.ones((self.size, 1), dtype=bool)
        self._resource_matrix = np.zeros((self.num_resources, 1), dtype=bool)
        self._container_matrix = np.hstack((
            np.random.randint(0, 256, size=(self.size, 1)),
            np.zeros((self.size, self.num_states-1), dtype=np.int8)
        ))  # Initially the containers are in Null state

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_reward(self):
        reward = 123456
        return reward      
        
    def _get_obs(self):
        return {
            "request_quantity": self._request_matrix,
            "remained_resource": self._resource_matrix,
            "container_traffic": self._container_matrix,
        }

    def _get_info(self):
        
        return {
            "energy_consumption": 100,
            "penalty_delay": 200,
            "penalty_abandone": 300,
            "system_profit": 400,
        }
        
    def _get_constraints(self):
        
        return True
    
    def reset(self, seed=None, options=None):
        '''
        Initialize the environment
        '''
        self.current_time = 0  # Start at time 0
        self._request_matrix = np.ones((self.size, 1), dtype=bool)
        self._resource_matrix = np.zeros((self.num_resources, 1), dtype=bool)
        self._container_matrix = np.hstack((
            np.random.randint(0, 256, size=(self.size, 1)),
            np.zeros((self.size, self.num_states-1), dtype=np.int8)
        ))  # Initially the containers are in Null state
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        '''
        Apply action masking to check the validity of status updates
        '''
        if (self._get_constraints):
            
        else: pass
        
        '''
        A learning round ends if time exceeds timeout, or resource usage exceeds the limit
        '''
        terminated = (self.current_time >= self.timeout)
        truncated = False
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        '''
        Implement a visualization method (if needed)
        '''
        info = self._get_info()
        reward = self._get_reward()
        print("SYSTEM EVALUATION PARAMETERS:")
        print("- Energy Consumption: {:.2f}J".format(info["energy_consumption"]))
        print("- Delay Penalty:      {:.2f}".format(info["penalty_delay"]))
        print("- Abandone Penalty:   {:.2f}".format(info["penalty_abandone"]))
        print("- Profit:             {:.2f}".format(info["system_profit"]))
        print("- Reward:             {:.2f}".format(reward))

    def close(self):
        '''
        Implement the close function to clean up (if needed)
        '''
        pass


if __name__ == "__main__":
    # Create the network grid environment
    rlss_env = ServerlessEnv()
    
    # Reset the environment to the initial state
    observation, info = rlss_env.reset()
    
    # Perform random actions
    while (True):
        print("----------------------------------------")
        action = rlss_env.action_space.sample()  # Random action
        # print("Action:", action)
        observation, reward, terminated, truncated, info = rlss_env.step(action)
        print(f"Reward: {reward}, Done: {terminated}")
        rlss_env.render()
        if terminated: 
            print("----------------------------------------")
            break
        else: continue
