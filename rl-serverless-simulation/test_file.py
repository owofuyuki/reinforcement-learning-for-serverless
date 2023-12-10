import enum
import numpy as np

class Actions(enum.Enum):
    previous_state = -1
    current_state = 0
    next_state = 1   

action = len(Actions)

n = 5
result_matrix = np.hstack((
    np.random.randint(0, 256, size=(n, 1)),
    np.zeros((n, n-1), dtype=np.int8)
))

def check():
    if (n > 6): 
        pass
    else:
        print("n <= 6")
        
    print("Initialized Matrix:")
    print(result_matrix)
    
check()

import gym
from gym import spaces
import numpy as np

class CustomEnvironment(gym.Env):
    def __init__(self, size=3):
        self.size = size
        self.max_container = 256
        self.num_states = 5
        
        '''
        Define action space containing two matrices by combining them into a Tuple space
        '''
        self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int8)
        self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int8)
        self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))
        
        # Set the main diagonal elements of _action_coefficient to be in the range [0, self.max_container] 
        np.fill_diagonal(self._action_coefficient.low, 0)
        np.fill_diagonal(self._action_coefficient.high, self.max_container)
        
        # Set the last column of the _action_unit to be always zero
        self._action_unit.low[:, -1] = 0
        self._action_unit.high[:, -1] = 0
        
        self._generate_valid_custom_matrix()  # Generate a valid custom matrix
        
    def _generate_valid_custom_matrix(self):
        while True:
            random_matrix = np.random.randint(-1, 2, size=(self.size, self.num_states-1))  # Generate a random matrix
            row_sums = np.sum(random_matrix, axis=1)  # Ensure the row sum is 0
            if np.all(row_sums == 0):  # If row sum is 0, assign the matrix to _action_unit
                self._action_unit.low[:, :-1] = random_matrix
                self._action_unit.high[:, :-1] = random_matrix
                break
        
    # ... (rest of the class methods remain the same)

# Example usage:
rlss_env = CustomEnvironment(size=3)  # Create an instance of the custom environment with size = 3

# Generate a sample action using action_space.sample()
action = rlss_env.action_space.sample()

# Display the sampled action
print("Sampled Action:")
print(action[1])

