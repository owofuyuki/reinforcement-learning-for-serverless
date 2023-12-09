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
    np.zeros((n, n-1), dtype=np.int)
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
        
        # Define the diagonal matrix of size n x n in the action space
        self.diagonal_matrix_space = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int8)
        # ... (rest of the initialization remains the same)
        
        # Define the matrix of size n x 5 with elements in the set {-1, 0, 1} and row sum = 0
        self.custom_matrix_space = spaces.Box(low=-1, high=1, shape=(self.size, 5), dtype=np.int8)
        
        # Set the last column of the custom matrix space to be always zero
        self.custom_matrix_space.low[:, -1] = 0
        self.custom_matrix_space.high[:, -1] = 0
        
        self._generate_valid_custom_matrix()  # Generate a valid custom matrix
        
        # Combine the two action spaces into a Tuple space
        self.action_space = spaces.Tuple((self.diagonal_matrix_space, self.custom_matrix_space))
        
    def _generate_valid_custom_matrix(self):
        while True:
            # Generate a random matrix
            random_matrix = np.random.randint(-1, 2, size=(self.size, 4))
            # Ensure the row sum is 0
            row_sums = np.sum(random_matrix, axis=1)
            if np.all(row_sums == 0):
                # If row sum is 0, assign the matrix to custom_matrix_space
                self.custom_matrix_space.low[:, :-1] = random_matrix
                self.custom_matrix_space.high[:, :-1] = random_matrix
                break
        
    # ... (rest of the class methods remain the same)

# Example usage:
rlss_env = CustomEnvironment(size=3)  # Create an instance of the custom environment with size = 3

# Generate a sample action using action_space.sample()
action = rlss_env.action_space.sample()

# Display the sampled action
print("Sampled Action:")
print(action[1])

