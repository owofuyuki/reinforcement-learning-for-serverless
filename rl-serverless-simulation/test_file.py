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
    np.zeros((n, n-1), dtype=np.int16)
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
        self.num_resources = 3
        
        '''
        Define action space containing two matrices by combining them into a Tuple space
        '''
        self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int16)
        self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int16)
        self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))
        
        # Set the main diagonal elements of _action_coefficient to be in the range [0, self.max_container] 
        np.fill_diagonal(self._action_coefficient.low, 0)
        np.fill_diagonal(self._action_coefficient.high, self.max_container)
        
        # Set the last column of the _action_unit to be always zero
        self._action_unit.low[:, -1] = 0
        self._action_unit.high[:, -1] = 0
        
        self._generate_valid_custom_matrix()  # Generate a valid custom matrix
        
        self._resource_matrix = np.zeros((self.num_resources, 1), dtype=np.int16)
        self._resource_matrix[0, 0] = 100
        
        
        
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
print("Action Coefficient Matrix:")
print(action[0])
print("Action Units Matrix:")
print(action[1])
print("Sampled Action:")
print(action[0] @ action[1])

print(rlss_env._resource_matrix)

# Create a sample matrix (3x3) for illustration purposes
matrix = np.array([[1, 2, 3],
                   [4, 0, 6],
                   [7, 8, 0]])

# Check if any element is less than 0
any_less_than_zero = np.any(matrix < 0)

print("Any element is less than 0:", any_less_than_zero)

sub_matrix = np.array([1, 5, 2])
matrix *= sub_matrix

print(matrix)

print(matrix[0, 0])

a, b = 100, 200

print(a)
print(b)

for i, row in enumerate(matrix):
    print(f"{i + 1}: {row[0]}") 
    
    
# Example matrices
matrix1 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

matrix2 = np.array([
    [11, 12, 13],
    [14, 15, 16],
    [17, 18, 19]
])

for i, (row1, row2) in enumerate(zip(matrix1, matrix2), start=1):
    for j, (elem1, elem2) in enumerate(zip(row1, row2), start=1):
        print(f"Matrix 1 - Row: {i}, Col: {j}, Value: {elem1}")
        print(f"Matrix 2 - Row: {i}, Col: {j}, Value: {elem2}")