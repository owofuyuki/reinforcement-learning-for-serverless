# import enum
# import numpy as np

# class Actions(enum.Enum):
#     previous_state = -1
#     current_state = 0
#     next_state = 1   

# action = len(Actions)

# n = 5
# result_matrix = np.hstack((
#     np.random.randint(0, 256, size=(n, 1)),
#     np.zeros((n, n-1), dtype=np.int16)
# ))

# def check():
#     if (n > 6): 
#         pass
#     else:
#         print("n <= 6")
        
#     print("Initialized Matrix:")
#     print(result_matrix)
    
# check()

# import gym
# from gym import spaces
# import numpy as np

# class CustomEnvironment(gym.Env):
#     def __init__(self, size=3):
#         self.size = size
#         self.max_container = 256
#         self.num_states = 5
#         self.num_resources = 3
        
#         '''
#         Define action space containing two matrices by combining them into a Tuple space
#         '''
#         self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int16)
#         self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int16)
#         self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))
        
#         # Set the main diagonal elements of _action_coefficient to be in the range [0, self.max_container] 
#         np.fill_diagonal(self._action_coefficient.low, 0)
#         np.fill_diagonal(self._action_coefficient.high, self.max_container)
        
#         # Set the last column of the _action_unit to be always zero
#         self._action_unit.low[:, -1] = 0
#         self._action_unit.high[:, -1] = 0
        
#         self._generate_valid_custom_matrix()  # Generate a valid custom matrix
        
#         self._resource_matrix = np.zeros((self.num_resources, 1), dtype=np.int16)
#         self._resource_matrix[0, 0] = 100
        
        
        
#     def _generate_valid_custom_matrix(self):
#         while True:
#             random_matrix = np.random.randint(-1, 2, size=(self.size, self.num_states-1))  # Generate a random matrix
#             row_sums = np.sum(random_matrix, axis=1)  # Ensure the row sum is 0
#             if np.all(row_sums == 0):  # If row sum is 0, assign the matrix to _action_unit
#                 self._action_unit.low[:, :-1] = random_matrix
#                 self._action_unit.high[:, :-1] = random_matrix
#                 break
        
#     # ... (rest of the class methods remain the same)

# # Example usage:
# rlss_env = CustomEnvironment(size=3)  # Create an instance of the custom environment with size = 3

# # Generate a sample action using action_space.sample()
# action = rlss_env.action_space.sample()

# # Display the sampled action
# print("Action Coefficient Matrix:")
# print(action[0])
# print("Action Units Matrix:")
# print(action[1])
# print("Sampled Action:")
# print(action[0] @ action[1])

# print(rlss_env._resource_matrix)

# # Create a sample matrix (3x3) for illustration purposes
# matrix = np.array([[1, 2, 3],
#                    [4, 0, 6],
#                    [7, 8, 0]])

# # Check if any element is less than 0
# any_less_than_zero = np.any(matrix < 0)

# print("Any element is less than 0:", any_less_than_zero)

# sub_matrix = np.array([1, 5, 2])
# matrix *= sub_matrix

# print(matrix)

# print(matrix[0, 0])

# a, b = 100, 200

# print(a)
# print(b)

# for i, row in enumerate(matrix):
#     print(f"{i + 1}: {row[0]}") 
    
    
# # Example matrices
# matrix1 = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])

# matrix2 = np.array([
#     [11, 12, 13],
#     [14, 15, 16],
#     [17, 18, 19]
# ])

# for i, (row1, row2) in enumerate(zip(matrix1, matrix2), start=1):
#     for j, (elem1, elem2) in enumerate(zip(row1, row2), start=1):
#         print(f"Matrix 1 - Row: {i}, Col: {j}, Value: {elem1}")
#         print(f"Matrix 2 - Row: {i}, Col: {j}, Value: {elem2}")

# import numpy as np

# array_set = {
#     'array_1': np.array([-1, 1, 0, 0, 0]),
#     'array_2': np.array([1, -1, 0, 0, 0]),
#     'array_3': np.array([0, -1, 1, 0, 0]),
#     'array_4': np.array([0, 1, -1, 0, 0]),
#     'array_5': np.array([0, 0, -1, 1, 0]),
#     'array_6': np.array([0, 0, 1, -1, 0]),
# }

# def initialize_matrix(size):
#     keys = list(array_set.keys())
#     matrix = np.array([array_set[np.random.choice(keys)] for _ in range(size)])
#     return matrix

# # Example usage:
# result_matrix = initialize_matrix(4)
# print(result_matrix)
# print(result_matrix[0])

# print(np.array_equal(result_matrix[0], array_set["array_1"]))

import numpy as np

import gymnasium as gym
from gymnasium import spaces

class Transitions:
    trans_0 = np.array([-1, 1, 0, 0, 0])   # N -> L0
    trans_1 = np.array([1, -1, 0, 0, 0])   # L0 -> N
    trans_2 = np.array([0, -1, 1, 0, 0])   # L0 -> L1
    trans_3 = np.array([0, 1, -1, 0, 0])   # L1 -> L0
    trans_4 = np.array([0, 0, -1, 1, 0])   # L1 -> L2
    trans_5 = np.array([0, 0, 1, -1, 0])   # L2 -> L1
    trans_6 = np.array([1, -1, -1, 1, 0])  # L0 -> N and L1 -> L2
    trans_7 = np.array([-1, 1, 1, -1, 0])  # N -> L0 and L2 -> L1
    trans_8 = np.array([1, -1, 1, -1, 0])  # L0 -> N and L2 -> L1
    trans_9 = np.array([-1, 1, -1, 1, 0]) 

class ServerlessEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, size=4):
        super(ServerlessEnv, self).__init__()
        '''
        Define environment parameters
        '''
        self.size = size  # The number of services
        self.num_states = 5  # The number of states in a container's lifecycle (N, L0, L1, L2, A)
        self.num_resources = 3  # The number of resource parameters (RAM, GPU, CPU)
        self.max_container = 256
        
        self.timeout = 10  # Set timeout value = 10s
        self.container_lifetime = 43200  # Set lifetime of a container = 1/2 day
        self.limited_ram = 64  # Set limited amount of RAM (server) = 64GB
        self.limited_request = 128

        '''
        Define observations (state space)
        '''
        self.observation_space = spaces.Dict({
            "execution_times":   spaces.Box(low=1, high=10, shape=(self.size, 1), dtype=np.int16),
            "request_quantity":  spaces.Box(low=0, high=self.limited_request, shape=(self.size, 1), dtype=np.int16),
            "remained_resource": spaces.Box(low=0, high=self.limited_ram, shape=(self.num_resources, 1), dtype=np.int16),
            "container_traffic": spaces.Box(low=0, high=self.max_container, shape=(self.size, self.num_states), dtype=np.int16),
        })
        
        '''
        Define action space containing two matrices by combining them into a Tuple space
        '''
        self._action_coefficient = spaces.Box(low=0, high=0, shape=(self.size, self.size), dtype=np.int16)
        self._action_unit = spaces.Box(low=-1, high=1, shape=(self.size, self.num_states), dtype=np.int16)
        self.action_space = spaces.Tuple((self._action_coefficient, self._action_unit))
        
        # Set the main diagonal elements of _action_coefficient to be in the range [0, self.max_container] 
        np.fill_diagonal(self._action_coefficient.low, 0)
        np.fill_diagonal(self._action_coefficient.high, self.max_container)
        
        # Set the last column of the _action_unit to be always zero and the sum of the elements in a row of the _action_unit = 0 using _get_units()
        # self._action_unit.low[:, -1] = 0
        # self._action_unit.high[:, -1] = 0
        self._get_units()
        
        '''
        Initialize the state and other variables
        '''
        self.current_time = 0  # Start at time 0
        self._custom_request = np.random.randint(0, 64, size=(self.size, 1))  # Randomly set the number of incoming requests every Δt seconds
        self._pending_request = np.zeros((self.size, 1), dtype=np.int16)  # Set an initial value
        self._ram_required_matrix = np.array([0, 0, 0, 0.9, 2])  # Set the required RAM each state
        self._action_matrix = np.zeros((self.size, self.num_states), dtype=np.int16)  # Set an initial value
        self._exectime_matrix = np.random.randint(2, 16, size=(self.size, 1))
        self._request_matrix = np.zeros((self.size, 1), dtype=np.int16)
        self._resource_matrix = np.ones((self.num_resources, 1), dtype=np.int16)
        self._resource_matrix[0, 0] = self.limited_ram
        self._container_matrix = np.hstack((
            np.random.randint(0, self.max_container, size=(self.size, 1)),  # Initially the containers are in Null state
            np.zeros((self.size, self.num_states-1), dtype=np.int16)
        ))  

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
    def _get_units(self):
        '''
        Define a function to create a random matrix such that the sum of the elements in a row = 0
        '''
        # while True:
        #     random_matrix = np.random.randint(-1, 2, size=(self.size, self.num_states-1))  # Generate a random matrix
        #     row_sums = np.sum(random_matrix, axis=1)  # Ensure the row sum is 0
        #     if np.all(row_sums == 0):  # If row sum is 0, assign the matrix to _action_unit
        #         self._action_unit.low[:, :-1] = random_matrix
        #         self._action_unit.high[:, :-1] = random_matrix
        #         break     
        array_set = [getattr(Transitions, attr) for attr in dir(Transitions) if not attr.startswith("__")]
        self._action_unit = np.array([array_set[np.random.randint(0, len(array_set))] for _ in range(self.size)])

# Example usage:
env = ServerlessEnv()
result_matrix = env._action_unit
print(result_matrix)

# Example matrices (replace these with your actual matrices)
matrix_4x5 = np.random.randint(1, 10, size=(4, 5))  # Example 4x5 matrix
matrix_1x5 = np.array([1, 2, 3, 4, 5])  # Example 1x5 matrix

sum_variable = 0

for row in matrix_4x5:
    product = np.dot(row, matrix_1x5)  # Compute the dot product of each row with the 1x5 matrix
    sum_variable += product  # Add the product to the sum

print(matrix_4x5)
print(matrix_1x5)
print(np.sum(matrix_4x5[0] @ matrix_1x5))
print("Sum of products:", sum_variable)

abc = np.random.randint(0, 64, size=(4, 1))
print(abc)
print(abc[1][0])
    
    