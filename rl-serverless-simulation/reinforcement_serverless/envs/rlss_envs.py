import numpy as np

import gymnasium as gym
from gymnasium import spaces


class ServerlessEnv(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None, size=5):
        super(ServerlessEnv, self).__init__()
        """
        Define environment parameters
        """
        self.size = size
        self.num_states = 5  # The number of states in a container's lifecycle

        """
        - Observations (state space) are dictionaries with 2 traffic matrices "traffic_demand" and "traffic_state".
        - .
        """
        self.observation_space = spaces.Dict(
            {
                "traffic_demand": spaces.Box(
                    low=0, high=256, shape=(self.size, 5), dtype=np.int
                ),
            }
        )

        """
        Initialize the state and other variables
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def close(self):
        """
        Implement the close function to clean up (if needed)
        """
        pass
