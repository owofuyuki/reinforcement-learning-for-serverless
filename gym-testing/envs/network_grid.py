import numpy as np

import gymnasium as gym
from gymnasium import spaces


class NetworkGridEnv(gym.Env):
    metadata = {}
    
    def __init__(self, render_mode=None, size=5):
        self.size = size 