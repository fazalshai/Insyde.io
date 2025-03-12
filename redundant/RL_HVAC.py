import gym
import numpy as np
from stable_baselines3 import PPO

# Define a simple HVAC environment
class HVACEnv(gym.Env):
    def __init__(self):
        super(HVACEnv, self).__init__()
        self.room_size = np.array([10, 8, 3])  # Fixed room dimensions (L, W, H)
        self.hvac_pos = np.array([5, 4, 2.5])  # Initial HVAC position
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=15, shape=(3,), dtype=np.float32)

    def step(self, action):
        self.hvac_pos += action  # Move HVAC unit
        self.hvac_pos = np.clip(self.hvac_pos, [0, 0, 0], self.room_size)  # Keep inside room
        reward = -np.sum((self.hvac_pos - np.array([6, 5, 2.5]))**2)  # Reward closer to ideal
        return self.hvac_pos, reward, False, {}

    def reset(self):
        self.hvac_pos = np.array([5, 4, 2.5])
        return self.hvac_pos

env = HVACEnv()

# Train a Deep Q-Network (DQN) to optimize HVAC placement
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Test the trained model
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"New HVAC Position: {obs}, Reward: {reward}")
