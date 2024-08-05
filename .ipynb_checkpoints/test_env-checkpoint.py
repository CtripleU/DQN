# test_env.py
import gym
from education_env import EducationEnv

env = EducationEnv()

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action
    state, reward, done, _ = env.step(action)
    env.render()
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
