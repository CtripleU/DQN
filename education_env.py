import gym
import numpy as np
from gym import spaces


class EducationEnv(gym.Env):
    def __init__(self):
        super(EducationEnv, self).__init__()

        # action space
        # 0: reading, 1: solving math problems, 2: participating in group discussions
        # 3: select easy difficulty, 4: select medium difficulty, 5: select hard difficulty
        # 6: request help from teacher
        self.action_space = spaces.Discrete(7)

        # observation space
        # [reading_skill, math_skill, social_skill, current_difficulty, focus_level, help_requested]
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        # initialize state
        self.state = None
        self.steps = 0
        self.max_steps = 50
        self.current_task = None

    def reset(self):
        self.state = np.array([0.3, 0.3, 0.3, 0.5, 1.0, 0.0])  # Initial state
        self.steps = 0
        self.current_task = None
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"

        # update state based on action
        if action <= 2:  # learning activities
            self._perform_learning_activity(action)
        elif action <= 5:  # difficulty selection
            self._select_difficulty(action - 3)
        elif action == 6:  # request help
            self._request_help()

        # simulate ADHD effects
        self._apply_adhd_effects()

        # calculate reward
        reward = self._calculate_reward()

        # increment step counter
        self.steps += 1

        # check if episode is done
        done = self.steps >= self.max_steps

        # clip state values between 0 and 1
        self.state = np.clip(self.state, 0, 1)

        return self.state, reward, done, {}

    def _perform_learning_activity(self, activity):
        self.current_task = activity
        if activity == 0:  # Reading
            self.state[0] += np.random.uniform(0.05, 0.1) * self.state[4]  # affected by focus
        elif activity == 1:  # Math
            self.state[1] += np.random.uniform(0.05, 0.1) * self.state[4]  # affected by focus
        elif activity == 2:  # Group discussion
            self.state[2] += np.random.uniform(0.05, 0.1) * self.state[4]  # affected by focus

    def _select_difficulty(self, level):
        self.state[3] = (level + 1) / 3  # easy: 0.33, medium: 0.67, hard: 1.0

    def _request_help(self):
        self.state[5] = 1.0  # requested help
        self.state[4] += 0.1  # slight increase in focus due to help

    def _apply_adhd_effects(self):
        # randomly decrease focus
        if np.random.random() < 0.3:  # 30% chance of distraction
            self.state[4] -= np.random.uniform(0.1, 0.3)

        # increase focus if the task matches the student's strength
        if self.current_task is not None:
            if self.state[self.current_task] == max(self.state[:3]):
                self.state[4] += 0.1

        # decrease help requested flag over time
        self.state[5] *= 0.9

    def _calculate_reward(self):
        reward = 0

        # reward for skill improvement
        reward += np.sum(self.state[:3]) * 0.3

        # reward for maintaining focus
        reward += self.state[4] * 0.2

        # penalty for off-task behavior (low focus)
        if self.state[4] < 0.3:
            reward -= 0.5

        # reward for appropriate difficulty level
        optimal_difficulty = np.mean(self.state[:3])
        reward -= abs(self.state[3] - optimal_difficulty) * 0.2

        # small reward for seeking help when needed
        if self.state[5] > 0.5 and self.state[4] < 0.5:
            reward += 0.2

        return reward

    def render(self, mode='human'):
        print(f"Current State: {self.state}")
        print(f"Steps: {self.steps}/{self.max_steps}")
        print(f"Current Task: {['Reading', 'Math', 'Group Discussion', 'None'][self.current_task if self.current_task is not None else 3]}")
        print(f"Difficulty: {['Easy', 'Medium', 'Hard'][int(self.state[3] * 3) - 1]}")
        print(f"Focus Level: {self.state[4]:.2f}")
        print(f"Help Requested: {'Yes' if self.state[5] > 0.5 else 'No'}")
