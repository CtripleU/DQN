import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from education_env import EducationEnv

env = EducationEnv()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights('dqn_education_weights.h5f')

for episode in range(5):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(dqn.model.predict(state.reshape(1, 1, 5, 5))[0])
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
