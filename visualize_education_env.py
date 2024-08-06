import numpy as np
import gym
import tensorflow as tf
import pygame

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from keras import __version__
tf.keras.__version__ = __version__

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from education_env import EducationEnv

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Education Environment Visualization')

# Load the environment
env = EducationEnv()

# Load the model
nb_actions = env.action_space.n
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Load the weights
dqn.load_weights('dqn_education_weights.h5f')

def draw_environment(state):
    screen.fill((255, 255, 255))

    pygame.draw.rect(screen, (255, 0, 0), (50, 50, state[0] * 200, 30))
    pygame.draw.rect(screen, (0, 0, 0), (50, 50, 200, 30), 2)
    screen.blit(pygame.font.SysFont(None, 24).render('Reading Skill', True, (0, 0, 0)), (50, 55))

    pygame.draw.rect(screen, (0, 255, 0), (50, 100, state[1] * 200, 30))
    pygame.draw.rect(screen, (0, 0, 0), (50, 100, 200, 30), 2)
    screen.blit(pygame.font.SysFont(None, 24).render('Math Skill', True, (0, 0, 0)), (50, 105))

    pygame.draw.rect(screen, (0, 0, 255), (50, 150, state[2] * 200, 30))
    pygame.draw.rect(screen, (0, 0, 0), (50, 150, 200, 30), 2)
    screen.blit(pygame.font.SysFont(None, 24).render('Social Skill', True, (0, 0, 0)), (50, 155))

    pygame.draw.rect(screen, (255, 255, 0), (50, 200, state[4] * 200, 30))
    pygame.draw.rect(screen, (0, 0, 0), (50, 200, 200, 30), 2)
    screen.blit(pygame.font.SysFont(None, 24).render('Focus Level', True, (0, 0, 0)), (50, 205))

    difficulty_text = ['Easy', 'Medium', 'Hard'][int(state[3] * 3) - 1]
    screen.blit(pygame.font.SysFont(None, 24).render(f'Difficulty: {difficulty_text}', True, (0, 0, 0)), (50, 250))

    help_text = 'Yes' if state[5] > 0.5 else 'No'
    screen.blit(pygame.font.SysFont(None, 24).render(f'Help Requested: {help_text}', True, (0, 0, 0)), (50, 280))

    pygame.display.flip()

# Main loop
running = True
state = env.reset()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Take a step in the environment
    action = env.action_space.sample()  # Replace with your agent's action
    state, reward, done, info = env.step(action)

    # Draw the environment
    draw_environment(state)

    if done:
        state = env.reset()

    pygame.time.wait(100)

pygame.quit()