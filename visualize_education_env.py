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

env = EducationEnv()
nb_actions = env.action_space.n

# # Build the model
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(24))
# model.add(Activation('relu'))
# model.add(Dense(24))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))

# memory = SequentialMemory(limit=50000, window_length=1)
# policy = EpsGreedyQPolicy()
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
#                target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# load trained weights
dqn.load_weights('dqn_education_weights.h5f')

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Education Environment Visualization")

def draw_environment(state, action, reward, episode, step):
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

    action_texts = ['Reading', 'Math', 'Group Discussion', 'Easy', 'Medium', 'Hard', 'Request Help']
    screen.blit(pygame.font.SysFont(None, 24).render(f'Action: {action_texts[action]}', True, (0, 0, 0)), (50, 310))
    screen.blit(pygame.font.SysFont(None, 24).render(f'Reward: {reward:.2f}', True, (0, 0, 0)), (50, 340))
    screen.blit(pygame.font.SysFont(None, 24).render(f'Episode: {episode}', True, (0, 0, 0)), (50, 370))
    screen.blit(pygame.font.SysFont(None, 24).render(f'Step: {step}', True, (0, 0, 0)), (50, 400))

    pygame.display.flip()

running = True
paused = False
state = env.reset()
episode = 1
step = 0
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_UP:
                clock.tick(10)  # increase speed
            elif event.key == pygame.K_DOWN:
                clock.tick(1)  # decrease speed

    if not paused:
        action = np.argmax(dqn.model.predict(state.reshape(1, 1, 6))[0])
        next_state, reward, done, _ = env.step(action)

        draw_environment(next_state, action, reward, episode, step)

        state = next_state
        step += 1

        if done:
            state = env.reset()
            episode += 1
            step = 0

    clock.tick(5)

pygame.quit()