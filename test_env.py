import gym
from education_env import EducationEnv

def test_environment():
    env = EducationEnv()
    
    for episode in range(3):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\nEpisode: {episode + 1}")
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            print(f"\nStep: {step + 1}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            env.render()
            
            state = next_state
            step += 1
        
        print(f"\nEpisode {episode + 1} finished after {step} steps")
        print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    test_environment()