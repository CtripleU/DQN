## Project Overview

In line with my mission to leverage my research skills and expertise in Machine Learning/Artificial Intelligence to develop innovative tools that address critical challenges in healthcare and education, this project uses a Deep Q-Network (DQN) to simulate a personalized learning approach for a student with Attention Deficit Hyperactivity Disorder (ADHD) in a virtual classroom environment. The environment, `EducationEnv`, models various learning activities and their impacts on the student's focus and skill levels. The agent interacts with various educational activities to find the optimal learning strategy.


## Project Structure

- `education_env.py`: Defines the custom Gym environment `EducationEnv`.
- `train.py`: Trains the DQN agent using Keras-RL.
- `play.py`: Simulates the trained agent's behavior.
- `visualize_education_env.py`: Script to visualize the environment using `pygame`.
- `requirements.txt` : Contains the requirements for this project.


## Custom Environment: EducationEnv

The `EducationEnv` simulates a classroom where the agent represents a student with ADHD. The agent can choose between various learning activities, select difficulty levels, and request help from a teacher. The environment is designed to encourage positive learning behaviors and focus, while also accounting for the challenges faced by students with ADHD.



## Actions
The agent can take one of the following actions:

- Read a book (action 0)
- Solve math problems (action 1)
- Join group discussions (action 2)
- Choose an easy task (action 3)
- Choose a hard task (action 4)
- Ask the teacher for help (action 5)

    ### Difficulty Levels:
    Easy, medium, hard

## States
The agent can choose between various learning activities like reading, solving math problems, or participating in group discussions.
Additional actions include selecting the level of difficulty or requesting help from a teacher.


## Rewards
Based on task completion, skill improvement, focus maintenance, and appropriate difficulty level.
The agent receives rewards based on their actions:

Reading: +1 point
Math: +2 points
Discussion: +1 point
Easy task: +1 point
Hard task: +3 points
Ask for help: -1 point


## Termination Conditions
The episode terminates when the agent completes 50 steps or completes all tasks.

## Training the Agent

I used Keras-RL's `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy` to train the agent. The agent learns to optimize its learning strategies by interacting with the environment, receiving rewards, and updating its policy based on Q-values.
The DQN agent is trained using:
- **Neural Network**: Two hidden layers with 24 units each.
- **Memory**: Sequential memory with a limit of 50000 steps.
- **Policy**: Epsilon-greedy policy for exploration.

## Simulation

The trained agent is tested in the environment to observe its behavior and effectiveness in improving the student's learning outcomes.

## Visualization

To better understand the agent's learning behavior, I used a visualization tool using `pygame`. This allows us to see the agent's actions and the environment's state in real-time.


## Running the Project

### Prerequisites :
Python 3.7+
Gym library
Keras and Keras-RL

## Installation

To set up the environment and run the project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/DQN-Education.git
    cd DQN-Education
    ```

2. **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Run the test script:**
    ```bash
    python test_env.py
    ```

4. **Run the training script:**
    ```bash
    python train.py
    ```

5. **Run the visualization:**
    ```bash
    python visualize_education_env.py
    ```


## Conclusion

Through this project, I aim to make strides in educational technology by providing tools that cater to the unique needs of students with ADHD, other Neurodivent conditions such as Autism, and Learning disaboilities such as Dyslexia. By leveraging the power of reinforcement learning and AI, I believe it is possible to create impactful educational solutions that drive positive outcomes in both healthcare and education.

---

## Video Demonstration
https://tldv.io/app/meetings/66b206b57701ed0013c399f0/






