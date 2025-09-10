import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

# Import necessary deep learning libraries if using Deep Q-Learning
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten
# from tensorflow.keras.optimizers import Adam

class GridWorldEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.size = 16
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.goal_position = (self.size - 1, self.size - 1)
        self.start_position = (0, 0)
        self.state = self.start_position
        self.grid = self.initialize_grid()
        
    def initialize_grid(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        # Example: Add obstacles
        for i in range(3, 6):
            for j in range(3, 6):
                grid[i, j] = -1  # Marking obstacles in the grid
        # You can also add dynamic or moving obstacles here in future extensions
        return grid

    def step(self, action):
        # Map action to movement
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
        next_position = (self.state[0] + moves[action][0], self.state[1] + moves[action][1])

        # Check boundaries and obstacles
        if 0 <= next_position[0] < self.size and 0 <= next_position[1] < self.size and self.grid[next_position] != -1:
            self.state = next_position

        # Calculate reward
        reward = -1  # Penalty for each move
        if self.state == self.goal_position:
            reward += 100  # Reward for reaching the goal
            done = True
        else:
            done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = self.start_position
        self.grid = self.initialize_grid()  # Reinitialize or keep the grid as is depending on your design
        return self.state

    def render(self, mode='human'):
        grid_display = np.array(self.grid, dtype=str)
        grid_display[grid_display == '0'] = ' '  # Free space
        grid_display[grid_display == '-1'] = 'X'  # Obstacles
        grid_display[self.goal_position[0]][self.goal_position[1]] = 'G'  # Goal
        grid_display[self.state[0]][self.state[1]] = 'A'  # Agent
        print("\n".join(" ".join(row) for row in grid_display))
        print("\n")

def state_to_index(self, state):
    """Converts a tuple state (x, y) into a single integer index."""
    x, y = state
    return x * self.size + y

def index_to_state(self, index):
    """Converts a single integer index back to tuple state (x, y)."""
    return (index // self.size, index % self.size)

# def train_q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
#     # Initialize the Q-table
#     q_table = np.zeros((env.observation_space.n, env.action_space.n))

#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         total_reward = 0

#         while not done:
#             # Epsilon-greedy strategy for exploration-exploitation trade-off
#             if random.uniform(0, 1) < epsilon:
#                 action = env.action_space.sample()  # Explore: select a random action
#             else:
#                 action = np.argmax(q_table[state])  # Exploit: select the action with max value (greedy)

#             # Take action and observe the outcome
#             next_state, reward, done, _ = env.step(action)
#             total_reward += reward

#             print(f"State: {state}, Action: {action}, Next State: {next_state}")
#             if next_state >= q_table.shape[0]:
#                 print(f"Warning: next_state {next_state} is out of bounds.")

#             # Q-table update
#             old_value = q_table[state, action]
#             next_max = np.max(q_table[next_state])
            
#             # Bellman Equation - Update the Q-value for the current state and action pair
#             new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#             q_table[state, action] = new_value
            
#             # Move to the next state
#             state = next_state

#         # Decaying epsilon if needed
#         epsilon *= 0.99

#         if (episode + 1) % 100 == 0:
#             print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

#     return q_table
def train_q_learning(env, episodes):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episodes):
        state = env.reset()
        state_index = env.state_to_index(state)  # Convert state to index

        done = False
        while not done:
            # Epsilon-greedy action selection
            action = np.argmax(q_table[state_index]) if np.random.rand() > epsilon else env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            next_state_index = env.state_to_index(next_state)  # Convert next state to index

            # Update Q-table
            old_value = q_table[state_index, action]
            next_max = np.max(q_table[next_state_index])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state_index, action] = new_value

            state_index = next_state_index  # Move to the next state

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode: {episode + 1}, Reward: {reward}")

    return q_table

import matplotlib.pyplot as plt
import seaborn as sns

def plot_q_values(q_table):
    # Reshape Q-table to a 2D array for heatmap if necessary
    q_values_max = np.max(q_table, axis=1).reshape((env.size, env.size))
    plt.figure(figsize=(10, 10))
    sns.heatmap(q_values_max, annot=True, cmap="coolwarm", cbar=False)
    plt.title("Heatmap of Q-values")
    plt.show()

def visualize_trajectory(env, q_table):
    env.reset()
    done = False
    plt.figure(figsize=(10, 10))
    grid = np.zeros((env.size, env.size))
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) in env.obstacles:
                grid[i, j] = -1
            else:
                grid[i, j] = 0

    state = env.state_to_index(env.start_position)
    grid[env.start_position[0]][env.start_position[1]] = 0.5  # Start

    while not done:
        action = np.argmax(q_table[state])  # Choose the best action
        next_state, _, done, _ = env.step(action)
        state = next_state
        pos = env.index_to_state(state)
        grid[pos[0]][pos[1]] = 0.5  # Path

    grid[env.goal_position[0]][env.goal_position[1]] = 0.9  # Goal
    plt.imshow(grid, cmap="viridis")
    plt.colorbar()
    plt.title("Agent's Trajectory")
    plt.show()

def evaluate_agent(env, q_table, episodes=100):
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(q_table[state])  # Always choose the best action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}")
    plt.plot(total_rewards)
    plt.title('Performance Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def main():
    env = GridWorldEnv()
    q_table = train_q_learning(env, 1000)
    plot_q_values(q_table)
    visualize_trajectory(env, q_table)
    evaluate_agent(env, q_table, 100)

if __name__ == "__main__":
    main()




# import numpy as np
# import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from collections import deque
# import gym

# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()

#     def _build_model(self):
#         # Neural Net for Deep-Q learning Model
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(self.learning_rate))
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])  # returns action

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

# def train_deep_q_learning(env, agent, episodes=1000):
#     for e in range(episodes):
#         state = env.reset()
#         state = np.reshape(state, [1, env.state_size])
#         for time in range(500):  # 500 timesteps per episode
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             reward = reward if not done else -10
#             next_state = np.reshape(next_state, [1, env.state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, time, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)

# import matplotlib.pyplot as plt

# def plot_training_details(agent):
#     plt.plot(agent.history['loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train'], loc='upper left')
#     plt.show()

#     plt.plot(agent.history['epsilon'])
#     plt.title('Epsilon decay')
#     plt.ylabel('Epsilon')
#     plt.xlabel('Episode')
#     plt.legend(['Decay'], loc='upper right')
#     plt.show()

# def evaluate_model(env, agent, episodes=100):
#     total_rewards = []
#     for e in range(episodes):
#         state = env.reset()
#         state = np.reshape(state, [1, env.state_size])
#         total_reward = 0
#         while True:
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             total_reward += reward
#             next_state = np.reshape(next_state, [1, env.state_size])
#             state = next_state
#             if done:
#                 total_rewards.append(total_reward)
#                 break
#     return total_rewards

# def main():
#     env = GridWorldEnv()  # Assume this is already defined as above
#     state_size = env.observation_space.n
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)
#     train_deep_q_learning(env, agent, episodes=1000)
#     plot_training_details(agent)
#     scores = evaluate_model(env, agent)
#     print("Average score over test episodes:", np.mean(scores))

# if __name__ == "__main__":
#     main()



def train_sarsa(env, episodes):
    # Implement SARSA training logic
    pass

def train_double_q_learning(env, episodes):
    # Setup and train using Double Q-Learning
    pass

def epsilon_greedy(Q, state, epsilon):
    # Exploration strategy
    pass

