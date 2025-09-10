import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class GridWorldEnv(gym.Env):
    """Custom Grid World Environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, size=16, dynamic_obstacles=False):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.goal_position = (self.size - 1, self.size - 1)
        self.start_position = (0, 0)
        self.state = self.start_position
        self.dynamic_obstacles = dynamic_obstacles
        self.grid = self.initialize_grid()
        self.obstacles = set()

    def initialize_grid(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        # Add static obstacles
        for i in range(3, 6):
            for j in range(3, 6):
                grid[i, j] = -1  # Mark obstacles
                self.obstacles.add((i, j))
        return grid

    def update_dynamic_obstacles(self):
        if not self.dynamic_obstacles:
            return
        # Move obstacles randomly while ensuring they don't overlap or leave the grid
        new_obstacles = set()
        for (x, y) in self.obstacles:
            self.grid[x, y] = 0
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size and (new_x, new_y) != self.goal_position:
                new_obstacles.add((new_x, new_y))
            else:
                new_obstacles.add((x, y))
        self.obstacles = new_obstacles
        for (x, y) in self.obstacles:
            self.grid[x, y] = -1

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
        next_position = (self.state[0] + moves[action][0], self.state[1] + moves[action][1])

        if 0 <= next_position[0] < self.size and 0 <= next_position[1] < self.size and self.grid[next_position] != -1:
            self.state = next_position

        reward = -1
        done = False
        if self.state == self.goal_position:
            reward += 100
            done = True

        self.update_dynamic_obstacles()
        return self.state_to_index(self.state), reward, done, {}

    def reset(self):
        self.state = self.start_position
        self.grid = self.initialize_grid()
        return self.state_to_index(self.state)

    def render(self, mode='human'):
        grid_display = np.array(self.grid, dtype=str)
        grid_display[grid_display == '0'] = ' '  # Free space
        grid_display[grid_display == '-1'] = 'X'  # Obstacles
        grid_display[self.goal_position[0]][self.goal_position[1]] = 'G'  # Goal
        grid_display[self.state[0]][self.state[1]] = 'A'  # Agent
        print("\n".join(" ".join(row) for row in grid_display))
        print("\n")

    def state_to_index(self, state):
        return state[0] * self.size + state[1]

    def index_to_state(self, index):
        return divmod(index, self.size)


def train_q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode}: Training Progress...")

    return q_table


def train_sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(episodes):
        state = env.reset()
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])

            old_value = q_table[state, action]
            next_value = q_table[next_state, next_action]
            q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)
            state, action = next_state, next_action

    return q_table


def train_double_q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table_1 = np.zeros((env.observation_space.n, env.action_space.n))
    q_table_2 = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table_1[state] + q_table_2[state])

            next_state, reward, done, _ = env.step(action)
            if random.random() < 0.5:
                max_action = np.argmax(q_table_1[next_state])
                q_table_1[state, action] += alpha * (reward + gamma * q_table_2[next_state, max_action] - q_table_1[state, action])
            else:
                max_action = np.argmax(q_table_2[next_state])
                q_table_2[state, action] += alpha * (reward + gamma * q_table_1[next_state, max_action] - q_table_2[state, action])

            state = next_state

    return q_table_1 + q_table_2


def evaluate_agent(env, q_table, episodes=100):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    print(f"Average Reward: {np.mean(rewards)}")


def visualize_policy(env, q_table):
    policy_grid = np.full((env.size, env.size), ' ')
    for state_index in range(env.observation_space.n):
        x, y = env.index_to_state(state_index)
        action = np.argmax(q_table[state_index])
        if (x, y) == env.goal_position:
            policy_grid[x, y] = 'G'
        elif (x, y) in env.obstacles:
            policy_grid[x, y] = 'X'
        else:
            policy_grid[x, y] = ['↑', '↓', '←', '→'][action]

    print("\nLearned Policy:")
    for row in policy_grid:
        print(" ".join(row))


def main():
    env = GridWorldEnv(dynamic_obstacles=True)
    q_table = train_q_learning(env, episodes=1000)
    evaluate_agent(env, q_table)
    visualize_policy(env, q_table)


if __name__ == "__main__":
    main()
