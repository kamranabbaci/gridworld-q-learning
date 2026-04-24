import numpy as np
import random
import matplotlib.pyplot as plt

# Grid size
GRID_SIZE = 5

# Actions: North, South, East, West
ACTIONS = {
    0: (-1, 0),  # North
    1: (1, 0),   # South
    2: (0, 1),   # East
    3: (0, -1)   # West
}

# Parameters
ALPHA = 1.0          # Learning rate
GAMMA = 0.9          # Discount factor
EPSILON = 0.1        # Exploration rate
EPISODES = 100

# Start and Goal
START = (1, 0)  # [2,1] in assignment (0-indexed)
GOAL = (4, 4)

# Special jump
JUMP_FROM = (1, 3)
JUMP_TO = (3, 3)

# Obstacles
OBSTACLES = [(2, 2), (3, 1)]  # example obstacles

# Initialize Q-table
Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))


def is_valid(state):
    r, c = state
    if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
        return False
    if state in OBSTACLES:
        return False
    return True


def step(state, action):
    # Special jump
    if state == JUMP_FROM:
        return JUMP_TO, 5

    dr, dc = ACTIONS[action]
    new_state = (state[0] + dr, state[1] + dc)

    if not is_valid(new_state):
        return state, -1

    if new_state == GOAL:
        return new_state, 10

    return new_state, -1


def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(list(ACTIONS.keys()))
    else:
        r, c = state
        return np.argmax(Q[r, c])


def train():
    rewards_per_episode = []

    for episode in range(EPISODES):
        state = START
        total_reward = 0

        while True:
            action = choose_action(state)
            next_state, reward = step(state, action)

            r, c = state
            nr, nc = next_state

            # Q-learning update
            Q[r, c, action] = Q[r, c, action] + ALPHA * (
                reward + GAMMA * np.max(Q[nr, nc]) - Q[r, c, action]
            )

            state = next_state
            total_reward += reward

            if state == GOAL:
                break

        rewards_per_episode.append(total_reward)

    return rewards_per_episode


def get_value_function():
    V = np.max(Q, axis=2)
    return V


def plot_values(V):
    plt.imshow(V, cmap='cool')
    plt.colorbar()
    plt.title("State Value Function")

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            plt.text(j, i, round(V[i, j], 1),
                     ha='center', va='center', color='black')

    plt.show()


if __name__ == "__main__":
    rewards = train()
    V = get_value_function()

    print("Q-table:\n", Q)
    print("Value Function:\n", V)

    plot_values(V)