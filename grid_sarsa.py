from grid import Grid
import numpy as np
import random

# Create environment
env = Grid()

episodes = 1000
max_steps = 1000
gamma = 0.9
lr = 0.1
decay = 0.8
epsilon = 1
epsilon_decay_rate = 0.003

#Initializing the Q-matrix
Q = np.zeros((len(env.stateSpace), len(env.actionSpace))) 

for ep in range(episodes):

    env.reset()
    state1 = env.currentState

    mat = np.zeros((len(env.stateSpace), len(env.actionSpace)))

    if random.uniform(0, 1) < epsilon:
        action1 = random.choice(env.actionSpace)
    else:
        action1 = np.argmax(Q[env.currentState])
        

    for ms in range(max_steps):

        state2, reward, done = env.step(action1)

        if random.uniform(0, 1) < epsilon:
            action2 = random.choice(env.actionSpace)
        else:
            action2 = np.argmax(Q[env.currentState])

        td_error = reward + gamma * Q[state2, action2] - Q[state1, action1]

        mat[state1, action1] += 1

        for s in env.stateSpace:
            for a in env.actionSpace:
                Q[s, a] += lr * td_error * mat[s, a]
                mat[s, a] = gamma * decay * mat[s, a]

        state1 = state2
        action1 = action2

        if done:
            break

        epsilon = np.exp(-epsilon_decay_rate * ep)

print('Action-Value function:')

print ("Last_State : ", env.currentState)
print(Q)

env.startGrid(Q)