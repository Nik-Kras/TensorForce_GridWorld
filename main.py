import tensorforce
import Environment
import Agent
import numpy as np
from MapGenerator.Grid import *

from keras.layers import *
from keras.models import *

import tensorflow as tf

SIZE = 4
ROWS = 12
COLS = 12

game = Environment.GridWorld(tot_row = ROWS, tot_col = COLS)

#Define the state matrix
Generator = Grid(SIZE)
state_matrix = Generator.GenerateMap() - 1
game.setStateMatrix(state_matrix)
game.setPosition()
game.render()

def main():
    print("Hi, Nikita")

    # Pre-defined or custom environment
    environment = tensorforce.Environment.create(environment='gym', level='CartPole-v1')

    # Instantiate a Tensorforce agent
    agent = tensorforce.Agent.create(
        agent='tensorforce',
        environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
        memory=10000,
        update=dict(unit='timesteps', batch_size=64),
        optimizer=dict(type='adam', learning_rate=3e-4),
        policy=dict(network='auto'),
        objective='policy_gradient',
        reward_estimation=dict(horizon=20)
    )

    # Train for 300 episodes
    for _ in range(300):

        # Initialize episode
        states = environment.reset()
        terminal = False

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

    agent.close()
    environment.close()


if __name__ == '__main__':
    main()