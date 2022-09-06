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

    environment = tensorforce.Environment.create(environment=dict(environment='gym', level='CartPole'), max_episode_timesteps=500)
    agent = tensorforce.Agent.create(agent='ppo', environment=environment, batch_size=10,
                                     learning_rate=1e-3, max_episode_timesteps=500)

    # Train for 100 episodes
    for episode in range(100):

        # Episode using act and observe
        states = environment.reset()
        terminal = False
        sum_rewards = 0.0
        num_updates = 0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    g_cnt = 0
    for g_cnt in range(100):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        cnt = 0
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
            print("{}/{}".format(cnt+1, g_cnt+1))
            cnt += 1
    print('Mean evaluation return:', sum_rewards / 100.0)

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()