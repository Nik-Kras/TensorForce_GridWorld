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
state_matrix = Generator.GenerateMap()
game.setStateMatrix(state_matrix)
game.setPosition()
game.render()

print(game.state_matrix)

def main():
    print("Hi, Nikita")

    environment = game # tensorforce.Environment.create(environment=dict(environment='gym', level='CartPole'), max_episode_timesteps=500)
    # agent = tensorforce.Agent.create(agent='ppo', environment=environment, batch_size=10,
    #                                  learning_rate=1e-3, max_episode_timesteps=500)

    agent = tensorforce.Agent.create(
        agent='ppo', environment=environment, max_episode_timesteps = 90,
        # Automatically configured network
        network='auto',
        # Optimization
        batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99,
        # Critic
        # baseline_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        # name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        # summarizer=None, recorder=None
    )

    # Train for 50,000 episodes
    num_train_episodes = 50000
    for episode in range(num_train_episodes):

        # Episode using act and observe
        states = environment.reset()
        terminal = False
        sum_rewards = 0.0
        num_updates = 0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(action=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

    # Evaluate for 500 episodes
    sum_rewards = 0.0
    g_cnt = 0
    num_eval_episodes = 500
    for g_cnt in range(num_eval_episodes):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        cnt = 0
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(action=actions)
            sum_rewards += reward
            print("{}/{}".format(cnt+1, g_cnt+1))
            cnt += 1
    print('Mean evaluation return:', sum_rewards / num_eval_episodes)

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()