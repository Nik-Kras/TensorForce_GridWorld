import tensorforce
import Environment
import Agent
import numpy as np
import matplotlib.pyplot as plt
from MapGenerator.Grid import *

import tensorflow as tf

SIZE = 4
ROWS = 12
COLS = 12

game = Environment.GridWorld(tot_row = ROWS, tot_col = COLS)

#Define the state matrix
# Generator = Grid(SIZE)
# state_matrix = Generator.GenerateMap()
# game.setStateMatrix(state_matrix)
# game.setPosition()
game.reset()
game.render()

print(game.getWorldState())

def main():
    print("Hi, Nikita")

    environment = game # tensorforce.Environment.create(environment=dict(environment='gym', level='CartPole'), max_episode_timesteps=500)
    # agent = tensorforce.Agent.create(agent='ppo', environment=environment, batch_size=10,
    #                                  learning_rate=1e-3, max_episode_timesteps=500)

    # agent = tensorforce.Agent.create(
    #     agent='ppo', environment=environment, max_episode_timesteps=30,
    #     # Automatically configured network
    #     network='auto',
    #     # Optimization
    #     batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
    #     # Reward estimation
    #     likelihood_ratio_clipping=0.2, discount=0.99,
    #     # Exploration
    #     exploration=0.0, variable_noise=0.0,
    #     # Regularization
    #     l2_regularization=0.0, entropy_regularization=0.0,
    #     # TensorFlow etc
    #     parallel_interactions=5
    # )

    agent = tensorforce.Agent.create(
        agent='tensorforce', environment=environment, update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient',

        reward_estimation=dict(horizon=15, discount=0.99, predict_terminal_values = True),    #

        policy = dict(
            network = dict(
                type="auto",
                # rnn=15,         # Set the Horizon for LSTM
                size = 128,
                depth = 4
            )
        )

        # exploration = dict(
        #         type='exponential', unit='episodes', num_steps=1000,
        #         initial_value=0.99, decay_rate=0.5
        #     )

        #     (
        #     type='linear', unit='episodes', num_steps=1000,
        #     initial_value=10, final_value=50
        # )

    )

    print(agent.get_architecture())

    # Train for 10,000 episodes
    num_train_episodes = 10000
    tracker = {
        "rewards": [0],
        "picked_goal": [0],
        "window": 50,
        "cnt": 0,       # Keep track on counting "window" elements per one array element
        "array_cnt": 0  # Keep track on array indexes
    }
    tracker["rewards"]     = [0] * np.math.ceil((int(num_train_episodes / tracker["window"])))
    tracker["picked_goal"] = [0] * np.math.ceil((int(num_train_episodes / tracker["window"])))

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

        tracker["rewards"][tracker["array_cnt"]] +=  sum_rewards
        if sum_rewards > 0: tracker["picked_goal"][tracker["array_cnt"]] += 1

        # Each "window" iterations count average and go to next array element
        if tracker["cnt"] == tracker["window"]:
            current_index = tracker["array_cnt"]
            tracker["rewards"][current_index]     = tracker["rewards"][current_index] / tracker["window"]
            tracker["picked_goal"][current_index] = tracker["picked_goal"][current_index] / tracker["window"]
            tracker["array_cnt"] += 1
            tracker["cnt"] = 0
        else:
            tracker["cnt"] += 1
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

    plt.plot(tracker["rewards"][:tracker["array_cnt"]-2])
    plt.show()
    plt.plot(tracker["picked_goal"][:tracker["array_cnt"]-2])
    plt.show()

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