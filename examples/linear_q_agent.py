"""
Q learning with linear function approximator
"""

import argparse
import pickle
import random
from typing import Union

import gym
import numpy as np

from src.controller import Controller
from src.env.agent import Agent
from wrappers import SkipFrame

import itertools


class LinearApproximator:

    def __init__(self, n_features: int):
        self.weights = np.zeros(shape=(n_features,))
        self.bias = 1.0
        self.learning_rate = 0.1

    def forward(self, inputs: list):
        if len(self.weights) != len(inputs):
            raise Exception("the number of input and the number of weight does not coincide")

        inputs = np.array(inputs)

        return inputs @ self.weights + self.bias

    def update(self, state: list, next_state: list, reward: float, discount: float):
        for i in range(len(self.weights)):
            self.weights += self.learning_rate * (
                    reward + discount * self.forward(next_state) - self.forward(state)
            ) * state[i]

    def save(self, path: str):
        w_dict = dict(weights=self.weights)

        with open(path, 'wb') as handle:
            pickle.dump(w_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    @staticmethod
    def load(weights: Union[np.array, list]):
        approximator = LinearApproximator(len(weights))
        approximator.weights = weights
        approximator.bias = 1.0

        return approximator


class LinQAgent(Agent):
    name = 'lin_q_agent'
    approximator: LinearApproximator

    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs['layout']
        self.version = kwargs['version']
        if self.version is None:
            self.filename = ''.join([self.name, '_', self.layout, '.pkl'])
        else:
            self.filename = ''.join([self.name, '_', self.layout, '_', self.version, '.pkl'])
        self.approximator = None

    def act(self, **kwargs):
        if self.approximator is None:
            self.load_approximator()

        state = LinQAgent.get_state(kwargs['player_pos'], kwargs['matrix'])

        return self.approximator.forward(state)

    def load_approximator(self):
        with open(self.filename, 'rb') as handle:
            weights = pickle.load(handle)
            handle.close()

        self.approximator = LinearApproximator.load(weights)

    @staticmethod
    def get_state(player_position, state_matrix):
        def neighbours_of(i, j):
            """Positions of neighbours (includes out of bounds but excludes cell itself)."""
            neighbours = list(itertools.product(range(i - 1, i + 2), range(j - 1, j + 2)))
            neighbours.remove((i, j))
            return neighbours

        g_y, g_x = np.where(state_matrix == -1)
        g_y, g_x = int(g_y[0]), int(g_x[0])

        neighbours_pos = neighbours_of(player_position[0], player_position[1])

        is_ghost_close: bool = sum([g_x == x and g_y == y for x, y in neighbours_pos]) >= 1


        return player_position[0], player_position[1], g_x, g_y

    def train(self, **kwargs):
        n_episodes = 10000
        discount = 0.99
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay_rate = 0.9999999
        max_steps = 60
        env = gym.make('pacman-v0', layout=self.layout)
        env = SkipFrame(env, skip=5)
        approximator = LinearApproximator(7)

        for episode in range(n_episodes):
            state = env.reset()
            total_rewards = 0

            for i in range(max_steps):
                env.render()
                if random.uniform(0, 1) > epsilon:
                    action = int(approximator.forward(state))
                else:
                    action = env.action_space.sample()

                obs, rewards, done, info = env.step(action)
                next_state = LinQAgent.get_state(info['player position'], info['state matrix'])

                if next_state != state:
                    approximator.update(state, next_state, rewards, discount)

                state = next_state
                total_rewards += rewards

                if done:
                    print(f'{episode} episode finished after {i} timesteps')
                    print(f'Total rewards: {total_rewards}')
                    print(f'win: {info["win"]}')
                    break

                if epsilon >= epsilon_min:
                    epsilon *= epsilon_decay_rate

        env.close()

        approximator.save(self.filename)


def train_agent(layout: str):
    agent = LinQAgent(layout=layout, version='1')

    agent.train()


def run_agent(layout: str):
    agent = LinQAgent(layout=layout, version='1')
    agent.load_approximator()
    controller = Controller(layout_name=layout, act_sound=True, act_state=True, ai_agent=agent)
    controller.load_menu()


def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the agent that interacts with the sm env')
    parser.add_argument('-lay', '--layout', type=str, nargs=1,
                        help="Name of layout to load in the game")
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the agent')
    parser.add_argument('-r', '--run', action='store_true',
                        help='run the trained agent')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    layout = args.layout[0]

    if args.train:
        train_agent(layout)

    if args.run:
        run_agent(layout)
