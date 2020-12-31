"""
Q learning with linear function approximator
"""

import argparse
import itertools
import math
from itertools import count
from typing import Union, Tuple, Dict

import gym
import numpy as np

from src.controller import Controller
from src.env.agent import Agent
from src.utils.action import Action
from wrappers import SkipFrame


class LinearApproximator:

    def __init__(self, n_features: int, n_actions: int):
        self._models: Dict[int, np.array] = {i: np.zeros(shape=(n_features,), dtype=float) for i in range(n_actions)}
        self.bias = 1.0
        self.learning_rate = 0.1
        self._n_features = n_features
        self.n_actions = n_actions

    def predict(self, state: Union[list, np.array], action=None):
        if self.n_features != len(state):
            raise Exception("the number of input and the number of weight does not coincide")

        if action is None:
            return np.array([state @ weights + self.bias for weights in self._models.values()])
        else:
            return state @ self._models[action] + self.bias

    def update(self, state: Union[np.array, list], next_state: Union[np.array, list], reward: float, discount: float,
               action: int):
        for weights in list(self._models.values()):
            for i in range(len(weights)):
                weights[i] += self.learning_rate * (
                        reward + discount * np.max(self.predict(next_state)) - self.predict(state, action=action)
                ) * state[i]

    def save(self, path: str):
        with open(path, 'wb') as f:
            np.save(f, np.array([weights for weights in list(self._models.values())]))
            f.close()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, weights: np.array):
        if len(weights[0]) == self.n_features:
            models = {i: weights[i] for i in range(self.n_actions)}
            self._models = models
        else:
            raise Exception("the number of input and the number of weight does not coincide")

    @property
    def n_features(self):
        return self._n_features

    @staticmethod
    def load(models: np.array, n_actions):
        approximator = LinearApproximator(len(models[0]), n_actions)
        approximator.models = models
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
            self.filename = ''.join([self.name, '_', self.layout, '.npy'])
        else:
            self.filename = ''.join([self.name, '_', self.layout, '_', self.version, '.npy'])
        self.approximator = None

    def act(self, **kwargs):
        if self.approximator is None:
            self.load_approximator()

        state: np.array = LinQAgent.get_state(
            player_position=kwargs['player_pos'],
            state_matrix=kwargs['matrix'],
            player_action=kwargs['player_action']
        )

        return np.argmax(self.approximator.predict(state))

    def load_approximator(self):
        with open(self.filename, 'rb') as handle:
            model: np.array = np.load(handle)
            handle.close()

        self.approximator = LinearApproximator.load(model, 4)

    @staticmethod
    def get_state(player_position: Tuple[int, int], state_matrix: np.array, player_action: Action) -> np.array:
        def neighbours_of(i, j) -> list:
            """Positions of neighbours (includes out of bounds but excludes cell itself)."""
            neighbours = list(itertools.product(range(i - 1, i + 2), range(j - 1, j + 2)))
            neighbours.remove((i, j))
            return neighbours

        def two_step_neighbours(i, j, one_step_neighbours: list) -> list:
            neighbours = list(itertools.product(range(i - 2, i + 3), range(j - 2, j + 3)))
            for x, y in one_step_neighbours:
                neighbours.remove((x, y))
            neighbours.remove((i, j))
            return neighbours

        def get_move(action: Action) -> Tuple[int, int]:
            if action == Action.LEFT:
                return -1, 0
            elif action == Action.RIGHT:
                return 1, 0
            elif action == Action.DOWN:
                return 0, 1
            elif action == Action.UP:
                return 1, 0

        def get_min_food_distance(player_pos: Tuple[int, int], matrix) -> float:
            ys, xs = np.where(matrix == 1)
            min_distance = 10000
            for x, y in zip(xs, ys):
                distance = np.sqrt(abs(x - player_pos[0]) ** 2 + (abs(y - player_pos[1]) ** 2))
                if distance < min_distance:
                    min_distance = distance

            return min_distance

        count_ghosts = lambda positions, neighbours: sum(
            [sum([g_x == x and g_y == y for x, y in neighbours]) for g_x, g_y in positions])

        move_x, move_y = get_move(player_action)

        gs_y, gs_x = np.where(state_matrix == -1)
        ghost_positions = [(int(x), int(y)) for x, y in zip(gs_x, gs_y)]
        gs_y, gs_x = np.where(state_matrix == 5)
        vulnerable_ghost_positions = [(int(x), int(y)) for x, y in zip(gs_x, gs_y)]

        one_step_neighbours_pos = neighbours_of(player_position[0], player_position[1])
        two_step_neighbours_pos = two_step_neighbours(player_position[0], player_position[1], one_step_neighbours_pos)

        n_ghost_one_step = count_ghosts(ghost_positions, one_step_neighbours_pos)
        n_ghost_two_step = count_ghosts(ghost_positions, two_step_neighbours_pos)
        n_vulnerable_ghosts_one_step = count_ghosts(vulnerable_ghost_positions, one_step_neighbours_pos)
        n_vulnerable_ghosts_two_step = count_ghosts(vulnerable_ghost_positions, two_step_neighbours_pos)
        eats_food = int(state_matrix[player_position[1] + move_y][player_position[0] + move_x] == 1)
        min_food_distance = get_min_food_distance(player_position, state_matrix)

        return np.array([
                n_ghost_one_step,
                n_ghost_two_step,
                n_vulnerable_ghosts_one_step,
                n_vulnerable_ghosts_two_step,
                eats_food,
                min_food_distance
            ], dtype=int)

    def train(self, **kwargs):
        n_episodes = 10000
        discount = 0.99
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay = 1e7
        env = gym.make('pacman-v0', layout=self.layout)
        env = SkipFrame(env, skip=5)
        approximator = LinearApproximator(6, env.action_space.n)

        epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon - epsilon_min) * math.exp(
            -1. * frame_idx / epsilon_decay)

        for episode in range(n_episodes):
            info = env.reset(mode='info')
            state = LinQAgent.get_state(info['player position'], info['state matrix'], info['player action'])
            total_rewards = 0

            epsilon = epsilon_by_frame(episode)

            for i in count():
                env.render()
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(approximator.predict(state)))

                obs, rewards, done, info = env.step(action)
                next_state = LinQAgent.get_state(info['player position'], info['state matrix'], info['player action'])

                if not np.array_equal(next_state, state):
                    approximator.update(state, next_state, rewards, discount, action)

                state = next_state
                total_rewards += rewards

                if done:
                    print(f'{episode} episode finished after {i} timesteps')
                    print(f'Total rewards: {total_rewards}')
                    print(f'win: {info["win"]}')
                    print(f'epsilon {epsilon}')
                    break

            if episode % 1000 == 0:
                approximator.save(self.filename)

        env.close()

        approximator.save(self.filename)


def train_agent(layout: str):
    agent = LinQAgent(layout=layout, version='1')

    agent.train()


def run_agent(layout: str):
    agent = LinQAgent(layout=layout, version='1')
    agent.load_approximator()
    controller = Controller(layout_name=layout, act_sound=True, act_state=False, ai_agent=agent)
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
