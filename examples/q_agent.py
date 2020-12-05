"""
An example of how to make the ai agent work
"""

import argparse
import pickle
import random
from collections import defaultdict

import numpy as np

from src.controller import Controller
from src.env.agent import Agent
import gym


class QAgent(Agent):
    name = 'q_agent'

    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs['layout']
        self.version = kwargs['version']
        if self.version is None:
            self.filename = ''.join([self.name, '_', self.layout, '.pkl'])
        else:
            self.filename = ''.join([self.name, '_', self.layout, '_', self.version, '.pkl'])
        self.q_table = None

    def act(self, state, **kwargs):
        if self.q_table is None:
            self.load_q_table()

        state = QAgent.get_state(state, kwargs['matrix'])

        return np.argmax(self.q_table[state])

    def load_q_table(self):
        with open(self.filename, 'rb') as handle:
            self.q_table = pickle.load(handle)
            handle.close()

    def __del__(self):
        del self.q_table

    @staticmethod
    def get_state(player_position, state_matrix):
        g_y, g_x = np.where(state_matrix == - 15)
        if len(g_x) != 0 and len(g_y) != 0:
            g_y, g_x = int(g_y[0]), int(g_x[0])
        else:
            g_y, g_x = np.where(state_matrix == 5)
            g_y, g_x = int(g_y[0]), int(g_x[0])

        return player_position[0], player_position[1], g_x, g_y

    def train(self, **kwargs):
        n_episodes = 5000
        discount = 0.99
        alpha = 0.6  # learning rate
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay_rate = 0.9999999
        max_steps = 60
        env = gym.make('pacman-v0', layout=self.layout, frame_to_skip=10)
        q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        state = QAgent.get_state(env.game.maze.get_player_home(), env.get_state_matrix())

        for episode in range(n_episodes):
            env.reset()
            total_rewards = 0

            for i in range(max_steps):
                env.render()
                if random.uniform(0, 1) > epsilon:
                    action = int(np.argmax(q_table[state]))
                else:
                    action = env.action_space.sample()

                obs, rewards, done, info = env.step(action)
                next_state = QAgent.get_state(info['player position'], obs)

                if next_state != state:
                    rewards = rewards + 2 if rewards > 0 else rewards
                    q_table[state][action] += alpha * (
                            rewards + discount * np.max(q_table[next_state]) - q_table[state][action])

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

        with open(self.filename, 'wb') as handle:
            pickle.dump(dict(q_table), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()


def train_agent(layout: str):
    agent = QAgent(layout=layout, version='1')

    agent.train()


def run_agent(layout: str):
    agent = QAgent(layout=layout, version='1')
    agent.load_q_table()
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
