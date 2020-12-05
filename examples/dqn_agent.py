"""
An example of how to make the dnq ai agent work
"""
import argparse
import pickle
from collections import deque
from functools import partial
from typing import Sequence

import gym
import jax
from flax import nn, optim, serialization
from jax import numpy as jnp, random, jit, vmap

from src.controller import Controller
from src.env.agent import Agent
import random

def rand(key, num_actions):
    return jax.random.randint(key, (1,), 0, num_actions)[0]


@jit
def policy(model, key, x, epsilon, num_actions):
    prob = jax.random.uniform(key)
    q = model(x)
    rnd = partial(rand, num_actions=num_actions)
    a = jax.lax.cond(prob < epsilon, key, rnd, q, jnp.argmax)
    return a


@vmap
def q_learning_loss(q, target_q, action, action_select, reward, done, gamma=0.9):
    td_target = reward + gamma * (1. - done) * target_q[action_select]
    td_error = jax.lax.stop_gradient(td_target) - q[action]
    return td_error ** 2


@jit
def train_step(optimizer, target_model, batch):
    def loss_fn(model):
        q = model(batch['state'])
        target_q = target_model(batch['next_state'])
        action_select = model(batch['next_state']).argmax(-1)
        return jnp.mean(q_learning_loss(q, target_q, batch['action'], action_select,
                                        batch['reward'], batch['done']))

    loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = jnp.expand_dims(state, 0)
        next_state = jnp.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return {
            'state': jnp.concatenate(state),
            'action': jnp.asarray(action),
            'reward': jnp.asarray(reward),
            'next_state': jnp.concatenate(next_state),
            'done': jnp.asarray(done)
        }

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """DQN network."""
    features: Sequence[int]

    def apply(self, x, num_actions):
        x = nn.Dense(x, features=133)
        x = nn.relu(x)
        x = nn.Dense(x, features=64)
        x = nn.relu(x)
        x = nn.Dense(x, features=num_actions)
        return x


class DNQAgent(Agent):
    model: nn.Model
    name = 'dnq_agent'

    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs['layout']
        self.version = kwargs['version']
        if self.version is None:
            self.filename = ''.join([self.name, '_', self.layout, '.pkl'])
        else:
            self.filename = ''.join([self.name, '_', self.layout, '_', self.version, '.pkl'])

    def act(self, state, **kwargs):
        return jnp.argmax(self.model(kwargs['matrix'].flatten()))

    def save_model(self, model):
        params = serialization.to_state_dict(model)['target']
        with open(self.filename, 'wb') as handle:
            pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    def load_model(self, num_actions):
        with open(self.filename, 'rb') as handle:
            params = pickle.load(handle)
            handle.close()

        self.model = nn.Model(DQN.partial(num_actions=num_actions), params['params'])

    def train(self,
              n_episodes,
              num_steps,
              batch_size,
              replay_size,
              target_update_frequency,
              gamma=0.9,
              frame_to_skip: int = 10,
              **kwargs):
        env = gym.make('pacman-v0', layout=self.layout, frame_to_skip=frame_to_skip, enable_render=False)
        replay_buffer = ReplayBuffer(replay_size)
        key = jax.random.PRNGKey(0)
        num_actions = env.action_space.n
        state = env.reset()
        module = DQN.partial(num_actions=num_actions)
        _, initial_params = module.init(key, state.flatten())
        model = nn.Model(module, initial_params)
        target_model = nn.Model(module, initial_params)
        optimizer = optim.Adam(1e-3).create(model)
        epsilon = 1.0
        epsilon_final = 0.05
        epsilon_decay = 500

        epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon - epsilon_final) * jnp.exp(
            -1. * frame_idx / epsilon_decay)

        # stats
        ep_losses = []
        ep_returns = []

        for episode in range(n_episodes):
            state = env.reset().flatten()
            epsilon = epsilon_by_frame(episode)
            ep_return = 0.
            loss = 0

            for t in range(num_steps):
                key, _ = jax.random.split(key)
                env.render()
                action = policy(optimizer.target, key, state, epsilon, num_actions)

                next_state, reward, done, _ = env.step(int(action))
                next_state = next_state.flatten()

                replay_buffer.push(next_state, action, reward, state, done)
                ep_return += reward

                if len(replay_buffer) > batch_size:
                    batch = replay_buffer.sample(batch_size)
                    optimizer, loss = train_step(optimizer, target_model, batch)
                    ep_losses.append(float(loss))

                if t % target_update_frequency == 0:
                    target_model = target_model.replace(params=optimizer.target.params)

                if done:
                    break

                state = next_state

            ep_returns.append(ep_return)

            print("Episode #{}, Return {}, Loss {}".format(episode, ep_return, loss))

        env.close()

        self.save_model(optimizer.state_dict())


def train_agent(layout: str, episodes: int = 1000, **kwargs):
    agent = DNQAgent(layout=layout, version='1')

    agent.train(
        n_episodes=episodes,
        num_steps=100000,
        batch_size=32,
        replay_size=100,
        target_update_frequency=10,
        frame_to_skip=kwargs['frames_to_skip']
    )


def run_agent(layout: str):
    agent = DNQAgent(layout=layout, version='1')
    agent.load_model(4)
    controller = Controller(layout_name=layout, act_sound=False, act_state=True, ai_agent=agent)
    controller.load_menu()


def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the agent that interacts with the sm env')
    parser.add_argument('-lay', '--layout', type=str, nargs=1,
                        help="Name of layout to load in the game")
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the agent')
    parser.add_argument('-e', '--episodes', type=int, nargs=1,
                        help="The number of episode to use during training")
    parser.add_argument('-frs', '--frames_to_skip', type=int, nargs=1,
                        help="The number of frames to skip during training, so the agent doesn't have to take "
                             "an action a every frame")
    parser.add_argument('-r', '--run', action='store_true',
                        help='run the trained agent')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    layout = args.layout[0]
    episodes = args.episodes[0] if args.episodes else 1000

    if args.train:
        frames_to_skip = args.frames_to_skip if args.frames_to_skip is not None else 10
        train_agent(layout, episodes, frames_to_skip=args.frames_to_skip)

    if args.run:
        run_agent(layout)
