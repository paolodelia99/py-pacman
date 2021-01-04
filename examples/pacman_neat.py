import argparse
import os
import pickle

import gym
import neat
import numpy as np

from src.controller import Controller
from src.env.agent import Agent
from wrappers import SkipFrame


class GeneticAgent(Agent):
    name = 'genetic_agent'
    net: neat.nn.FeedForwardNetwork

    def __init__(self, layout: str, **kwargs):
        super().__init__(**kwargs)
        self.layout = layout
        self.config_file_name = 'config_' + layout
        self.filename = "winner_" + self.layout + ".pkl"

    def act(self, *args, **kwargs) -> int:
        return int(np.argmax(self.net.activate(kwargs['matrix'].flatten())))

    def load_net(self):
        with open(self.filename, "rb") as f:
            genome = pickle.load(f)

        self.net = neat.nn.FeedForwardNetwork.create(genome, self.load_config())

    def fintness_func(self, genome, config, env):
        state = env.reset(mode='human')
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        tot_rewards = 0

        for _ in range(0, 1000):
            env.render()
            state = state.flatten()
            action = int(np.argmax(net.activate(state)))
            obs, rewards, done, info = env.step(action)

            next_state = info['state matrix']

            if not np.array_equal(next_state, state):
                tot_rewards += rewards + 0.1
            else:
                tot_rewards -= 0.1

            if done:
                tot_rewards -= 5
                break

            if info['win']:
                tot_rewards += 500

            state = next_state

        return tot_rewards

    def eval_genomes(self, genomes, config):
        env = gym.make('pacman-v0', layout=self.layout)
        env = SkipFrame(env, 4)
        idx, genomes = zip(*genomes)

        for genome in genomes:
            genome.fitness = 0

        for genome in genomes:
            fitness = self.fintness_func(genome, config, env)
            genome.fitness = fitness

        env.close()

    def load_config(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_' + self.layout)
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    def train(self, n_generations: int, checkpoints: bool = False, **kwargs):
        config = self.load_config()

        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        if checkpoints:
            p.add_reporter(neat.Checkpointer(generation_interval=100))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        winner = p.run(self.eval_genomes, n_generations)
        pickle.dump(winner, open(self.filename, 'wb'))

        print('\nBest genome:\n{!s}'.format(winner))


def train_agent(layout: str, n_generations: int = 1000, checkpoints: bool = False):
    agent = GeneticAgent(layout)

    agent.train(n_generations, checkpoints)


def run_agent(layout: str):
    agent = GeneticAgent(layout)
    agent.load_net()
    controller = Controller(layout_name=layout, act_sound=True, act_state=False, ai_agent=agent)
    controller.load_menu()


def parse_args():
    parser = argparse.ArgumentParser(description='Argument for the agent that interacts with the sm env')
    parser.add_argument('-lay', '--layout', type=str, nargs=1,
                        help="Name of layout to load in the game")
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the agent')
    parser.add_argument('-g', '--generations', type=int, nargs=1,
                        help="The number of episode to use during training")
    parser.add_argument('-c', '--checkpoints', action='store_true',
                        help='activate checkpoints')
    parser.add_argument('-r', '--run', action='store_true',
                        help='run the trained agent')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    layout = args.layout[0]

    if args.train:
        generations = args.generations[0] if args.generations is not None else 1000
        train_agent(layout, generations, args.checkpoints)

    if args.run:
        run_agent(layout)
