"""
An example of how to make the dnq ai agent work
"""

import argparse
import pickle
import random

import numpy as np

from src.controller import Controller
from src.env.agent import Agent
from src.env.pacman_env import PacmanEnv


class DNQAgent(Agent):
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
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train(self, q_func, optimizer_spec, batch_size,  **kwargs):

        env = PacmanEnv(
            layout=self.layout,
            frame_to_skip=10,
            enable_render=True
        )
        gamma = 0.99
        learning_starts = 50000
        learning_freq = 4
        target_update_freq = 10000
        stopping_criterion = None
        num_actions = env.action_space.n
        input_arg = env.observation_space.shape[0]


        # Initialize target q function and q function
        Q = q_func(input_arg, num_actions).type(dtype)
        target_Q = q_func(input_arg, num_actions).type(dtype)

        # Construct Q network optimizer function
        optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

        # Construct the replay buffer
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

        ###############
        # RUN ENV     #
        ###############
        num_param_updates = 0
        mean_episode_reward = -float('nan')
        best_mean_episode_reward = -float('inf')
        last_obs = env.reset()
        LOG_EVERY_N_STEPS = 10000

        for t in count():
            ### Check stopping criterion
            if stopping_criterion is not None and stopping_criterion(env):
                break

            ### Step the env and store the transition
            # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
            last_idx = replay_buffer.store_frame(last_obs)
            # encode_recent_observation will take the latest observation
            # that you pushed into the buffer and compute the corresponding
            # input that should be given to a Q network by appending some
            # previous frames.
            recent_observations = replay_buffer.encode_recent_observation()

            # Choose random action if not yet start learning
            if t > learning_starts:
                action = select_epilson_greedy_action(Q, recent_observations, t)[0, 0]
            else:
                action = random.randrange(num_actions)
            # Advance one step
            obs, reward, done, _ = env.step(action)
            # clip rewards between -1 and 1
            reward = max(-1.0, min(reward, 1.0))
            # Store other info in replay memory
            replay_buffer.store_effect(last_idx, action, reward, done)
            # Resets the environment when reaching an episode boundary.
            if done:
                obs = env.reset()
            last_obs = obs

            ### Perform experience replay and train the network.
            # Note that this is only done if the replay buffer contains enough samples
            # for us to learn something useful -- until then, the model will not be
            # initialized and random actions should be taken
            if (t > learning_starts and
                    t % learning_freq == 0 and
                    replay_buffer.can_sample(batch_size)):
                # Use the replay buffer to sample a batch of transitions
                # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
                # in which case there is no Q-value at the next state; at the end of an
                # episode, only the current state reward contributes to the target
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
                # Convert numpy nd_array to torch variables for calculation
                obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
                act_batch = Variable(torch.from_numpy(act_batch).long())
                rew_batch = Variable(torch.from_numpy(rew_batch))
                next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
                not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

                if USE_CUDA:
                    act_batch = act_batch.cuda()
                    rew_batch = rew_batch.cuda()

                # Compute current Q value, q_func takes only state and output value for every state-action pair
                # We choose Q based on action taken.
                current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
                # Compute next Q value based on which action gives max Q values
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
                next_Q_values = not_done_mask * next_max_q
                # Compute the target of the current Q values
                target_Q_values = rew_batch + (gamma * next_Q_values)
                # Compute Bellman error
                bellman_error = target_Q_values - current_Q_values
                # clip the bellman error between [-1 , 1]
                clipped_bellman_error = bellman_error.clamp(-1, 1)
                # Note: clipped_bellman_delta * -1 will be right gradient
                d_error = clipped_bellman_error * -1.0
                # Clear previous gradients before backward pass
                optimizer.zero_grad()
                # run backward pass
                current_Q_values.backward(d_error.data.unsqueeze(1))

                # Perfom the update
                optimizer.step()
                num_param_updates += 1

                # Periodically update the target network by Q network to target Q network
                if num_param_updates % target_update_freq == 0:
                    target_Q.load_state_dict(Q.state_dict())

            ### 4. Log progress and keep track of statistics
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            Statistic["mean_episode_rewards"].append(mean_episode_reward)
            Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

            if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
                print("Timestep %d" % (t,))
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("best mean reward %f" % best_mean_episode_reward)
                print("episodes %d" % len(episode_rewards))
                print("exploration %f" % exploration.value(t))
                sys.stdout.flush()

                # Dump statistics to pickle
                with open('statistics.pkl', 'wb') as f:
                    pickle.dump(Statistic, f)
                    print("Saved to %s" % 'statistics.pkl')



def train_agent(layout: str):
    agent = DNQAgent(layout=layout, version='1')

    agent.train()


def run_agent(layout: str):
    agent = DNQAgent(layout=layout, version='1')
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
