import argparse
import math
import random
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from src.controller import Controller
from src.env.agent import Agent

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
steps_done = 0


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_screen(screen, player_position):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = screen.transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    pacman_location = player_position
    if pacman_location[0] < view_width // 2:
        slice_range = slice(view_width)
    elif pacman_location[0] > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(pacman_location[0] - view_width // 2,
                            pacman_location[0] + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def select_action(state, epsilon, policy_net, n_actions):
    global steps_done
    sample = np.random.uniform()

    steps_done += 1
    if sample > epsilon:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(memory, batch_size, policy_net, optimizer, target_net, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


class DNQAgent(Agent):
    name = 'dnq_agent'
    model: nn.Module

    def __init__(self, **kwargs):
        super().__init__()
        self.layout = kwargs['layout']
        self.version = kwargs['version']
        if self.version is None:
            self.filename = ''.join([self.name, '_', self.layout, '.pth'])
        else:
            self.filename = ''.join([self.name, '_', self.layout, '_', self.version, '.pth'])

    def act(self, *args, **kwargs):
        screen = get_screen(screen=kwargs['screen'], player_position=kwargs['player_pixel_pos'])
        print(self.model(screen))
        return self.model(screen).max(1)[1].view(1, 1).item()

    def save_model(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_model(self, screen_w, screen_h, n_actions):
        self.model = DQN(screen_h, screen_w, n_actions)
        state_dict = torch.load(self.filename, map_location="cuda:0")
        self.model.load_state_dict(state_dict)
        self.model.to(device)

    def train(self,
              n_episodes,
              num_steps,
              batch_size,
              replay_size,
              target_update_frequency,
              gamma=0.9,
              frame_to_skip: int = 10,
              **kwargs):
        GAMMA = gamma
        EPSILON = 1.0
        EPS_END = 0.1
        EPS_DECAY = 2000
        TARGET_UPDATE = target_update_frequency
        BATCH_SIZE = batch_size

        epsilon_by_frame = lambda frame_idx: EPS_END + (EPSILON - EPS_END) * math.exp(
            -1. * frame_idx / EPS_DECAY)

        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        env = gym.make('pacman-v0', layout=self.layout, frame_to_skip=frame_to_skip, player_lives=1)
        screen = env.get_screen_rgb_array()
        player_position = env.get_player_pixel_position()
        init_screen = get_screen(screen, player_position)
        _, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        n_actions = env.action_space.n

        policy_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net = DQN(screen_height, screen_width, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.RMSprop(policy_net.parameters())
        memory = ReplayMemory(1000)

        for i_episode in range(n_episodes):
            # Initialize the environment and state
            env.reset()
            screen = env.get_screen_rgb_array()
            player_position = env.get_player_pixel_position()
            last_screen = get_screen(screen, player_position)
            current_screen = get_screen(screen, player_position)
            state = current_screen - last_screen
            ep_reward = 0.
            EPSILON = epsilon_by_frame(i_episode)

            for t in count():
                # Select and perform an action
                env.render(mode='human')
                action = select_action(state, EPSILON, policy_net, n_actions)
                obs, reward, done, info = env.step(action.item())
                if info['player pixel position'] == info['prev player pixel position']:
                    reward -= 1
                ep_reward += reward

                reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                current_screen = get_screen(screen=obs, player_position=info['player pixel position'])
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                optimize_model(memory, BATCH_SIZE, policy_net, optimizer, target_net, GAMMA)
                if done:
                    print("Episode #{}, lasts for {} timestep, total reward: {}".format(i_episode, t + 1, ep_reward))
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print('Complete')
        env.render()
        env.close()

        self.save_model(target_net)


def train_agent(layout: str, episodes: int = 1000, **kwargs):
    agent = DNQAgent(layout=layout, version='1')

    agent.train(
        n_episodes=episodes,
        num_steps=100000,
        batch_size=64,
        replay_size=200,
        target_update_frequency=10,
        frame_to_skip=kwargs['frames_to_skip']
    )


def run_agent(layout: str):
    agent = DNQAgent(layout=layout, version='1')
    agent.load_model(screen_h=63, screen_w=40, n_actions=4)
    controller = Controller(layout_name=layout, act_sound=False, act_state=False, ai_agent=agent)
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
        train_agent(layout, episodes, frames_to_skip=frames_to_skip)

    if args.run:
        run_agent(layout)
