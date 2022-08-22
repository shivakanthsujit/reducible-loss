import argparse
import logging
import os
import random
import time
from collections import namedtuple

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import utils
from examples.per_buffer import PERReplayMemory
from minatar import Environment
from tqdm import tqdm

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_EPS = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = f.relu(self.conv(x))
        x = nn.Flatten()(x)
        # Rectified output from the final hidden layer
        x = f.relu(self.fc_hidden(x))

        # Returns the output from the fully-connected linear layer
        return self.output(x)


def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net):

    # A uniform random policy is run before the learning starts
    if t < replay_start_size:
        action = random.randrange(num_actions)
    else:
        # Epsilon-greedy behavior policy for action selection
        # Epsilon is annealed linearly from 1.0 to END_EPSILON over the FIRST_N_FRAMES and stays 0.1 for the
        # remaining frames
        epsilon = (
            END_EPSILON
            if t - replay_start_size >= FIRST_N_FRAMES
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size)
            + EPSILON
        )

        if numpy.random.binomial(1, epsilon) == 1:
            action = random.randrange(num_actions)
        else:
            # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
            # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
            # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
            with torch.no_grad():
                s = get_state(s)
                action = policy_net(s).max(1)[1].squeeze().cpu().numpy()

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = env.state()

    return s_prime, action, reward, terminated


################################################################################################################
# train
#
# This is where learning happens. More specifically, this function learns the weights of the policy network
# using huber loss.
#
# Inputs:
#   sample: a batch of size 1 or 32 transitions
#   policy_net: an instance of QNetwork
#   target_net: an instance of QNetwork
#   optimizer: centered RMSProp
#
################################################################################################################
def train(sample, weights, policy_net, target_net, optimizer):
    (states, actions, returns, next_states, nonterminals) = sample
    weights = weights[..., None]
    Q_s_a = policy_net(states).gather(1, actions)
    with torch.no_grad():
        Q_s_prime_a_prime = target_net(next_states).max(1)[0].unsqueeze(1).detach()
        assert Q_s_prime_a_prime.shape == Q_s_a.shape
    target = returns + nonterminals * GAMMA * Q_s_prime_a_prime
    assert target.shape == Q_s_a.shape
    # Huber loss
    loss = f.smooth_l1_loss(target, Q_s_a, reduction="none")
    assert loss.shape == weights.shape, f"{loss.shape} != {weights.shape}"
    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    is_loss = weights * loss
    assert is_loss.shape == loss.shape, f"{is_loss.shape} != {loss.shape}"
    is_loss.mean().backward()
    optimizer.step()
    return loss


################################################################################################################
# dqn
#
# DQN algorithm with the option to disable replay and/or target network, and the function saves the training data.
#
# Inputs:
#   env: environment of the game
#   replay_off: disable the replay buffer and train on each state transition
#   target_off: disable target network
#   output_file_name: directory and file name prefix to output data and network weights, file saved as
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: file path for a checkpoint to load, and continue training from
#   step_size: step-size for RMSProp optimizer
#
#################################################################################################################
def dqn(
    env,
    summary_writer: utils.Logger,
    output_file_name,
    store_intermediate_result=False,
    load_path=None,
    step_size=STEP_SIZE,
):

    # Get channels and number of actions specific to each game
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    # Instantiate networks, optimizer, loss and buffer
    policy_net = QNetwork(in_channels, num_actions).to(device)
    target_net = QNetwork(in_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # r_buffer = per_replay_buffer(REPLAY_BUFFER_SIZE, PER_ALPHA, PER_BETA, PER_EPS)
    r_buffer = PERReplayMemory(
        (10, 10, in_channels), REPLAY_BUFFER_SIZE, PER_ALPHA, PER_BETA, PER_EPS, device
    )
    replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(
        policy_net.parameters(),
        lr=step_size,
        alpha=SQUARED_GRAD_MOMENTUM,
        centered=True,
        eps=MIN_SQUARED_GRAD,
    )

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    avg_ep_len_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        policy_net.load_state_dict(checkpoint["policy_net_state_dict"])

        target_net.load_state_dict(checkpoint["target_net_state_dict"])

        r_buffer = checkpoint["replay_buffer"]

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        e_init = checkpoint["episode"]
        t_init = checkpoint["frame"]
        policy_net_update_counter_init = checkpoint["policy_net_update_counter"]
        avg_return_init = checkpoint["avg_return"]
        avg_ep_len_init = checkpoint["avg_ep_len"]
        data_return_init = checkpoint["return_per_run"]
        frame_stamp_init = checkpoint["frame_stamp_per_run"]

    # Set to training mode
    policy_net.train()
    target_net.train()
    for param in target_net.parameters():
        param.requires_grad = False

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init
    avg_ep_len = avg_ep_len_init

    should_log = utils.Every(5e3)
    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    priority_beta_increase = (1 - PER_BETA) / (NUM_FRAMES - REPLAY_START_SIZE)
    t_start = time.time()
    pbar = tqdm(initial=t, total=NUM_FRAMES)
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        s = env.state()
        is_terminated = False
        while (not is_terminated) and t < NUM_FRAMES:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(
                t, replay_start_size, num_actions, s, env, policy_net
            )

            # Write the current frame to replay buffer
            r_buffer.add(s, action, reward, s_prime, is_terminated)
            sample = None
            # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
            if t > REPLAY_START_SIZE and len(r_buffer) >= BATCH_SIZE:
                # Sample a batch
                idxs, sample, weights = r_buffer.sample(BATCH_SIZE)
                r_buffer.priority_beta = min(
                    r_buffer.priority_beta + priority_beta_increase, 1
                )

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                policy_net_update_counter += 1
                loss = train(sample, weights, policy_net, target_net, optimizer)
                loss = torch.abs(loss)
                r_buffer.update_priorities(
                    idxs, loss.detach().cpu().numpy().squeeze(-1)
                )

            # Update the target network only after some number of policy network updates
            if (
                policy_net_update_counter > 0
                and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0
            ):
                target_net.load_state_dict(policy_net.state_dict())

            G += reward

            t += 1
            pbar.update(1)

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)
        if len(frame_stamp) > 1:
            ep_len = frame_stamp[-1] - frame_stamp[-2]
        else:
            ep_len = frame_stamp[-1]
        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        avg_ep_len = 0.99 * avg_ep_len + 0.01 * ep_len
        if should_log(t):
            summary_writer.step = t
            summary_writer.scalar("beta", r_buffer.priority_beta)
            summary_writer.scalar("return", G)
            summary_writer.scalar("ep_len", ep_len)
            summary_writer.scalar("avg_ep_len", avg_ep_len)
            summary_writer.scalar("avg_return", avg_return)
            summary_writer.write(True)

        if e % 1000 == 0:
            log_dict = {
                "Episode": e,
                "Return:": G,
                "Avg return": f"{avg_return:.2f}",
                "Ep Len": ep_len,
                "Avg length": f"{avg_ep_len:.2f}",
                "Frame": t,
                "Time per frame": (time.time() - t_start) / t,
            }
            msg = " | ".join([f"{k}: {v}" for k, v in log_dict.items()])
            summary_writer.info(msg)

        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 1000 == 0:
            torch.save(
                {
                    "episode": e,
                    "frame": t,
                    "policy_net_update_counter": policy_net_update_counter,
                    "policy_net_state_dict": policy_net.state_dict(),
                    "target_net_state_dict": target_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_return": avg_return,
                    "return_per_run": data_return,
                    "avg_ep_len": avg_ep_len,
                    "frame_stamp_per_run": frame_stamp,
                    # 'replay_buffer': r_buffer
                },
                output_file_name + "_checkpoint",
            )

    summary_writer.step = t
    summary_writer.scalar("return", G)
    summary_writer.scalar("avg_return", avg_return)
    summary_writer.write(True)
    # Print final logging info
    summary_writer.info(
        f"Avg return: {avg_return:.2f} | Time per frame: {str((time.time()-t_start)/t)}"
    )

    # Write data to file
    torch.save(
        {
            "returns": data_return,
            "frame_stamps": frame_stamp,
            "policy_net_state_dict": policy_net.state_dict(),
        },
        output_file_name + "_data_and_weights",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--id", default="baseline", type=str)
    parser.add_argument("--logdir", default="logs", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
    parser.add_argument("--save", "-s", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_name = os.path.join(args.game, f"per_dqn_{args.id}", f"seed{args.seed}")
    logdir = os.path.join(args.logdir, run_name)
    os.makedirs(logdir, exist_ok=True)
    summary_writer = utils.Logger(logdir, 0)
    summary_writer.info("Training PER DQN on MinAtar")
    summary_writer.info(f"Logging to {logdir}")

    file_name = os.path.join(logdir, "result")

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    utils.set_seed(args.seed)
    env = Environment(args.game, random_seed=args.seed)

    print("Cuda available?: " + str(torch.cuda.is_available()))
    dqn(env, summary_writer, file_name, args.save, load_file_path, args.alpha)


if __name__ == "__main__":
    main()
