import bz2
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from addict import Dict
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms as T
from tqdm import tqdm


def print_config(config):
    lines = ["\nConfig:"]
    keys, vals, typs = [], [], []
    for key, val in vars(config).items():
        keys.append(key + ":")
        vals.append(_format_value(val))
        typs.append(_format_type(val))
    max_key = max(len(k) for k in keys) if keys else 0
    max_val = max(len(v) for v in vals) if vals else 0
    for key, val, typ in zip(keys, vals, typs):
        key = key.ljust(max_key)
        val = val.ljust(max_val)
        lines.append(f"{key}  {val}  ({typ})")
    return "\n".join(lines)


def _format_value(value):
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(x) for x in value) + "]"
    return str(value)


def _format_type(value):
    if isinstance(value, (list, tuple)):
        assert len(value) > 0, value
        return _format_type(value[0]) + "s"
    return str(type(value).__name__)


def set_seed(seed: int, cuda=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if cuda:
        torch.cuda.manual_seed(np.random.randint(1, 10000))


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, "rb") as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, "rb") as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, "wb") as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, "wb") as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


# Simple ISO 8601 timestamped logger
def log(s):
    msg = "[" + str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S")) + "] " + s
    tqdm.write(f"{msg}")


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


def make_bar(splits, data, path, fname):
    plt.figure()
    plt.bar(splits, data)
    plt.xlabel("Data split")
    plt.ylabel("Num samples")
    plt.tight_layout()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fname = path / fname
    plt.savefig(fname, bbox_inches="tight", dpi=120)
    plt.close()


def calculate_td_error(
    agent,
    online_net,
    target_net,
    states,
    actions,
    returns,
    next_states,
    nonterminals,
    weights,
):
    # Calculate current state probabilities (online network noise already sampled)
    log_ps = online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
    log_ps_a = log_ps[:, actions]  # log p(s_t, a_t; θonline)
    with torch.no_grad():
        # Calculate nth next state probabilities
        pns = online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
        dns = (
            agent.support.expand_as(pns) * pns
        )  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
        argmax_indices_ns = dns.sum(2).argmax(
            1
        )  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
        target_net.reset_noise()  # Sample new target net noise
        pns = target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
        pns_a = pns[
            :, argmax_indices_ns
        ]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

        # Compute Tz (Bellman operator T applied to z)
        Tz = returns.unsqueeze(1) + nonterminals * (
            agent.discount**agent.n
        ) * agent.support.unsqueeze(
            0
        )  # Tz = R^n + (γ^n)z (accounting for terminal states)
        Tz = Tz.clamp(min=agent.Vmin, max=agent.Vmax)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - agent.Vmin) / agent.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (agent.atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        m = states.new_zeros(agent.batch_size, agent.atoms)
        offset = (
            torch.linspace(0, ((agent.batch_size - 1) * agent.atoms), agent.batch_size)
            .unsqueeze(1)
            .expand(agent.batch_size, agent.atoms)
            .to(actions)
        )
        m.view(-1).index_add_(
            0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1)
        )  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(
            0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1)
        )  # m_u = m_u + p(s_t+n, a*)(b - l)

    loss = -torch.sum(m * log_ps_a, 1)
    loss = weights * loss
    return loss


def make_stats(name, x):
    data = {}
    data[f"{name}_mean"] = x.mean()
    data[f"{name}_std"] = x.std()
    data[f"{name}_min"] = x.min()
    data[f"{name}_max"] = x.max()
    return data
