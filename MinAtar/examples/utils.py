import collections
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def now():
    return datetime.now().isoformat()


class Logger:
    def __init__(self, logdir: Path, step: int):
        self._logdir = Path(logdir)
        self.writer = SummaryWriter(log_dir=str(self._logdir))
        self._scalars = collections.defaultdict(list)
        self._images = {}
        self._videos = {}
        self._last_step = None
        self._last_time = None
        self.step = step

    def info(self, msg):
        tqdm.write(f"{now()} | {msg}")

    def scalar(self, name, value):
        value = float(value)
        self._scalars[name].append(value)

    def write(self, fps=False):
        scalars = {k: np.mean(v) for k, v in self._scalars.items()}
        scalars = list(scalars.items())
        if len(scalars) == 0:
            return
        if fps:
            scalars.append(("perf/fps", self._compute_fps(self.step)))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": self.step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            prefix = "" if "/" in name else "scalars/"
            self.writer.add_scalar(prefix + name, np.mean(value), self.step)

        self._scalars = collections.defaultdict(list)
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def close(self):
        self.writer.close()


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return False
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


def get_stats(name, x):
    x_metrics = {
        f"{name}_min": x.min(),
        f"{name}_max": x.max(),
        f"{name}_mean": x.mean(),
        f"{name}_std": x.std(),
    }
    return x_metrics


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
