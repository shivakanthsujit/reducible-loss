import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class Logger:
    def __init__(self, logdir: Path, step: int):
        self._logdir = Path(logdir)
        self.writer = SummaryWriter(log_dir=str(self._logdir))
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._last_step = None
        self._last_time = None
        self.step = step

    def scalar(self, name, value):
        if name in self._scalars:
            if type(self._scalars[name]) != list:
                self._scalars[name] = [self._scalars[name]]
            self._scalars[name].append(value)
        else:
            self._scalars[name] = float(value)

    @torch.no_grad()
    def image(self, name, value):
        value = value.clip(0.0, 1.0)
        self._images[name] = value

    def video(self, name, value, log_to_tensorboard=True):
        self._videos[name] = [np.array(value), log_to_tensorboard]

    def write(self, fps=False):
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("perf/fps", self._compute_fps(self.step)))
        video_fps = 5
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": self.step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            prefix = "" if "/" in name else "scalars/"
            self.writer.add_scalar(prefix + name, np.mean(value), self.step)
        for name, value in self._images.items():
            s_name = name.replace("/", "_")
            os.makedirs(self._logdir / "images", exist_ok=True)
            save_path = self._logdir / "images" / f"{s_name}.png"
            tqdm.write(f"Saved image to {save_path}")
            torchvision.utils.save_image(torchvision.utils.make_grid(value), save_path)
            # prefix = "" if "/" in name else "images/"
            # self.writer.add_images(prefix + name, value, self.step)
        for name, value in self._videos.items():
            write_gif(name, value[0], self._logdir, video_fps)
            if value[1] is True:
                self.writer.add_video(name, value[0], self.step, video_fps)

        self._scalars = {}
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


def write_gif(name: str, frames: np.ndarray, logdir: Path, fps: int):
    B, T, C, H, W = frames.shape
    frames = frames.transpose((1, 3, 0, 4, 2)).reshape((T, H, B * W, C))
    frames = np.clip(frames * 255.0, 0, 255.0).astype(np.uint8)
    video_dir = logdir / "gifs"
    video_dir.mkdir(parents=True, exist_ok=True)
    filename = video_dir / f"{name.replace('/', '_')}.gif"
    write_gif_to_disk(frames, filename, fps)


def write_gif_to_disk(frames: np.ndarray, filename: str, fps: int = 5):
    from moviepy.editor import ImageSequenceClip

    try:
        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_gif(filename, fps=fps, logger=None)
        tqdm.write(f"GIF saved to {filename}")
    except Exception as e:
        tqdm.write(frames.shape)
        tqdm.write("GIF Saving failed.", e)
