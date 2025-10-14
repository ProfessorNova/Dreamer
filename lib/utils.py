"""
Utility functions for DreamerV3.

This module implements a handful of helper functions that are used across
the DreamerV3 implementation. In particular, it provides the symlog and
symexp transformations described in the DreamerV3 paper for robust
prediction targets. The paper introduces the symlog function as a
bi‑symmetric logarithmic mapping that compresses the magnitude of large
positive and negative values while leaving small values unchanged. Its
inverse, symexp, maps back into the original scale【145968576409203†L592-L633】.

In addition, this module contains helpers for building two‑hot encoded
targets for value predictions and a small convenience wrapper for logging
videos to TensorBoard.

This file does not depend on any other part of the codebase and can be
imported by all other modules.
"""
from __future__ import annotations

from typing import Iterable

import gymnasium as gym
import numpy as np
import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Apply the symlog transformation element‑wise.

    The symlog function compresses both the positive and negative range of
    its input while remaining roughly linear near the origin. It is defined
    as sign(x) * log(|x| + 1). This transformation is symmetric around
    zero and avoids the domain restrictions of the standard logarithm【145968576409203†L592-L633】.

    Args:
        x: Tensor of arbitrary shape.

    Returns:
        The symlog transform of `x`.
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Apply the inverse symlog (symexp) transformation.

    The inverse of symlog maps back into the original domain as
    sign(x) * (exp(|x|) - 1). It is the element‑wise inverse of symlog and
    preserves the sign of the input【145968576409203†L618-L624】.

    Args:
        x: Tensor of arbitrary shape.

    Returns:
        The symexp transform of `x`.
    """
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def value_bins(num_bins: int = 255, min_value: float = -20.0, max_value: float = 20.0) -> torch.Tensor:
    """Construct exponentially spaced bins for value distribution heads.

    DreamerV3 represents the critic’s value distribution using a categorical
    distribution over exponentially spaced bins and trains the critic via
    two‑hot regression【145968576409203†L638-L667】.  The bins are obtained by first
    uniformly spacing values in symlog space and then applying symexp to map
    them back into the original value domain. The default range of
    [−20, 20] covers a wide range of returns after applying symlog; these
    limits can be adjusted depending on the expected reward scale of the
    environment.

    Args:
        num_bins: Number of discrete bins to create.
        min_value: Minimum value in symlog space.
        max_value: Maximum value in symlog space.

    Returns:
        A 1D tensor containing the bin centers in original space.
    """
    lin = torch.linspace(min_value, max_value, num_bins)
    bins = symexp(lin)
    return bins


def two_hot_encode(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode continuous values into two‑hot vectors over the given bins.

    Two‑hot encoding generalizes one‑hot encoding to continuous targets by
    distributing probability mass between the two nearest bins【145968576409203†L655-L659】.  For
    each scalar value, this function finds the two closest bins and assigns
    weights such that their sum is one and the weight assigned to the bin
    closest to the value is larger.  This encoding allows gradients to flow
    smoothly when performing categorical regression.

    Args:
        values: Tensor of shape (batch,) containing continuous target values.
        bins: 1D tensor of shape (num_bins,) containing bin centers.

    Returns:
        Tensor of shape (batch, num_bins) with two‑hot encoded rows.
    """
    # Flatten input for easier indexing
    v = values.view(-1).unsqueeze(-1)  # shape (batch, 1)
    # Compute absolute distance between values and bins
    diff = (v - bins.view(1, -1)).abs()  # shape (batch, num_bins)
    # Find indices of the two closest bins
    top2 = diff.topk(k=2, largest=False)
    idx1, idx2 = top2.indices[:, 0], top2.indices[:, 1]
    d1, d2 = diff[torch.arange(diff.size(0)), idx1], diff[torch.arange(diff.size(0)), idx2]
    # Avoid division by zero by adding a small epsilon
    denom = d1 + d2 + 1e-8
    w1, w2 = d2 / denom, d1 / denom
    one_hot = torch.zeros((v.size(0), bins.numel()), device=values.device)
    one_hot.scatter_(1, idx1.unsqueeze(-1), w1.unsqueeze(-1))
    one_hot.scatter_(1, idx2.unsqueeze(-1), w2.unsqueeze(-1), reduce="add")
    return one_hot


def log_video(writer: "torch.utils.tensorboard.SummaryWriter", tag: str, frames: Iterable[np.ndarray], step: int,
              fps: int = 30) -> None:
    """Log a video sequence to TensorBoard.

    TensorBoard expects videos as a 5D tensor with shape (1, T, C, H, W) and
    pixel values in [0, 1]. This helper converts a sequence of HxWxC numpy
    images into the required format and writes it to the supplied summary
    writer.

    Args:
        writer: A `SummaryWriter` instance.
        tag: Name of the summary.
        frames: Iterable of numpy arrays of shape (H, W, C) in uint8 format.
        step: Global step associated with the video.
        fps: Frames per second for the video playback.
    """
    # Convert list of frames to a numpy array
    frame_list = list(frames)
    if not frame_list:
        return
    video = np.stack(frame_list, axis=0)  # shape (T, H, W, C)
    # Normalize to [0, 1] and reorder axes to (B, T, C, H, W)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    writer.add_video(tag, video, global_step=step, fps=fps)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def make_env(end_id) -> gym.Env:
    env = gym.make(end_id)
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = ImageToPyTorch(env)
    return env
