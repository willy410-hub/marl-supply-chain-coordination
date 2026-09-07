"""
Pure NumPy Multi-Layer Perceptron with manual backpropagation and Adam.
========================================================================

No PyTorch / TensorFlow dependency by design: the state and action spaces
in this environment are small (10-45 dims), so a hand-rolled MLP is more
than sufficient computationally, and implementing backprop explicitly
here also makes the whole learning pipeline auditable end-to-end for a
project review -- every gradient is visible math, not a black-box
`.backward()` call.

Provides:
  - `MLP`: a configurable feed-forward network with tanh/relu hidden
    activations and a choice of output activation (linear / tanh).
  - `Adam`: a standard Adam optimizer operating on the MLP's parameter
    dict, used for both actor and critic updates.
  - Soft target-network updates (Polyak averaging), needed for the
    MADDPG-style centralized critic below.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional


def _init_layer(fan_in: int, fan_out: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # Xavier/Glorot-style init, scaled for tanh/relu stability.
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
    b = np.zeros(fan_out)
    return W, b


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def tanh_grad(y):
    # y is tanh(x) already (we cache activations, not pre-activations)
    return 1.0 - y ** 2


class MLP:
    """
    Simple feed-forward network: Linear -> ReLU -> ... -> Linear -> [out_act]

    Parameters are stored in a flat dict so Adam can be generic over any
    network shape (actor or critic).
    """

    def __init__(
        self,
        layer_sizes: List[int],
        out_activation: str = "linear",   # "linear", "tanh", or "sigmoid"
        seed: Optional[int] = None,
    ):
        assert len(layer_sizes) >= 2
        self.layer_sizes = layer_sizes
        self.out_activation = out_activation
        self.rng = np.random.default_rng(seed)

        self.params: Dict[str, np.ndarray] = {}
        for l in range(len(layer_sizes) - 1):
            W, b = _init_layer(layer_sizes[l], layer_sizes[l + 1], self.rng)
            self.params[f"W{l}"] = W
            self.params[f"b{l}"] = b
        self.n_layers = len(layer_sizes) - 1

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray, cache: Optional[dict] = None) -> np.ndarray:
        """
        x: shape (batch, in_dim)
        Returns output of shape (batch, out_dim). If `cache` dict is
        passed, stores intermediate activations needed for backward().
        """
        a = x
        if cache is not None:
            cache["a0"] = a
        for l in range(self.n_layers):
            W, b = self.params[f"W{l}"], self.params[f"b{l}"]
            z = a @ W + b
            is_last = (l == self.n_layers - 1)
            if not is_last:
                a = relu(z)
            else:
                if self.out_activation == "tanh":
                    a = np.tanh(z)
                elif self.out_activation == "sigmoid":
                    a = 1.0 / (1.0 + np.exp(-z))
                else:
                    a = z  # linear
            if cache is not None:
                cache[f"z{l}"] = z
                cache[f"a{l+1}"] = a
        return a

    # ------------------------------------------------------------------ #
    def backward(self, cache: dict, d_out: np.ndarray) -> Dict[str, np.ndarray]:
        """
        d_out: gradient of loss w.r.t. network output, shape (batch, out_dim).
        Returns grads dict with same keys as self.params, plus 'dX' for the
        gradient w.r.t. the network's input (useful for policy gradients
        flowing back through a critic into an actor's output).
        """
        grads = {}
        batch = d_out.shape[0]

        # Output-layer activation gradient
        a_last = cache[f"a{self.n_layers}"]
        if self.out_activation == "tanh":
            d_z = d_out * tanh_grad(a_last)
        elif self.out_activation == "sigmoid":
            d_z = d_out * a_last * (1 - a_last)
        else:
            d_z = d_out

        for l in reversed(range(self.n_layers)):
            a_prev = cache[f"a{l}"] if l > 0 else cache["a0"]
            W = self.params[f"W{l}"]

            grads[f"W{l}"] = (a_prev.T @ d_z) / batch
            grads[f"b{l}"] = d_z.mean(axis=0)

            if l > 0:
                d_a_prev = d_z @ W.T
                z_prev = cache[f"z{l-1}"]
                d_z = d_a_prev * relu_grad(z_prev)
            else:
                d_x = d_z @ W.T
                grads["dX"] = d_x

        return grads

    # ------------------------------------------------------------------ #
    def get_flat_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def set_flat_params(self, params: Dict[str, np.ndarray]):
        for k, v in params.items():
            self.params[k] = v.copy()

    def copy(self) -> "MLP":
        new_net = MLP(self.layer_sizes, self.out_activation)
        new_net.set_flat_params(self.params)
        return new_net

    def polyak_update(self, source: "MLP", tau: float):
        """Soft-update this network's params toward `source`'s params."""
        for k in self.params:
            self.params[k] = tau * source.params[k] + (1 - tau) * self.params[k]


class RunningNormalizer:
    """
    Tracks a running mean/std (Welford's online algorithm) and normalizes
    inputs to roughly zero-mean, unit-variance. Critical for this
    environment because raw observations mix very different natural
    scales (inventory ~0-100, backlog ~0-100, normalized timestep ~0-1),
    and unnormalized large-magnitude inputs saturate tanh-based actor
    networks almost immediately (gradient ~0 at the saturated tails),
    which is exactly the "actor collapses to always output max/min
    action" failure mode this project hit during development.
    """

    def __init__(self, dim: int, epsilon: float = 1e-4, clip: float = 5.0):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = epsilon
        self.clip = clip

    def update(self, x: np.ndarray):
        x = np.atleast_2d(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var) + 1e-6
        normed = (x - self.mean) / std
        return np.clip(normed, -self.clip, self.clip)


class Adam:
    """Standard Adam optimizer over a dict-of-arrays parameter set."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], grad_clip: float = 5.0):
        """
        grad_clip: max L2 norm for any single parameter tensor's gradient.
        Prevents the exploding-Q-value divergence that DDPG-family methods
        are notoriously prone to when TD targets grow large early in
        training (before the critic has calibrated to the reward scale).
        """
        self.t += 1
        for k in params:
            if k not in grads:
                continue
            g = grads[k]
            if grad_clip is not None:
                norm = np.linalg.norm(g)
                if norm > grad_clip:
                    g = g * (grad_clip / (norm + 1e-8))
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
