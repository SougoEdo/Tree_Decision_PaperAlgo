#!/usr/bin/env python3
"""Mean-reverting feature and feature-dependent stock price. Requires NumPy."""

import numpy as np


def ou_process(mean, variance, k, *, n_steps=1000, dt=1.0, x0=None, rng=None):
    """Simulate dX = k(mean - X)dt + sqrt(2*k*variance)dW exactly on a grid.

    `variance` is the long-run variance; k > 0 controls reversion speed.
    Return n_steps + 1 samples, at times 0, dt, ..., n_steps*dt.
    If x0 is omitted, start from the stationary distribution N(mean, variance).
    `rng` can be a NumPy Generator, an integer seed, or None.
    """
    if variance < 0 or k <= 0 or dt <= 0 or n_steps < 0:
        raise ValueError("Require variance >= 0, k > 0, dt > 0, n_steps >= 0.")

    rng = np.random.default_rng(rng)
    decay = np.exp(-k * dt)
    noise_std = np.sqrt(variance * -np.expm1(-2 * k * dt))
    x = np.empty(n_steps + 1)
    x[0] = rng.normal(mean, np.sqrt(variance)) if x0 is None else x0
    noise = rng.normal(0, noise_std, size=n_steps)
    for t in range(n_steps):
        x[t + 1] = mean + decay * (x[t] - mean) + noise[t]
    return x


def moving_average(x, window):
    """Trailing mean over full windows, with no padding or future values.

    Output has len(x) - window + 1 values, aligned with x[window - 1:].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not 1 <= window <= len(x):
        raise ValueError("Require a 1D series and 1 <= window <= len(x).")
    return np.convolve(x, np.ones(window) / window, mode="valid")


def signal_process(x, mu, d, variance, *, s0=100.0, dt=1.0, rng=None):
    """Simulate dS = drift(X)*S*dt + sqrt(variance)*S*dB, starting at s0.

    drift(X) is -mu below d, +mu otherwise; mu is a return drift per unit time.
    `variance` is the return variance rate (volatility squared), not price variance.
    Use the same dt as the feature process. Return len(x) prices, with S[0] = s0.
    x[t] determines the drift from S[t] to S[t+1], without looking ahead.
    The exponential update freezes drift over each step and keeps prices positive.
    Price noise draws are independent of feature noise.
    Pass a shared Generator or distinct seeds to keep the noise draws separate.
    """
    if not np.all(np.isfinite([mu, d, variance, s0, dt])):
        raise ValueError("Parameters must be finite.")
    if mu < 0 or variance < 0 or s0 <= 0 or dt <= 0:
        raise ValueError("Require mu >= 0, variance >= 0, s0 > 0, dt > 0.")
    rng = np.random.default_rng(rng)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Require a nonempty, finite 1D feature series.")

    drift = np.where(x[:-1] < d, -mu, mu)
    noise = rng.normal(0, np.sqrt(variance * dt), size=len(x) - 1)
    log_returns = (drift - 0.5 * variance) * dt + noise
    prices = np.empty(len(x))
    prices[0] = s0
    prices[1:] = s0 * np.exp(np.cumsum(log_returns))
    return prices


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dt = 1.0
    x = ou_process(mean=0.0, variance=1.0, k=0.2, n_steps=1000, dt=dt, rng=rng)
    x_ma = moving_average(x, window=20)
    prices = signal_process(x, mu=0.01, d=0.0, variance=0.0001, s0=100, dt=dt, rng=rng)
    # To pair all three series, use x[19:], x_ma, and prices[19:].
    print("Feature:", x[:5])
    print("Moving average:", x_ma[:5])
    print("Prices:", prices[:5])
