#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("WebAgg")  # Use a non-blocking backend for interactive plotting
import matplotlib.pyplot as plt


def latex_style(font_size=11, width=6, use_tex=True):
    """Set plot defaults. Width is in inches; use_tex requires local LaTeX."""
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman" if use_tex else "cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.8,
            "figure.figsize": (width, width * 0.62),
            "figure.constrained_layout.use": False,
            "savefig.format": "pdf",
            "savefig.dpi": 300,
        }
    )


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
    sigma = np.where(x[:-1] < d, np.sqrt(variance), 0.5*np.sqrt(variance))
    noise = rng.normal(0, np.sqrt(dt)*sigma, size=len(x) - 1)
    log_returns = (drift - 0.5 * sigma**2) * dt + noise
    prices = np.empty(len(x))
    prices[0] = s0
    prices[1:] = s0 * np.exp(np.cumsum(log_returns))
    return prices


if __name__ == "__main__":
    latex_style(font_size=11, width=6)
    rng = np.random.default_rng(42)
    dt = 1.0
    s = -1.0
    average = 20
    x = ou_process(mean=0.0, variance=2.0, k=0.01, n_steps=1000, dt=dt, rng=rng)
    x_ma = moving_average(x, window=average)
    prices = signal_process(x, mu=1e-4, d=s, variance=0.0001, s0=100, dt=dt, rng=123)
    prices_2 = signal_process(x_ma, mu=1e-4, d=s, variance=0.0001, s0=100, dt=dt, rng=123)
    # To pair all three series, use x[average-1:], x_ma, and prices[average-1:].
    time = np.arange(len(x)) * dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="none")
    #fig.subplots_adjust(left=0.12, right=0.96, bottom=0.12, top=0.95, hspace=0.25)

    axes[0].plot(time, x, label="OU feature")
    axes[0].axhline(s, color="gray", linestyle="--", label="Threshold")
    axes[0].plot(time[average-1:], x_ma, label="Moving average")
    axes[0].set_ylabel("Feature Value")
    axes[0].legend()

    axes[1].plot(time, prices, label="Price with OU")
    axes[1].plot(time[average-1:], prices_2, label="Price with Moving Average")
    axes[1].set_ylabel("Price")
    axes[1].set_xlabel("Time")
    axes[1].legend()

    # Save before opening the interactive window.
    #fig.savefig("synthetic_data.png", dpi=150)
    plt.show()
