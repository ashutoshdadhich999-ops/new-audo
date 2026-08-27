"""
Forward corruption process for the audio (SpeechCommands) branch.

This is intentionally NOT a Gaussian DDPM process. It simulates a
Poisson (spike-count) sensor: as the "timestep" t increases, the
effective firing rate decays (gamma -> 0) and a small amount of
Gaussian jitter is layered on top. The model is trained to predict
the residual (noisy - clean) at a given t, i.e. single-step residual
denoising conditioned on corruption strength -- not iterative ancestral
sampling.
"""

import torch


def poison(x0: torch.Tensor, t: torch.Tensor, T_audio: int, max_rate: float,
           device: str, scale: float = 15.0, decay: float = 0.06,
           jitter_std: float = 0.05):
    """Corrupt a clean waveform batch `x0` at diffusion step `t`.

    Returns:
        noisy: corrupted waveform, clamped to [0, 1]
        target: (noisy - x0), the residual the model is trained to predict
    """
    t = t.view(-1, 1).float().to(device)
    gamma = torch.exp(-decay * t * 6 / T_audio)

    rate = x0 * gamma * max_rate
    # FIX: clamp to an explicit [min, max] range instead of only a lower
    # bound. Mathematically rate stays <= max_rate here since x0 in [0, 1]
    # and gamma <= 1, but making the bound explicit avoids silent bugs if
    # those assumptions ever change.
    rate = torch.clamp(rate, min=1e-4, max=max_rate)

    counts = torch.poisson(rate * scale)
    noisy = counts / scale
    noisy = noisy + torch.randn_like(noisy) * jitter_std * (1 - gamma)

    noisy = noisy.clamp(0, 1)
    return noisy, noisy - x0
