# Spiking vs Non-Spiking Residual Denoising

A controlled comparison between a spiking neural network (SNN) and a
topologically matched non-spiking (ANN) residual denoiser, evaluated on two
modalities — noisy MNIST digits and noisy speech waveforms — to isolate the
effect of spiking (LIF) dynamics on denoising quality, latency, and
activation sparsity.

## Overview

Diffusion-style corruption processes are commonly denoised with standard
convolutional ANNs. This project asks a narrower question: **if you swap
the nonlinearity in a residual denoiser's blocks from a standard activation
(SiLU) to spiking Leaky-Integrate-and-Fire (LIF) neurons — keeping the
conv/norm/skip topology and parameter budget otherwise identical — what do
you gain or lose?**

Two independent branches test this on different signal types:

- **Image branch (MNIST):** a Gaussian diffusion forward process corrupts
  digits; the network predicts the added noise (standard epsilon-prediction).
- **Audio branch (SpeechCommands):** a Poisson spike-count sensor model
  corrupts waveforms (simulating a degraded neuromorphic/cochlea-like
  sensor); the network predicts the residual (`noisy − clean`).

In both branches, the spiking model and its non-spiking counterpart share
the same encoder/decoder, channel widths, and time-conditioning — the only
difference is the neuron model inside each residual block.

**Headline result:** *pending a full training run — see [Usage](#usage) to
reproduce, and [`REPORT.md`](REPORT.md) for the full write-up once
populated with real run output.*

See [`REPORT.md`](REPORT.md) for the full write-up, including the fixes
applied to the original experimental script, the evaluation methodology,
and honestly-reported limitations.

## Architecture

- **Spiking residual block** — Conv → GroupNorm → LIF neuron (surrogate
  gradient, `fast_sigmoid`), unrolled over `N` internal timesteps per block
  and rate-averaged, with a skip connection.
- **Non-spiking residual block** — identical Conv/GroupNorm topology with a
  SiLU activation in place of the LIF neuron; no internal unroll.
- **Time conditioning** — sinusoidal timestep embedding (image branch) /
  learned scalar embedding (audio branch), projected and added inside every
  block, matched identically between spiking and non-spiking variants.
- **Image backbone** — 3-block conv residual stack, channel width doubling
  at block 2 (`32 → 64 → 64`).
- **Audio backbone** — 3-block 1D conv residual stack at constant width
  (`64` channels).

## Repository Structure

```
.
├── main.py                  # Entry point: trains + evaluates both branches
├── requirements.txt
├── src/
│   ├── diffusion_image.py   # DDPM-style noise schedule + time embedding (image)
│   ├── diffusion_audio.py   # Poisson corruption process (audio)
│   ├── models_image.py      # Spiking / non-spiking image residual blocks + nets
│   ├── models_audio.py      # Spiking / non-spiking audio residual blocks + nets
│   ├── datasets.py          # SpeechCommands dataset wrapper
│   ├── metrics.py           # SI-SDR, SNR, safe averaging
│   ├── train.py             # Training loops
│   ├── evaluate.py          # Paired evaluation, sparsity, latency measurement
│   └── visualize.py         # Figure generation (saved to outputs/figures/)
├── outputs/                 # Created at runtime: figures + logs (gitignored)
├── REPORT.md                # Full technical report
└── LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full comparison (both branches, default hyperparameters):

```bash
python main.py
```

Run a single branch only:

```bash
python main.py --skip-audio     # image branch only
python main.py --skip-image     # audio branch only
```

Quick smoke test (few epochs, small audio subset):

```bash
python main.py --epochs-img 2 --epochs-audio 2 --audio-subset-size 500
```

All hyperparameters (batch size, epochs, learning rate, diffusion
timesteps, internal spiking steps, etc.) are exposed via CLI flags — run
`python main.py --help` for the full list. Figures are written to
`outputs/figures/`.

## Results

*To be filled in after a full run of `main.py`* (see [`REPORT.md`](REPORT.md)
for the reporting template and methodology this table will follow):

| Branch | Metric | Spiking | Non-Spiking |
|---|---|---|---|
| Image (MNIST) | Denoised MSE / Improvement % | — | — |
| Audio (SpeechCommands) | SI-SDR Improvement (dB) | — | — |
| Audio (SpeechCommands) | SNR Improvement (dB) | — | — |
| Audio (SpeechCommands) | Spike rate / Sparsity | — | 1.0 / 0.0 (dense, by definition) |
| Audio (SpeechCommands) | Latency (ms/sample) | — | — |

## Limitations

This is a small-scale exploratory comparison, not a publication-ready
paper. Notable open items (detailed in `REPORT.md`):
- Single-seed by default; `main.py` does not yet run a multi-seed ablation
  the way the results table above implies it should for statistical
  reliability.
- The audio corruption process is a custom Poisson spiking-sensor model,
  not a standard Gaussian diffusion process — results are not directly
  comparable to published audio-diffusion denoising benchmarks.
- No comparison yet against established non-spiking denoising baselines
  from the literature (only the matched-topology ANN ablation included here).
- Evaluated on MNIST and a subset of SpeechCommands only; no test on
  natural images or full-length/real-world noisy audio.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic
  Models. *NeurIPS*.
- Eshraghian, J. K., et al. (2021). Training Spiking Neural Networks Using
  Lessons from Deep Learning. *arXiv:2109.12894* (snnTorch).
- Le Roux, N., et al. (2023). SI-SDR: Half-baked or well done? *ICASSP*
  (metric background).

## License

MIT — see [LICENSE](LICENSE).
