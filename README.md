# Spiking vs Non-Spiking Residual Denoising

> **Note:** This README is a working draft. Structure/style will be updated to match the sample format once provided.

A controlled comparison between **spiking neural network (SNN)** and matched **non-spiking (ANN)** residual denoisers, evaluated on two modalities:

- **Image (MNIST):** diffusion-noised digits, denoised via epsilon-prediction.
- **Audio (SpeechCommands):** Poisson spiking-sensor-style corrupted waveforms, denoised via residual prediction.

Both branches use topologically matched spiking/non-spiking backbones so the only real variable is the neuron model (LIF spiking vs. SiLU-activated ANN), isolating the effect of spiking dynamics on denoising quality, latency, and activation sparsity.

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
└── outputs/                 # Created at runtime: figures + logs (gitignored)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt
```

## Usage

```bash
# Full run (both branches, default hyperparameters)
python main.py

# Only one branch
python main.py --skip-audio
python main.py --skip-image

# Quick smoke test
python main.py --epochs-img 2 --epochs-audio 2 --audio-subset-size 500
```

All hyperparameters (batch size, epochs, learning rate, timesteps, etc.) are exposed via CLI flags — run `python main.py --help` for the full list.

## Method Summary

| | Image | Audio |
|---|---|---|
| Corruption | Gaussian diffusion forward process | Poisson spike-count sensor model + Gaussian jitter |
| Objective | Predict added noise (`epsilon`) | Predict residual (`noisy - clean`) |
| Spiking block | LIF neurons, rate-coded over `N` internal steps | LIF neurons, rate-coded over `N` internal steps |
| Baseline | Matched ANN block (SiLU) | Matched ANN block (SiLU) |

## Fixes Applied to the Original Script

This version corrects a few issues found in the original monolithic notebook code:

1. **Fair spiking vs. non-spiking comparison** — both models in each branch are now evaluated on the *same* random noise draw (seeded), instead of independently sampled noise, removing an extra source of variance from the comparison.
2. **Sparsity/latency measured on realistic input** — `measure_sparsity` and `measure_time` now feed the audio models their actual corrupted ("noisy") input, matching what they were trained on, instead of the clean waveform.
3. **Explicit diffusion horizon** — `Diffusion(T=...)` is now passed explicitly rather than relying on a constructor default that happened to match the timestep constant.
4. **Explicit clamp bounds** — the Poisson rate in the audio corruption process is clamped with an explicit `min`/`max` instead of a lower bound only.
5. **Script-safe plotting** — figures are saved to `outputs/figures/` instead of relying on interactive `plt.show()`.

## License

MIT — see [LICENSE](LICENSE).
