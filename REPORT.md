# Spiking vs Non-Spiking Residual Denoising

**Domain:** Neuromorphic Computing / Spiking Neural Networks / Denoising

---

## 1. Motivation

Spiking Neural Networks (SNNs) are often motivated by two claims: they can
be more energy-efficient than standard ANNs (due to sparse, event-driven
activations), and their temporal dynamics may suit signals that are
naturally time-structured (like audio). Denoising is a good testbed for
both claims — it has a clear, quantitative quality metric, and a clear
efficiency metric (activation sparsity / latency).

This project asks a narrow, controlled question: **if the only thing that
changes between two otherwise-identical residual denoisers is the neuron
model — Leaky Integrate-and-Fire (LIF) spiking neurons vs. a standard SiLU
activation — what is the effect on (a) denoising quality, (b) activation
sparsity, and (c) inference latency?**

Two modalities are tested so the comparison isn't an artifact of one
signal type:

- **Image (MNIST):** a standard Gaussian diffusion forward process.
- **Audio (SpeechCommands):** a Poisson spike-count sensor model, chosen
  because it is itself an event-based corruption process — a more natural
  fit for a spiking sensor/denoiser pairing than Gaussian noise.

## 2. Architecture

Both branches use a `Strong*Net` (spiking) and a `NonSpike*Net` (ANN
baseline) with matched topology:

- **Spiking residual block** (`ResBlock` / `ConvResBlock1D`) — Conv →
  GroupNorm → LIF neuron (`snntorch.Leaky`, surrogate gradient via
  `fast_sigmoid`) → Conv → GroupNorm → LIF neuron → add skip. The block is
  unrolled over `N` internal timesteps (constant input drive per step; only
  the membrane potential evolves) and the output is the mean spike-rate
  over those steps, added to the skip connection.
- **Non-spiking residual block** (`NonSpikeResBlock` / `NonSpikeConvRes1D`)
  — identical Conv/GroupNorm topology with `SiLU` in place of the LIF
  neuron; single forward pass, no internal unroll.
- **Time conditioning** — a timestep embedding (sinusoidal for the image
  branch, a small learned MLP for the audio branch) is projected per-block
  and added to the pre-activation feature map, identically in both variants.
- **Image backbone:** encoder conv → 3 residual blocks (`32 → 32 → 64 →
  64` channels) → output conv. Predicts the added Gaussian noise
  (epsilon-prediction), following standard DDPM training.
- **Audio backbone:** input conv → 3 residual blocks (constant 64
  channels) → output conv. Predicts the residual `noisy − clean`.

## 3. Experimental Setup

- **Image task:** MNIST, Gaussian diffusion forward process, `T=20`
  timesteps, epsilon-prediction objective, evaluated by reconstructing
  `x0` from the predicted noise and comparing denoised vs. noisy MSE
  against the clean image.
- **Audio task:** a subset of SpeechCommands (6,000 clips by default, 1s
  each, resampled to 16kHz), corrupted via a Poisson spike-count process
  with `T=40` corruption steps; the model predicts the residual noise
  component. Evaluated via MSE, SI-SDR improvement, and SNR improvement
  against the clean waveform.
- **Efficiency metrics (audio only):** spike rate / sparsity, measured via
  forward hooks on every `snntorch.Leaky` layer; and per-sample latency,
  measured with CUDA-synchronized timing.
- **Fair-comparison protocol:** spiking and non-spiking models are
  evaluated on the **same** noise/corruption draw (seeded), and sparsity /
  latency are measured on the **same corrupted inputs the models were
  trained on** — see Section 4 for why this needed to be fixed.
- **Hardware:** designed to run on a single consumer/free-tier GPU
  (CUDA if available, CPU fallback).

## 4. Development / Debugging Journey

This project started from an existing monolithic experiment script. Before
any new experiments were run, the script was audited for conceptual and
methodological issues, since several would have silently invalidated any
comparison between the spiking and non-spiking models.

| Issue found | Problem | Fix |
|---|---|---|
| Unpaired evaluation | Spiking and non-spiking models were each evaluated with independently sampled random noise/timesteps, adding an uncontrolled source of variance to a comparison that is supposed to isolate the neuron-model effect | Both models in a branch are now evaluated on the same seeded noise draw (`eval_img_pair` / `evaluate_audio_pair`) |
| Sparsity/latency measured off-distribution | `measure_sparsity` and `measure_time` fed the audio model **clean** waveforms directly, but the model was trained on, and only ever meant to see, **corrupted** waveforms — so the reported spike-rate and latency did not reflect real inference conditions | Both functions now corrupt the input via the same `poison()` process used in training/eval before measuring |
| Implicit diffusion horizon | `Diffusion(device=device)` relied on a constructor default (`T=20`) that happened to equal the `TIMESTEPS` constant used elsewhere; changing one without the other would silently desynchronize the noise schedule from the training loop | `T` is now passed explicitly everywhere it's used |
| Under-specified clamp | The Poisson corruption rate was clamped with only a lower bound (`torch.clamp(rate, 1e-4)`); correct under current assumptions (`x0 ≤ 1`, `gamma ≤ 1`) but silently reliant on those assumptions holding | Explicit `min`/`max` clamp added |
| Notebook-only plotting | Figures relied on interactive `plt.show()`, which doesn't persist output in a script/CI context | Figures are now saved to `outputs/figures/` |

No architectural or optimizer changes were made at this stage — the goal
was to make the *comparison* trustworthy before generating any numbers to
report on it. The refactor into `src/` modules (Section 2 in
[`README.md`](README.md)) was done alongside this audit.

**Status at time of writing:** these fixes have been applied and the
codebase compiles and runs end-to-end, but a full multi-epoch training run
has not yet been executed in the environment this report was written in
(no internet access to download MNIST/SpeechCommands there). Section 5
below is a template to be filled in from an actual `python main.py` run
rather than fabricated numbers.

## 5. Results

> **This section is a template.** Run `python main.py` (see
> [Usage](README.md#usage)) and fill in the numbers it prints /
> `outputs/figures/*.png` it generates. Do not treat the placeholders
> below as real measurements.

### 5.1 Image branch (MNIST)

| Model | Noisy MSE | Denoised MSE | Improvement |
|---|---|---|---|
| Spiking | — | — | —% |
| Non-Spiking | — | — | —% |

### 5.2 Audio branch (SpeechCommands)

| Metric | Spiking | Non-Spiking |
|---|---|---|
| MSE (denoised) | — | — |
| SI-SDR improvement (dB) | — | — |
| SNR improvement (dB) | — | — |
| Spike rate | — | 1.0000 (dense) |
| Sparsity | — | 0.0000 |
| Latency (ms/sample) | — ± — | — ± — |

### 5.3 How to read these once populated

- **Quality:** compare denoised MSE / SI-SDR / SNR improvement between
  rows. A spiking model that matches or beats the ANN baseline on quality
  *and* is sparser/faster is the strongest possible result; a spiking
  model that trails on quality but is much sparser is a
  quality-for-efficiency trade-off, which is still a legitimate finding.
- **Efficiency:** spike rate close to the ANN's implicit "1.0" (always
  active) means the spiking model isn't exploiting sparsity; a low spike
  rate with maintained quality is the headline claim this kind of study
  is usually trying to support.
- Because evaluation is now paired (Section 4), any quality gap between
  the two models reflects the architecture/neuron-model difference and
  not differing noise draws.

## 6. Limitations

- **No results yet** — Section 5 is unpopulated; every claim about
  spiking-vs-non-spiking performance in this project is currently
  hypothetical until a run is completed and reported.
- **Single seed by default.** `main.py` runs once end-to-end; unlike a
  proper ablation study, it does not yet aggregate multiple seeds into a
  mean ± std the way a statistically reliable comparison would need
  (particularly important given how much variance a small MNIST/SNN setup
  can show, as seen in related continual-learning work).
- **Audio corruption is a custom process, not standard diffusion.** The
  Poisson spiking-sensor model was chosen for topical fit (event-based
  corruption paired with an event-based denoiser) rather than because it
  matches any established audio-diffusion benchmark. Results should not be
  compared directly to papers using Gaussian audio diffusion.
- **No external baselines.** The only non-spiking comparison is the
  matched-topology ablation in this repo; there is no comparison against
  established denoising architectures (e.g. standard U-Net denoisers,
  DEMUCS-style audio denoisers) from the literature.
- **Limited data scope.** MNIST (not natural images) and a 6,000-clip
  subset of SpeechCommands (not full-length or real-world noisy audio).
- **Sparsity is only measured for the audio branch.** The image branch has
  no equivalent spike-rate instrumentation yet.

## 7. Future Work

- Run `main.py` to completion (both branches) and populate Section 5 with
  real numbers and generated figures.
- Extend to a multi-seed ablation (mirroring the `run_experiments.py`
  pattern from related continual-learning work in this space) to report
  mean ± std rather than a single run.
- Add spike-rate/sparsity instrumentation to the image branch, mirroring
  what already exists for audio.
- Compare against at least one established non-spiking denoising baseline
  from the literature, not just the matched-topology ablation.
- Test the audio branch on full-length, real-world noisy recordings rather
  than short fixed-length SpeechCommands clips.
- Ablate the number of internal spiking timesteps (`num_steps`) to
  characterize the quality/latency/sparsity trade-off curve directly,
  rather than reporting a single operating point.

## 8. Conclusion

This report documents the methodology and codebase for a controlled
spiking-vs-non-spiking residual denoising comparison across image and
audio modalities, including a set of conceptual/methodological fixes
(paired evaluation, on-distribution sparsity/latency measurement, explicit
diffusion horizon, explicit clamping) applied to make the eventual
comparison trustworthy. No quality or efficiency claims are made yet — the
next step is to run the pipeline end-to-end and populate Section 5 with
real results before drawing any conclusions about spiking vs. non-spiking
performance.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic
  Models. *NeurIPS*.
- Eshraghian, J. K., Ward, M., Neftci, E., et al. (2021). Training Spiking
  Neural Networks Using Lessons from Deep Learning. *arXiv:2109.12894*
  (snnTorch).
- Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019). SDR –
  Half-baked or Well Done? *ICASSP*.
