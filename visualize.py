"""Plotting utilities. Figures are saved to disk (script-safe) instead of
relying on an interactive plt.show(), which doesn't persist output when
running outside a notebook."""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 11


def plot_image_samples(model, test_loader, diff, timesteps, device, out_dir, n=8):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x[:n].to(device)
        t = torch.full((n,), timesteps // 2, device=device)
        noise = torch.randn_like(x)
        xt = diff.q_sample(x, t, noise)
        pred = model(xt, t)
        a = torch.sqrt(diff.alpha_bar[t])[:, None, None, None]
        b = torch.sqrt(1 - diff.alpha_bar[t])[:, None, None, None]
        x0_pred = ((xt - b * pred) / a.clamp(1e-8)).clamp(0, 1)

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(xt[i, 0].cpu(), cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(x0_pred[i, 0].cpu(), cmap="gray"); axes[2, i].axis("off")
    axes[0, 0].set_title("Clean", fontsize=10)
    axes[1, 0].set_title("Noisy", fontsize=10)
    axes[2, 0].set_title("Denoised (Spiking)", fontsize=10)
    plt.suptitle("Image Denoising Examples (Spiking Model)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "image_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_audio_waveforms(model, test_loader, T_audio, max_rate, device, out_dir, n=4):
    from src.diffusion_audio import poison

    model.eval()
    with torch.no_grad():
        x0 = next(iter(test_loader))[:n].to(device)
        t = torch.full((n,), int(T_audio * 0.7), device=device)
        noisy, _ = poison(x0, t, T_audio, max_rate, device)
        pred = model(noisy, t)
        den = (noisy - pred).clamp(0, 1)

    fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n), sharex=True)
    for i in range(n):
        axes[i].plot(x0[i].cpu(), label="Clean", alpha=0.8, linewidth=1.2)
        axes[i].plot(noisy[i].cpu(), label="Noisy", alpha=0.6, linewidth=1)
        axes[i].plot(den[i].cpu(), label="Denoised (Spiking)", alpha=0.9, linewidth=1.2)
        axes[i].set_ylabel(f"Sample {i + 1}")
        axes[i].legend(loc="upper right", fontsize=9)
        axes[i].set_ylim(-0.05, 1.05)
    axes[-1].set_xlabel("Time steps")
    plt.suptitle("Audio Denoising Examples (Spiking Model)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "audio_waveforms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison_bars(img_s_imp, img_ns_imp, audio_s_sdr_imp, audio_ns_sdr_imp,
                          sparsity, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(["Spiking", "Non-Spiking"], [img_s_imp, img_ns_imp],
                color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Image: MSE Improvement (%)")
    axes[0].set_ylabel("Improvement %")
    axes[0].set_ylim(0, 100)

    axes[1].bar(["Spiking", "Non-Spiking"], [audio_s_sdr_imp, audio_ns_sdr_imp],
                color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Audio: SI-SDR Improvement (dB)")
    axes[1].set_ylabel("dB")

    axes[2].bar(["Spiking", "Non-Spiking"], [sparsity, 0.0],
                color=["#4C72B0", "#DD8452"])
    axes[2].set_title("Sparsity")
    axes[2].set_ylabel("Sparsity (1 - spike rate)")
    axes[2].set_ylim(0, 1)

    plt.suptitle("Spiking vs Non-Spiking Comparison", fontsize=14, y=1.03)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
