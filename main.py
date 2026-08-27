"""
Spiking vs Non-Spiking Residual Denoising -- main experiment script.

Runs both the image (MNIST) and audio (SpeechCommands) branches:
trains a spiking (SNN) and a matched non-spiking (ANN) residual
denoiser for each modality, evaluates them under a fair, paired
comparison, and saves result plots.

Usage:
    python main.py                     # run everything with defaults
    python main.py --skip-audio        # image branch only
    python main.py --skip-image        # audio branch only
    python main.py --epochs-img 5 --epochs-audio 5   # quick smoke test
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchaudio

from src.diffusion_image import Diffusion
from src.models_image import StrongImgNet, NonSpikeImgNet
from src.models_audio import StrongAudioNet, NonSpikeAudioNet
from src.datasets import AudioDS
from src.train import train_img_model, train_audio_model
from src.evaluate import (
    eval_img_pair, evaluate_audio_pair, measure_sparsity, measure_time,
)
from src.visualize import plot_image_samples, plot_audio_waveforms, plot_comparison_bars


def parse_args():
    p = argparse.ArgumentParser(description="Spiking vs Non-Spiking Residual Denoising")
    p.add_argument("--skip-image", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs")

    # Image branch
    p.add_argument("--batch-size-img", type=int, default=64)
    p.add_argument("--epochs-img", type=int, default=12)
    p.add_argument("--timesteps-img", type=int, default=20)
    p.add_argument("--num-steps-img", type=int, default=5)
    p.add_argument("--base-channels-img", type=int, default=32)
    p.add_argument("--lr-img", type=float, default=2e-4)

    # Audio branch
    p.add_argument("--audio-len", type=int, default=8000)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--timesteps-audio", type=int, default=40)
    p.add_argument("--num-steps-audio", type=int, default=8)
    p.add_argument("--epochs-audio", type=int, default=20)
    p.add_argument("--batch-size-audio", type=int, default=32)
    p.add_argument("--max-rate-audio", type=float, default=0.9)
    p.add_argument("--lr-audio", type=float, default=3e-4)
    p.add_argument("--audio-subset-size", type=int, default=6000)

    return p.parse_args()


def run_image_branch(args, device):
    print("\n" + "=" * 70)
    print("PART 1: MNIST Image Denoising")
    print("=" * 70)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size_img, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_img, shuffle=False,
                              num_workers=2, pin_memory=True)

    # FIX: pass T explicitly instead of relying on the constructor default
    diff = Diffusion(T=args.timesteps_img, device=device)

    model_img = StrongImgNet(base_channels=args.base_channels_img,
                              num_steps=args.num_steps_img).to(device)
    model_ns_img = NonSpikeImgNet(base_channels=args.base_channels_img).to(device)

    model_img = train_img_model(model_img, "Spiking Image", train_loader, diff,
                                 args.timesteps_img, args.epochs_img, args.lr_img, device)
    model_ns_img = train_img_model(model_ns_img, "Non-Spiking Image", train_loader, diff,
                                    args.timesteps_img, args.epochs_img, args.lr_img, device)

    print("\n--- Image Results (paired, matched noise draws) ---")
    (img_s_noisy, img_s_den, img_s_imp), (img_ns_noisy, img_ns_den, img_ns_imp) = eval_img_pair(
        model_img, model_ns_img, "Spiking Image", "Non-Spiking Image",
        test_loader, diff, args.timesteps_img, device, seed=args.seed,
    )

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_image_samples(model_img, test_loader, diff, args.timesteps_img, device, fig_dir)

    return {
        "img_s_noisy": img_s_noisy, "img_s_den": img_s_den, "img_s_imp": img_s_imp,
        "img_ns_noisy": img_ns_noisy, "img_ns_den": img_ns_den, "img_ns_imp": img_ns_imp,
    }


def run_audio_branch(args, device):
    print("\n" + "=" * 70)
    print("PART 2: Audio Denoising (SpeechCommands)")
    print("=" * 70)

    os.makedirs("./data", exist_ok=True)
    base = torchaudio.datasets.SPEECHCOMMANDS("./data", download=True)
    subset = Subset(base, range(min(args.audio_subset_size, len(base))))
    tr_size = int(0.85 * len(subset))
    tr_sub, te_sub = random_split(subset, [tr_size, len(subset) - tr_size])

    train_loader = DataLoader(AudioDS(tr_sub, args.audio_len, args.audio_sr),
                               batch_size=args.batch_size_audio, shuffle=True,
                               num_workers=2, pin_memory=True)
    test_loader = DataLoader(AudioDS(te_sub, args.audio_len, args.audio_sr),
                              batch_size=args.batch_size_audio, shuffle=False,
                              num_workers=2, pin_memory=True)

    model_a = StrongAudioNet(num_steps=args.num_steps_audio, T_audio=args.timesteps_audio).to(device)
    model_ns_a = NonSpikeAudioNet(T_audio=args.timesteps_audio).to(device)

    model_a = train_audio_model(model_a, "Spiking Audio", train_loader,
                                 args.timesteps_audio, args.max_rate_audio,
                                 args.epochs_audio, args.lr_audio, device)
    model_ns_a = train_audio_model(model_ns_a, "Non-Spiking Audio", train_loader,
                                    args.timesteps_audio, args.max_rate_audio,
                                    args.epochs_audio, args.lr_audio, device)

    print("\n--- Audio Results (paired, matched noise draws) ---")
    res_s, res_ns = evaluate_audio_pair(
        model_a, model_ns_a, "Spiking Audio", "Non-Spiking Audio",
        test_loader, args.timesteps_audio, args.max_rate_audio, device, seed=args.seed,
    )

    # FIX: sparsity/timing now measured on realistic (corrupted) inputs
    spike_rate, sparsity = measure_sparsity(model_a, test_loader, args.timesteps_audio,
                                             args.max_rate_audio, device)
    (t_s, t_s_std), (t_ns, t_ns_std) = measure_time(model_a, model_ns_a, test_loader,
                                                      args.timesteps_audio,
                                                      args.max_rate_audio, device)

    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_audio_waveforms(model_a, test_loader, args.timesteps_audio,
                          args.max_rate_audio, device, fig_dir)

    return {
        "res_s": res_s, "res_ns": res_ns,
        "spike_rate": spike_rate, "sparsity": sparsity,
        "t_s": t_s, "t_s_std": t_s_std, "t_ns": t_ns, "t_ns_std": t_ns_std,
    }


def print_final_tables(img_results, audio_results):
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    if img_results:
        print("\n[IMAGE]")
        print(f"{'Model':<25} {'Noisy MSE':>12} {'Denoised MSE':>14} {'Improvement':>12}")
        print("-" * 65)
        print(f"{'Spiking':<25} {img_results['img_s_noisy']:>12.5f} "
              f"{img_results['img_s_den']:>14.5f} {img_results['img_s_imp']:>11.2f}%")
        print(f"{'Non-Spiking':<25} {img_results['img_ns_noisy']:>12.5f} "
              f"{img_results['img_ns_den']:>14.5f} {img_results['img_ns_imp']:>11.2f}%")

    if audio_results:
        res_s, res_ns = audio_results["res_s"], audio_results["res_ns"]
        print("\n[AUDIO]")
        print(f"{'Metric':<22} {'Spiking':>12} {'Non-Spiking':>14}")
        print("-" * 50)
        print(f"{'MSE (Denoised)':<22} {res_s['MSE']:>12.5f} {res_ns['MSE']:>14.5f}")
        print(f"{'SI-SDR Imp (dB)':<22} {res_s['SI-SDR Imp']:>12.2f} {res_ns['SI-SDR Imp']:>14.2f}")
        print(f"{'SNR Imp (dB)':<22} {res_s['SNR Imp']:>12.2f} {res_ns['SNR Imp']:>14.2f}")
        print(f"{'Spike Rate':<22} {audio_results['spike_rate']:>12.4f} {'1.0000':>14}")
        print(f"{'Sparsity':<22} {audio_results['sparsity']:>12.4f} {'0.0000':>14}")
        print(f"{'Time (ms/sample)':<22} "
              f"{audio_results['t_s']:>9.3f}\u00b1{audio_results['t_s_std']:.2f} "
              f"{audio_results['t_ns']:>10.3f}\u00b1{audio_results['t_ns_std']:.2f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    os.makedirs(args.out_dir, exist_ok=True)

    img_results = run_image_branch(args, device) if not args.skip_image else None
    audio_results = run_audio_branch(args, device) if not args.skip_audio else None

    print_final_tables(img_results, audio_results)

    if img_results and audio_results:
        fig_dir = os.path.join(args.out_dir, "figures")
        plot_comparison_bars(
            img_results["img_s_imp"], img_results["img_ns_imp"],
            audio_results["res_s"]["SI-SDR Imp"], audio_results["res_ns"]["SI-SDR Imp"],
            audio_results["sparsity"], fig_dir,
        )

    print("\n" + "=" * 70)
    print(f"ALL DONE. Figures saved under: {os.path.join(args.out_dir, 'figures')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
