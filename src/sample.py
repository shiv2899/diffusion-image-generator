"""Sampling - Pure shor se nayi MNIST digit banao."""

import torch
import matplotlib.pyplot as plt
import os
from diffusion import Diffusion
from unet import UNet

def sample(n_images=16):
    device = "cpu"
    print("Trained model load ho raha hai...")

    model = UNet().to(device)
    model.load_state_dict(torch.load("../checkpoints/diffusion_mnist.pt", map_location=device))
    model.eval()

    diff = Diffusion(n_steps=400, device=device)

    # 1. Pure shor se start karo - jaise TV ka static
    x = torch.randn(n_images, 1, 28, 28).to(device)
    print("Pure shor se start - ab 400 step me saaf karenge...")

    # 2. Ulta chalao - 399 se 0 tak, thoda-thoda noise hatao
    for t_step in reversed(range(400)):
        t = torch.full((n_images,), t_step, device=device, dtype=torch.long)
        x = diff.sample_step(model, x, t)
        if t_step % 100 == 0:
            print(f"Step {t_step}/400 bacha hai...")

    # 3. -1 to +1 se 0 to 1 me wapas lao, photo banane ke liye
    x = (x.clamp(-1, 1) + 1) / 2

    # 4. 4x4 grid me dikhao aur save karo
    os.makedirs("../results", exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i, 0].cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("../results/generated_digits.png", dpi=150)
    print("Ho gaya! Save ho gaya: results/generated_digits.png")
    plt.show()

if __name__ == "__main__":
    sample(n_images=16)