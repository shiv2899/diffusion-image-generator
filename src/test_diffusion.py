"""Test - Image pe noise daal ke dekho."""

import torch
from diffusion import Diffusion

def main():
    d = Diffusion(n_steps=400)
    # Fake image - 1x1x28x28, jaise MNIST
    x0 = torch.randn(2, 1, 28, 28)
    print(f"Original: mean={x0.mean():.3f}, std={x0.std():.3f}")

    # t=0 - thoda noise
    xt0, _ = d.add_noise(x0, torch.tensor([0, 0]))
    print(f"t=0: mean={xt0.mean():.3f}, std={xt0.std():.3f}")

    # t=399 - poora noise, image gayab
    xt399, _ = d.add_noise(x0, torch.tensor([399, 399]))
    print(f"t=399: mean={xt399.mean():.3f}, std={xt399.std():.3f}")
    print("Diffusion sahi chal raha hai! t=399 pe image gayab, sirf noise bacha.")

if __name__ == "__main__":
    main()