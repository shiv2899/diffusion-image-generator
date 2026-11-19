"""Training - MNIST pe U-Net ko noise pehchanna sikhao."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from diffusion import Diffusion
from unet import UNet

def train(n_epochs=5):
    device = "cpu"
    print("MNIST download ho raha hai...")
    # MNIST - 0-9 handwritten digits, 28x28
    tf = transforms.Compose([
        transforms.ToTensor(),  # 0-255 -> 0-1
        transforms.Lambda(lambda x: x * 2 - 1),  # 0-1 -> -1 to +1, Diffusion ke liye
    ])
    ds = datasets.MNIST(root="../data", train=True, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = UNet().to(device)
    diff = Diffusion(n_steps=400, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

    print(f"Training start - {n_epochs} epochs, {len(ds)} images...")

    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for x0, _ in loader:
            x0 = x0.to(device)
            # Har image ke liye random t - 0 se 399 tak
            t = torch.randint(0, 400, (x0.shape[0],), device=device)
            # Noise daalo
            xt, noise = diff.add_noise(x0, t)
            # U-Net guess karega kitna noise hai
            pred_noise = model(xt, t)
            # Loss = asli noise vs guess me gap
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}/{n_epochs} - Loss: {avg:.4f}")

    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "../checkpoints/diffusion_mnist.pt")
    print("Training khatam! Model save: checkpoints/diffusion_mnist.pt")

if __name__ == "__main__":
    train(n_epochs=5)