"""U-Net - Noise ko guess karne wala dimaag."""

import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    """t = 0-399 ko number se vector me badlo, taaki U-Net ko pata chale kitna noise hai."""
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t):
        # Sinusoidal - Transformer wala idea
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        emb = torch.relu(self.fc1(emb))
        emb = self.fc2(emb)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = torch.relu(self.norm1(self.conv1(x)))
        # Time ka info add karo - har channel pe
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = torch.relu(self.norm2(self.conv2(h)))
        return h

class UNet(nn.Module):
    """Chhota U-Net - 28x28 MNIST ke liye."""
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.time_emb = TimeEmbedding(32)

        # Down - Image ko chhota karo, feature nikalo
        self.down1 = Block(in_ch, base)
        self.pool1 = nn.Conv2d(base, base, 3, stride=2, padding=1)  # 28 -> 14
        self.down2 = Block(base, base * 2)
        self.pool2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)  # 14 -> 7

        # Middle
        self.mid = Block(base * 2, base * 2)

        # Up - Wapas bada karo, skip connection se detail wapas lao
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 4, stride=2, padding=1)  # 7 -> 14
        self.up_block2 = Block(base * 4, base)
        self.up1 = nn.ConvTranspose2d(base, base, 4, stride=2, padding=1)  # 14 -> 28
        self.up_block1 = Block(base * 2, base)

        self.out = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        d1 = self.down1(x, t_emb)
        p1 = self.pool1(d1)
        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)

        m = self.mid(p2, t_emb)

        u2 = self.up2(m)
        u2 = torch.cat([u2, d2], dim=1)  # Skip - purani detail wapas
        u2 = self.up_block2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1, t_emb)

        return self.out(u1)

if __name__ == "__main__":
    net = UNet()
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([10, 200])
    out = net(x, t)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
    print(f"Total Parameters: {sum(p.numel() for p in net.parameters())}")
    print("U-Net sahi chal raha hai!")