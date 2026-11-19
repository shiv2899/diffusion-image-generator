# Diffusion Image Generator - DDPM Style

AI that learns to clean noise, and then creates new images from pure static.

## Demo

Pure noise se bani 16 nayi digits:

[Generated Digits](results/generated_digits.png)

## How it Works (4 Phases)

- Phase 1: Forward Diffusion - image pe step-by-step noise daalo
- Phase 2: U-Net - noise ko guess karna seekho
- Phase 3: Training - MNIST digits pe train karo
- Phase 4: Sampling - pure noise se nayi digit banao

## How to Run

```bash
pip install -r requirements.txt
cd src
python test_diffusion.py   # Diffusion test
python train.py   # MNIST pe train karo
python sample.py   # Nayi digits banao
```

## Files

- `src/diffusion.py` - Forward + Reverse diffusion
- `src/unet.py` - U-Net, 391,713 parameters, noise predictor
- `src/train.py` - Training loop
- `src/sample.py` - Pure noise se nayi digit banao
- `src/test_diffusion.py` - Diffusion test

## Results

- Phase 1: Forward Diffusion working, t=399 pe image gayab
- Phase 2: U-Net working, 391,713 parameters
- Phase 3: Training working, 5 Epochs, Loss 0.072 -> 0.034, 60,000 MNIST images
- Phase 4: Sampling working, pure noise se 16 nayi digits bani
