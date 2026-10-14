# Denoising Autoencoder

**A TensorFlow convolutional autoencoder for reconstructing cleaner images from noisy inputs.**

This project demonstrates the core encoder-decoder pattern used in image restoration. The training starter loads CIFAR-10, adds synthetic noise and learns to reconstruct the clean image.

## Learning objective

The model compresses an image into a smaller representation and then reconstructs it. Training compares the reconstruction with the clean target using mean squared error.

```text
clean image → add noise → encoder → bottleneck → decoder → reconstructed image
                                                     ↘ compare with clean image
```

## Features

- CIFAR-10 dataset loader.
- Synthetic Gaussian noise generation.
- Compact convolutional encoder and decoder.
- Keras training and evaluation loop.
- Simple baseline suitable for experimentation.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 train.py
```

The first run downloads CIFAR-10 through Keras.

## Project structure

```text
train.py          # dataset, model and short training run
requirements.txt  # TensorFlow dependency
README.md         # documentation
```

## Roadmap

- Save side-by-side noisy and reconstructed image grids.
- Compare MSE and SSIM across noise levels.
- Add skip connections with a U-Net-style architecture.
- Track experiments with reproducible configuration files.
- Add a small web demo for image upload.

## Limitations

The default script is intentionally a short demo run and is not a benchmark. Production image restoration needs curated data, longer training, perceptual evaluation and careful handling of image privacy.

## Tech stack

Python 3 · TensorFlow · Keras · CNN autoencoder

## Author

Built by **Tallal Kiani**.
