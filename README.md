# CelebA Face Generation with Generative Adversarial Networks

This repository contains a PyTorch implementation of a **Vanilla Generative Adversarial Network (GAN)** trained on the CelebA (Celebrity Attributes) dataset. The goal of the project is to explore the fundamentals of deep generative modeling by training a Generator to synthesize realistic $64 \times 64$ RGB human faces from latent random noise, while concurrently training a Discriminator to distinguish real dataset images from synthetic ones.

## 🛠️ Technical Stack & Architecture
* **Framework:** PyTorch & Torchvision
* **Dataset:** CelebA (aligned and cropped)
* **Data Pipeline:** Custom PyTorch `ImageProcessor` (Dataset/DataLoader wrapper) executing a center crop to $178 \times 178$, downsampling to $64 \times 64$, tensor conversion, and $[-1, 1]$ normalization.
* **Generator:** A fully connected (dense) sequential network expanding a 100-dimensional latent vector ($z$) through hidden layers (256 $\rightarrow$ 512 $\rightarrow$ 1024 nodes) with ReLU activations, outputting a reshaped 3-channel image tensor via a `Tanh` activation function.
* **Discriminator:** A fully connected network that flattens the input image tensor and passes it through decreasing dense layers with `LeakyReLU` activations ($0.2$ slope). A final `Sigmoid` layer outputs the scalar probability of the image being real.
* **Optimization:** Binary Cross-Entropy Loss (`BCELoss`) optimized via the Adam optimizer with a learning rate of $0.0002$ and hyperparameter $\beta_1 = 0.5$ to stabilize adversarial training.

## 📊 Project Status & Limitations
The baseline model achieves stable loss convergence across both adversarial networks. Because vanilla GANs rely entirely on dense, fully connected layers, the spatial features of the color images are structural but inherently blurry. This project serves as an ideal baseline benchmark for adversarial training mechanics before transitioning to more complex, spatially-aware architectures.
