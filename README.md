## Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (SiT) — Modified Version

> **⚠️ Notice:** This repository is a **fork/modification of the original [SiT implementation](https://github.com/willisma/SiT)**. It introduces **linear attention** (via LiT) and **long skip connections** (U-ViT style) between Transformer blocks to explore architecture variations, efficiency, and performance trade-offs.

### [Original Paper](https://arxiv.org/pdf/2401.08740.pdf) | [Original Project Page](https://scalable-interpolant.github.io/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/willisma/SiT/blob/main/run_SiT.ipynb)

![SiT samples](visuals/visual.png)

This repo contains a modified PyTorch implementation based on the paper *Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (SiTs)*. In addition to the original baseline, this version incorporates **linear attention** modules and **inter-block skip connections**.

> [**Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**](https://arxiv.org/pdf/2401.08740.pdf)<br>
> [Nanye Ma](https://willisma.github.io), [Mark Goldstein](https://marikgoldstein.github.io/), [Michael Albergo](http://malbergo.me/), [Nicholas Boffi](https://nmboffi.github.io/), [Eric Vanden-Eijnden](https://wp.nyu.edu/courantinstituteofmathematicalsciences-eve2/), [Saining Xie](https://www.sainingxie.com)
> <br>New York University<br>

### Key Modifications in this Repository

1. **Linear Attention Integration:** Replaces standard full self-attention in selected Transformer blocks with LiT linear attention.
2. **Skip Connections (U-ViT style):** Adds residual long skip connections linking shallow Transformer block outputs to deeper block inputs.
3. **Architecture Ablations & Benchmarks:** Built-in options (`baseline`, `linear`, `uvit`, `linear_uvit`) to benchmark and evaluate the impact of linear attention and skip connections independently and combined.

---

## Content Overview

This repository contains:

- 🪐 A modified PyTorch [implementation](models.py) of SiT with support for linear attention and skip connections
- ⚡️ Support for pre-trained class-conditional SiT models trained on ImageNet 256x256
- 🛸 A SiT [training script](train.py) with options for modified variants and backbone freezing
- 📊 Tools for precomputing latent representations, variant verification, and training throughput benchmarking

## Setup

First, download and set up the repo:

```bash
git clone [https://github.com/willisma/SiT.git](https://github.com/willisma/SiT.git)
cd SiT
