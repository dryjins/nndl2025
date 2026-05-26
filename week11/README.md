# VAE Lab Guide (Fashion-MNIST)

This document explains how to use the Jupyter notebook for the **Autoencoder (AE) and Variational Autoencoder (VAE)** lab based on the Fashion-MNIST dataset.

The lab has three main learning goals.
- Understand how an autoencoder **compresses** input images into a latent representation and **reconstructs** them.
- Compare plain autoencoders and VAEs, especially in terms of the structure of the **latent space**.
- See why **reconstruction loss**, **KL divergence**, and the **reparameterization trick** are essential components of VAEs.

> This lab is not about "just running the code." The main objective is to interpret the results and connect them to the underlying concepts.

## 1. Files and notebook structure

The notebook is typically organized into the following sections.

1. Dataset loading: Fashion-MNIST download and preprocessing  
2. Plain Autoencoder implementation and training  
3. Autoencoder reconstruction inspection  
4. VAE implementation  
5. VAE training and loss decomposition (reconstruction vs. KL)  
6. Latent space visualization  
7. Prior sampling and interpolation experiments  
8. Discussion questions  

## 2. Learning objectives

By the end of the lab, students should be able to answer the following questions.

### 2.1 What does an autoencoder learn?

- It maps an input image $$x$$ to a low-dimensional latent vector $$z$$ (encoder).  
- It then reconstructs an image $$\hat{x}$$ from $$z$$ (decoder).  
- The model learns a compressed representation that keeps the most important information for reconstruction.

### 2.2 Why is a plain autoencoder not enough?

A plain autoencoder can reconstruct inputs well, but it does not guarantee that the latent space is smooth, structured, or easy to sample from.  
As a result, randomly sampling latent vectors and decoding them does not necessarily yield realistic images.

### 2.3 What does a VAE add?

A VAE replaces deterministic latent points with **probabilistic latent variables**.  
- The encoder outputs a mean $$\mu$$ and a log-variance $$\log \sigma^2$$ for each input.  
- The latent vector is sampled from this Gaussian distribution.  
- A regularization term encourages these posteriors to stay close to a standard normal prior, which shapes a more structured, sample-friendly latent space.

## 3. Environment requirements

Recommended software environment.

- Python 3.10+  
- Jupyter Notebook or JupyterLab  
- PyTorch  
- torchvision  
- matplotlib  
- numpy  
- scikit-learn  

Example installation.

```bash
pip install torch torchvision matplotlib numpy scikit-learn jupyter
```

A GPU is helpful but not strictly required for Fashion-MNIST; CPU-only training is feasible, though more epochs will take longer.

## 4. How to run the notebook

1. Start Jupyter in the project directory.  
2. Open the VAE lab notebook.  
3. Run the cells from top to bottom in order.  
4. After each major block, inspect both numerical outputs (losses) and visual outputs (images, plots).

Example.

```bash
jupyter notebook
```

## 5. Key code-level concepts to watch

### 5.1 Data preprocessing

Fashion-MNIST consists of 28×28 grayscale images of fashion items (e.g., shirts, shoes, bags).  
In this lab, images are typically converted to tensors and either flattened for MLP-based encoders or kept as 2D tensors for CNN-based encoders.

### 5.2 Plain autoencoder pipeline

The basic AE flow is:

```text
x -> encoder -> z -> decoder -> x_hat
```

The objective is to minimize a reconstruction loss between $$x$$ and $$\hat{x}$$, for example mean squared error or binary cross-entropy.

### 5.3 VAE encoder outputs

Instead of a single latent vector, the VAE encoder outputs two quantities.

- $$\mu$$: mean of the approximate posterior  
- $$\log \sigma^2$$: log-variance of the approximate posterior  

Each input is thus mapped to a Gaussian distribution in latent space, not a single point.

### 5.4 Reparameterization trick

Naively sampling from the latent distribution would break standard backpropagation.  
The reparameterization trick expresses sampling as

$$z = \mu + \sigma \odot \epsilon$$  

with $$\epsilon \sim \mathcal{N}(0, I)$$, which keeps the overall computation graph differentiable with respect to $$\mu$$ and $$\sigma$$.

### 5.5 VAE loss formulation

The VAE loss has two components.

1. Reconstruction loss  
   Measures how close the reconstructed image is to the original (e.g., BCE or MSE).  

2. KL divergence  
   Regularizes the approximate posterior toward the standard normal prior.  

A common formulation is

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}}$$  

with $$\beta = 1$$ in the basic VAE, but $$\beta \neq 1$$ in beta-VAE variants.

## 6. What you should actively inspect

These are checkpoints that students should not skip.

### 6.1 Reconstruction quality

- How well does the plain AE reconstruct input images?  
- Are VAE reconstructions slightly blurrier?  
- What trade-off does this illustrate between sharp reconstruction and a regularized latent space?

### 6.2 Latent space structure

- How does the latent space of the plain AE look when visualized (e.g., 2D projection or 2D latent)?  
- Is the VAE latent space more continuous and better aligned with a Gaussian shape?  
- How do different classes occupy this space?  

### 6.3 Sampling behavior

- For a plain AE, decoding random latent vectors often fails to produce realistic-looking images.  
- For a VAE, sampling from the standard normal prior in latent space typically yields plausible fashion images.

### 6.4 Interpolation

Take two latent vectors and linearly interpolate between them.  
If the latent space is well structured, decoded images along the interpolation path should change smoothly rather than producing abrupt artifacts.

## 7. Suggested lab workflow

A recommended sequence for students.

1. Train the plain autoencoder.  
2. Inspect its reconstructions.  
3. Visualize its latent space.  
4. Train the VAE on the same dataset.  
5. Monitor total loss and decomposition into reconstruction vs. KL components.  
6. Inspect VAE reconstructions and prior samples.  
7. Run latent interpolation experiments.  
8. Write down a qualitative comparison between AE and VAE.

## 8. Common conceptual pitfalls

### 8.1 Why use log-variance?

Instead of outputting $$\sigma^2$$ directly, the encoder outputs $$\log \sigma^2$$.  
This helps numerical stability and implicitly enforces positive variance after exponentiation.

### 8.2 Why are VAE reconstructions sometimes blurrier?

The VAE is optimizing both reconstruction and latent regularization.  
It may sacrifice some sharpness in reconstructions to maintain a smooth, well-structured latent space that supports robust sampling.

### 8.3 Interpreting KL loss magnitude

- If KL is very small, the latent variables might not be sufficiently regularized, and the model may behave like a plain AE.  
- If KL is very large, reconstruction quality may collapse; in that case, check learning rate, latent dimensionality, and the weight $$\beta$$.

## 9. Extension ideas

To deepen the lab, consider the following extensions.

- Vary latent dimensionality (e.g., 2, 8, 16, 32) and compare results.  
- Implement beta-VAE by changing $$\beta$$.  
- Replace MLP encoder/decoder with CNN versions.  
- Swap Fashion-MNIST with MNIST or KMNIST and compare behaviors.  
- Discuss conditional generation (e.g., class-conditional VAE).  
- Analyze the trade-off between reconstruction quality and generative quality.

## 10. Questions for reports or presentations

After completing the lab, students should be able to answer:

1. How does an autoencoder perform representation learning?  
2. What are the main limitations of a plain autoencoder as a generative model?  
3. What do $$\mu$$ and $$\log \text{var}$$ represent in a VAE?  
4. Why is the reparameterization trick needed?  
5. What roles do reconstruction loss and KL divergence play in the VAE objective?  
6. Why is the VAE latent space better suited for sampling and interpolation than that of a plain AE?

## 11. Teaching tips

From an instructional perspective, a useful flow is:

- First, show autoencoder reconstructions and ask: "Why is this not yet a good generative model?"  
- Then introduce the VAE as a move from deterministic to probabilistic latent representations.  
- Finally, emphasize that evaluation should include **latent structure** and **sampling behavior**, not just reconstruction loss.

## 12. Limitations and caveats

This lab focuses on conceptual understanding of VAEs, not on state-of-the-art generative performance.  
Fashion-MNIST is a low-resolution, relatively simple dataset, so conclusions should be interpreted in a didactic, not production, context.

The provided implementation is intentionally simplified for teaching purposes. For example:  
- Shallow model architectures  
- Limited number of training epochs  
- Simplified likelihood modeling  
- No explicit techniques to avoid posterior collapse  

Students should be made aware that modern generative models use more advanced architectures and training schemes.

## 13. End-of-lab checklist

By the end of the lab, students should be able to check off:

- [ ] Ran both plain AE and VAE training.  
- [ ] Compared reconstruction results between the two models.  
- [ ] Decomposed VAE loss into reconstruction and KL terms and interpreted both.  
- [ ] Visualized the latent space.  
- [ ] Generated samples from the prior.  
- [ ] Performed latent interpolation.  
- [ ] Explained, in their own words, the conceptual differences between AE and VAE.
