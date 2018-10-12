---
layout: page
title : Improved Techniques for Training GANs
permalink: /improvedtraining/
ordinal: 7
---

## Convergent GAN Training

For many scenarios it has been seen that traditional techniques for training GANs do not perform that well and may enter an infinite loop (like an orbit around the original solution), so few techniques that heuristically encourage convergence are described below:

### Feature Matching

We replace the objective function of the generator to prevent overtraining discriminator. Instead of directly maximizing the output of discriminator we train the generator to match the expected value of features on an intermediate layer of the discriminator

Let $f(x)​$ be the activation of intermediate layer of the discriminator, then our new objective function for generator becomes $$\vert\vert{\mathbb E}_{x\sim p_{\rm data}}f(x)-{\mathbb E}_{z\sim p(z)}f(G(z))\vert\vert_2^2​$$.

### Minibatch Discrimination

One main failure of GANs is when generator keeps generating same point (example). When this thing is going to happen the discriminator provides gradients that propels the generators outputs towards a point that it believes to be quite similar to real data and after that makes it move around the point forever, thus resulting in no learning. This is because discriminator processes each point independently and cannot gauge the coordination between the gradients of different points which results in no way t tell that the outputs of the generator should become diverse. To solve this problem we need the discriminator to look at multiple data points in combination which is called “minibatch discrimination”.

- Let $f(x_i)\in\mathbb R^A$ denote vector of features for input $x_i$ by an intermediate layer in the discriminator.
- We then multiply this by tensor $T\in\mathbb R^{A\times B\times C}$ which gives us $M_i\in\mathbb R^{B\times C}$
- We then compute $L_1$ distance between the rows of the resulting matrix $M_i$ across samples $i\in\{1,2,\ldots n\}$ and apply a negative exponential $c_b(x_i, x_j)=\exp (-\vert\vert M_{i,b}-M_{j, b}\vert\vert_{L_1})\in {\mathbb R}$ 
- The output $o(x_i)_b=\sum\limits\_{j=1}^nc_b(x_i, x_j)\in {\mathbb R}$ for the minibatch layer for a sample $x_i$
- The output for the layer is $o(x_i)=[o(x_i)_1,o(x_i)_2,\ldots,o(x_i)_B]\in\mathbb R^B​$
- This is concatenated with $f(x_i)$ and fed to next layer of discriminator that helps as side information.

### Historical Averaging

In this technique we add a cost term to discriminator and generator $\left \vert\left\vert\theta - \frac 1t\sum_{i=1}^t\theta[i]\right\vert\right\vert^2​$ where $\theta[i]​$ is the value of parameters at past time $i​$. This approach was found to be able to find equilibria of low-dimensional, continuous non-convex cases.

### One Sided Label Smoothing

In this approach we replace $0$ and $1$ by smoother values like $0.9$ and $0.1$. We replace positive classification targets with $\alpha$ and negative targets with $\beta$, which results in the optimal discriminator becoming:

$$D(x)=\frac{\alpha p_{\rm data}(x)+\beta p_{\rm model}(x)}{p_{\rm data}(x)+p_{\rm model}(x)}$$

Here $p_{\rm model}$ in numerator is problematic because where $p_{\rm data}\sim 0$ and $p_{\rm model }\gg1$ then erroneous samples from $p_{\rm model}$ would not move towards data. So we only smooth the positive labels to $\alpha$ and leave negative labels at $0$.

### Virtual Batch Normalization

Batch normalization is quite successful in improving the performance of neural networks and works quite well for DCGANs, but it makes the output of a neural network for $x$ to be dependent on other $x’$in the same batch. So we use “Virtual Batch Normalization” where we normalize based on a reference batch fixed at the start of the training and on $x$. VBN is computationally expensive because we need forward propagation on 2 minibatches so we use it only on generator network.

## Assessment of Image Quality

To get an overall objective function for GAN which is usually done by human annotation we need an automatic method to evaluate samples. 

- We apply the inception model to every generated image to get the conditional label distribution $p(y\mid x)$. 
  - Meaningful objects containing images will have $p(y\mid x)​$ with low entropy.
  - We require varied images so we have $\int p\left(y\mid x=G(z)\right)dz$ should have high entropy.
- So the combined metric is $\exp({\mathbb E}_x{\rm KL}(p(y\mid x)\vert\vert p(y)))$ where $\rm KL$ is KL-divergence.