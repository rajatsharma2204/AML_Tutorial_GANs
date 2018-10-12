---
layout: page
title : Spectral Normalization for GANs
permalink: /spectralnormalization/
ordinal: 9
---

The training of the discriminator amounts to the training of a good estimator for the density ratio between the model distribution and the target which is also employed in implicit models for variational optimization.

In high dimensional spaces, 

-  the density ratio estimation by the discriminator is often inaccurate & unstable during the training
- generator networks fail to learn the multimodal structure of the target distribution 

Moreover, when the support of the model distribution & of the target are disjoint, we can have a perfect discriminator, which results in halting of the training of the generator.

So we need a better weight normalization method that can stabilize the training of discriminator networks, which is what we will study now “spectral normalization”.

## Basics

Lets us have a simple neural network with input $x$ that gives output $f(x, \theta)$ where $\theta = \{W^1, W^2, ... W^{L+1}\}$ is the set of weights and $a_i$ are the non-linear activation functions

$$f(x,\theta)=W^{L+1}a_L(W^L(a_{L-1}(W^{L-1}(\ldots a_1(W^1x)\ldots))))$$

Here we have omitted the bias terms. Discriminator gives:

$$D(x,\theta)={\cal A}(f(x, \theta))$$

where $\cal A$ is the activation function corresponding to the divergence of distance measure. In GAN we will have:

$$\min\limits_G\max\limits_DV(G,D)$$

with $V(G, D)$ as described in the introduction section. For a fixed generator G, the optimal discriminator is known to be $\displaystyle D_G^*(x)=\frac{q_{\rm data}(x)}{q_{data}(x)+p_G(x)}$  which can be written as:

$$D_G^*(x)={\rm sigmoid}(f^*(x))\text{, where }f^*(x)=\log q_{\rm data}(x)-\log p_G(x)$$

whose derivate is:

$$\nabla_xf^*(x)=\frac1{q_{\rm data}(x)}\nabla_xq_{data}(x)-\frac1{p_G(x)}\nabla_xp_G(x)$$

which can be unbounded or incomputable. So we need to add some regularize condition to derivate of $f(x)$. One successful approach was to control the Lipschitz constant of the discriminator by adding regularization terms. Spectral normalization is based on a similar approach.

## Spectral Normalization

**Spectral Norm:** is the largest singular value in a matrix $A$ and we can think of it as the largest amount by which it can scale a value, i.e.

$$\sigma(A)=\max\limits_{h:h\neq0}\frac{||Ah||_2}{||h||_2}=\max\limits_{||h||_2\le 1}||Ah||_2$$

Here $h$ are the values that are used to find the scaling.

In spectral normalization we constrain the spectral norm of each layer $g:h_{in}\mapsto h_{out}$ 

**Lipschitz Norm:** $$||g||_{\rm Lip} = \sup_h\sigma(\nabla g(h))$$ is the supremum of the spectral norm of gradient of $g$

- For a linear layer we have $||g||_{\rm Lip}=\sup_h\sigma(W)=\sigma (W)$
- We assume that the Lipschitz norm of the activation function $||a_l||_{\rm Lip}=1$
- We have $||g_1\circ g_2||_{\rm Lip}\le ||g_1||_{\rm Lip}\cdot ||g_2||_{\rm Lip} $

So we have:

$$\begin{align*}||f||_{\rm Lip}\le &||(h_L\mapsto W^{L+1}h_L)||_{\rm Lip}\cdot &||a_L||_{\rm Lip}\\\cdot &||(h_{L-1}\mapsto W^Lh_{L-1})||_{\rm Lip}\cdot&||a_{L-1}||_{\rm Lip}\\\cdot&\ldots\\\cdot&||(h_0\mapsto W^1h_0)||_{\rm Lip}\\=&\prod_{l=1}^{L+1}||(h_{l-1}\mapsto W^lh_{l-1}||_{\rm Lip}\\=&\prod_{l=1}^{L+1}\sigma(W^l)\end{align*}​$$

Finally we normalize the spectral norm of the weight matrix $W$ so that we get $\sigma(W)=1$, i.e.

$$\displaystyle \bar W_{SN}(W)=\frac W{\sigma(W)}$$

We can normalize each weight using this and we will finally get that $||f||_{\rm Lip}\le 1$

**Note:** If we use SVD then it would become very inefficient to compute $\sigma(W)$, instead we can use the power iteration method.

## References

- [Miyato, Takeru, et al. "Spectral normalization for generative adversarial networks." *arXiv preprint arXiv:1802.05957* (2018).](https://arxiv.org/abs/1802.05957)

