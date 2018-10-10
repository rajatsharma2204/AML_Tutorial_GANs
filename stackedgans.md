---
layout: page
title : Stacked GANs
permalink: /stackedgans/
ordinal: 3
---

Stacked GANs are top-down stack of GANs, each trained to generate “plausible” lower-level representations conditioned on higher-level representations.

## Pre-trained encoder

A bottom up DNN pre-trained for classification is referred to as the encoder $E$. Each DNN entails a mapping $h_{i+1}=E_i(h_i)$, where $i\in\{0,1,\ldots, N-1\}$

## Stacked Generators

 Our goal is to train a top-down generator $G$ that inverts $E$. $G$ consists of a top-down stack of generators $G_i$, each trained to invert a bottom up mapping $E_i$. Each takes in a higher-level feature and a noise vector as inputs, and outputs the lower-level feature $h_i$.

We first train each GAN independently and then train them jointly in an end-to-end manner.

Each generator receives conditional input from encoders in the independent training stage, and from the upper generators in the joint training stage. In other words, $\hat h_i=G_i(h_{i+1}, z_i)$ during independent training and  $h_i=G_i(\hat h_{i+1}, z_i)$ during joint training. The loss equations are for independent training stage but can be easily modified to joint training by replacing  $h_{i+1}$ with $\hat h_{i+1}$ Intuitively, the total variations of images could be decomposed into multiple levels, with higher-level semantic variations (e.g., attributes, object categories, rough shapes) and lower-level variations (e.g., detailed contours and textures, background clutters).

![](/images/stackGAN.png)

This allows using different noise variables to represent different levels of variations. The training procedure is shown in Figure. Each generator  is trained with a linear combination of three loss terms: adversarial loss, conditional loss, and entropy loss with different parametric weights.  
$${\cal L}_{G_i}=\lambda {\cal L}_{G_i}^{adv}+{\cal L}_{G_i}^{cond}+{\cal L}_{G_i}^{ent}$$

For each generator $G_i$ we have a representation discriminator $D_i$ that distinguishes generated representations $\hat h_i$ from real representations $h_i$ . $D_i$ is trained with the loss function:

$$\displaystyle {\cal L}_{D_i}={\mathbb E}_{h_i\sim P_{data, E}}[-\log(D_i(h_i))] + {\mathbb E_{z_i\sim P_{z_i}, h_{i+1}\sim P_{data, E}}}[-\log(1-D_i(G_I(h_{i+1}, z)))]$$

And  is trained to fool the representation discriminator  with the adversarial loss defined by:

$$\displaystyle {\cal L}_{G_i}^{adv}={\mathbb E}_{h_{i+1}\sim P_{data, E}, z_i\sim P_{z_i}}[-\log(D_i(G_i(h_{i+1}, z)))]$$

During joint training, the adversarial loss provided by representational discriminators can also be regarded as a type of deep supervision, providing intermediate supervision signals. In our current formulation, $E$ is a discriminative model, and $G$ is a generative model conditioned on labels. However, it is also possible to train SGAN without using label information: $E$ can be trained with an unsupervised objective and $G$ can be cast into an unconditional generative model by removing the label input from the top generator.

## Sampling

To sample images, all $G_i$s are stacked together in a top-down manner, as shown in Figure. We have the data distribution conditioned on the class label: 

$$ \displaystyle p_G(\hat x\mid y)=p_G(\hat h_0\mid\hat h_N)\propto p_G(\hat h_0, \hat h_1,\ldots, \hat h_{N-1}\mid \hat h_N)=\prod\limits_{0\le i\le N-1}p_{G_i}(\hat h_i\mid \hat h_{i+1})$$

where each $p_{G_i}(\hat h_i\mid \hat h_{i+1})$ is modeled by a generator  $G_i$ . From an information-theoretic perspective, SGAN factorizes the total entropy of the image distribution $H(x)$ into multiple (smaller) conditional entropy terms:  $H(x)=H(h_0, h_1, \ldots, h_N)=\sum\limits_{i=0}^{N-1}H(h_i\mid h_{i+1})$, thereby decomposing one difficult task into multiple easier tasks.

## Conditional Loss

At each stack, a generator  $G_i$ is trained to capture the distribution of lower-level representations $\hat h_i$ , conditioned on higher-level representations $h_{i+1}$. However, in the above formulation, the generator might choose to ignore $h_{i+1}$, and generate plausible  $\hat h_i$ from scratch. We regularize the generator by adding a loss term ${\cal L}_{G_i}^{cond}$. We feed the generated lower-level representations  back to the encoder $E$ , and compute the recovered higher-level representations. We then enforce the recovered representations to be similar to the conditional representations. Formally: $${\cal L}_{G_i}^{cond}={\mathbb E}_{h_{i+1}\sim p_{\rm data}, z_i\sim p_{z_i}}[f(E_i(G_i(h_{i+1}, z_i)), h_{i+1})]$$ where $f$ is an Euclidean distance measure and crossentropy for labels.

## Entropy Loss

Simply adding the conditional loss $${\cal L}_{G_i}^{cond}$$ leads to another issue: the generator $G_i$ learns to ignore the noise $z_i$, and compute  $\hat h_i$ deterministically from $h_{i+1}$ . To tackle this problem we would like to encourage the generated representation  to be sufficiently diverse when conditioned on $h_{i+1}$, i.e., the conditional entropy  $H(\hat h_i\mid h_{i+1})$ should be as high as possible. Since directly maximizing $H(\hat h_i\mid h_{i+1})$ is intractable, we propose to maximize instead a variational lower bound on the conditional entropy. Specifically, we use an auxiliary distribution $Q_i(z_i\mid \hat h_i)$ to approximate the true posterior  $p_i(z_i\mid \hat h_i)$, and augment the training objective with a loss term named entropy loss:  $${\cal L}^{ent}={\mathbb E}_{z_i\sim p_{z_i}}[{\mathbb E}_{\hat h_i\sim G_i(\hat h_i\mid z_i)}[-\log Q_i(z_i\mid \hat h_i)]]$$ . It can be proved that minimizing this is equivalent to maximizing a variational lower bound for  $H(\hat h_i\mid h_{i+1})$ . In practice we parametrize  $Q_i$ with a deep network that predicts the posterior distribution of $z_i$ given $\hat h_i$ . $Q_i$ shares most of the parameters with $D_i$ . We treat the posterior as a diagonal Gaussian with fixed standard deviations, and use the network  to only predict the posterior mean, making $${\cal L}_{G_i}^{ent}$$ equivalent to the Euclidean reconstruction error. In each iteration we update both  $G_i$ and $Q_i$ to minimize ${\cal L}_{G_i}^{ent}$

## References 

- [Huang, Xun, et al. "Stacked Generative Adversarial Networks." *CVPR*. Vol. 2. 2017.](https://arxiv.org/pdf/1606.03657.pdf)