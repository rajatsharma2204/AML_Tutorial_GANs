---
layout: page
title : Info GANs
permalink: /infogans/
ordinal: 2
---

InfoGANs are a variation of GANs. The problem they try to resolve is

## Problem with GANs that InfoGANs try to resolve

The noise vectors that are taken as inputs in the generator are unstructured and contain no semantic information, i.e. they are completely random variables. However, most datasets contain some semantic features (such as in case of MNIST digits dataset, the number being represented, the thickness of the stroke, angle of digit etc). If these semantic features were an input to GANs, then the final output of the GANs would be more meaningful and easy to understand.

## Approach

InfoGANs use a set of structured latent variables $c_1, c_2, \ldots, c_L$ that are responsible for encoding the semantic features in the dataset. So, now the GANs take two inputs, an unstructured noise vector $z$ and a set of structured vectors (together denoted as $c$). However, the generator might learn to ignore c completely. So, somehow it has to be factored in the function that the generator would try to minimize.

For this, mutual information between $c$ and $G(z, c)$ is used. If the vectors in $c$ are very closely related with the output, then the generator is not ignoring $c$. So, increasing mutual information would mean that the generator has to consider the c and can't ignore it.
So, the following function is used instead of the regular GAN function -

$$\displaystyle \min\limits_G \max\limits_D V_I(D, G)  = V(D, G) - \lambda I(c; G(z, c))$$

where $I(c; G(z, c))$ is the mutual information between c and the output of the generator.
So, if there is a correlation, the function that the GAN is trying to decrease decreases due to high correlation.

## Advantages

This leads to finding vectors c that directly influence the output.

![]({{site.baseurl}}/images/infoGANs.png)

Here, varying

- $c_1$ on InfoGAN leads to change in digit type (from left to right).
- $c_2$ on InfoGAN leads to change in the rotation of the digit.
- $c_3$ on InfoGAN leads to change in the width of the digit.

## References

- [Chen, Xi, et al. "Infogan: Interpretable representation learning by information maximizing generative adversarial nets." *Advances in neural information processing systems*. 2016.](https://arxiv.org/pdf/1606.03657.pdf)