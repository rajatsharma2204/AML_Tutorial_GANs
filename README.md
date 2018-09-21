# AML_Tutorial_GANs
This repository contains the code for the tutorial on GANs for the course Advanced Machine Learning
# Introduction
Generative Adversarial Networks (GANs) are machine learning models that solve the problem of generating new data from the probability space of the training data instead of simply classification and regression. GANs consist of two neural networks that compete against one another in a zero-sum game. The two networks are called Generator and Discriminator and their functions are as follows -
## Generator
The task of the generator is to create new data points that resemble closely with the data points of the training data. The generator should be able to generate new points that appear as if they have been sampled from the original distribution. The generator typically takes input from the latent space and generates a data point in the space of training data. The input to the network is a point in latent space that is sampled randomly. The generator typically has deconvolution layers for converting the latent representations into original representations (such as images). The generator tries to increase the loss of the discriminator by fooling discriminator into believing that the data point generated belongs in the original space.
## Discriminator
