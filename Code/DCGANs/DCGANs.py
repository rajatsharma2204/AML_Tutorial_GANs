import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape, Flatten, LeakyReLU, UpSampling2D, BatchNormalization, Convolution2D
from keras.datasets import mnist
from keras.optimizers import Adam
(X_Train, Y_Train), (X_Val, Y_Val) = mnist.load_data()
X_Train = X_Train/127.5 - 1.0
class GAN:
        def __init__(self):
                self.latent_space_dim = 100
                self.batch_size = 200
                
                self.Generator = Sequential()
                self.Generator.add(Dense(256 * 7 * 7, input_dim = self.latent_space_dim, activation = "relu"))
                self.Generator.add(Reshape((7, 7, 256)))
                self.Generator.add(Dropout(0.25))
                self.Generator.add(UpSampling2D())
                self.Generator.add(Convolution2D(128, (3, 3), padding = "same"))
                self.Generator.add(BatchNormalization())
                self.Generator.add(Activation('relu'))
                self.Generator.add(Dropout(0.25))
                self.Generator.add(UpSampling2D())
                self.Generator.add(Convolution2D(64, (3, 3), padding = "same"))
                self.Generator.add(BatchNormalization())
                self.Generator.add(Activation('relu'))
                self.Generator.add(Dropout(0.25))
                self.Generator.add(Convolution2D(32, (3, 3), padding = "same"))
                self.Generator.add(BatchNormalization())
                self.Generator.add(Activation('relu'))
                self.Generator.add(Convolution2D(1, (3, 3), padding = "same", activation = "tanh"))
                
                self.Discriminator = Sequential()
                self.Discriminator.add(Convolution2D(64, (3, 3), strides = 2, input_shape = (28, 28, 1)))
                self.Discriminator.add(BatchNormalization())
                self.Discriminator.add(LeakyReLU(alpha = 0.1))
                self.Discriminator.add(Dropout(0.25))
                self.Discriminator.add(Convolution2D(128, (3, 3), strides = 2))
                self.Discriminator.add(BatchNormalization())
                self.Discriminator.add(LeakyReLU(alpha = 0.1))
                self.Discriminator.add(Dropout(0.25))
                self.Discriminator.add(Convolution2D(256, (3, 3)))
                self.Discriminator.add(BatchNormalization())
                self.Discriminator.add(LeakyReLU(alpha = 0.1))
                self.Discriminator.add(Flatten())
                self.Discriminator.add(Dense(1, activation = "sigmoid"))
                self.Discriminator.compile(loss = "binary_crossentropy", optimizer = Adam(0.0002, 0.5), metrics = ["accuracy"])
                
                inp_gen_tensor = Input(shape = (self.latent_space_dim,))
                out_gen_model = Model(inp_gen_tensor, self.Generator(inp_gen_tensor))
                inp_dis_tensor = Input(shape = (28, 28, 1))
                out_dis_model = Model(inp_dis_tensor, self.Discriminator(inp_dis_tensor))
                out_dis_model.trainable = False
                inp_tensor = Input(shape = (self.latent_space_dim,))
                self.Cascaded_model = Model(inp_tensor, out_dis_model(out_gen_model(inp_tensor)))                
                self.Cascaded_model.compile(loss = "binary_crossentropy", optimizer = Adam(0.0002, 0.5))
        def train_one_epoch(self):
                for i in range(1):
                    sampled_x = X_Train[np.random.randint(0, X_Train.shape[0], self.batch_size), :, :]
                    sampled_Gz = self.Generator.predict(np.random.normal(0, 1, (self.batch_size, self.latent_space_dim)))
                    #all_samples contains both real and generated samples.
                    all_samples = np.zeros((self.batch_size * 2, 28, 28, 1))
                    all_samples[0:self.batch_size, :, :, 0] = sampled_x
                    all_samples[self.batch_size:self.batch_size * 2, :, :, :] = sampled_Gz
                    ground_truth = np.zeros((self.batch_size * 2))
                    ground_truth[0:self.batch_size] = np.ones((self.batch_size))
                    (discriminator_loss, discriminator_acc) = self.Discriminator.train_on_batch(all_samples[0:self.batch_size, :, :, :], ground_truth[0:self.batch_size])
                    (discriminator_loss, discriminator_acc) = self.Discriminator.train_on_batch(all_samples[self.batch_size:self.batch_size * 2, :, :, :], ground_truth[self.batch_size:self.batch_size * 2])
                
                sampled_z = np.random.normal(0, 1, (self.batch_size, self.latent_space_dim))
                discriminator_loss_after_train_gen = self.Cascaded_model.train_on_batch(sampled_z, np.ones((self.batch_size)))
        def train(self, epochs):
              for epoch in range(epochs):
                    self.train_one_epoch()
                    if epoch % 200 == 0:
                          print "Epoch: " + str(epoch) + " completed"
                          sampled_Gz = self.Generator.predict(np.random.normal(0, 1, (25, self.latent_space_dim))) * 0.5 + 0.5
                          complete_image = np.ones((28 * 5 + 4, 28 * 5 + 4, 1))
                          for i in range(0, 5):
                                for j in range(0, 5):
                                      complete_image[29 * i:29 * i + 28, 29 * j:29 * j + 28, :] = sampled_Gz[i * 5 + j, :, :, :]
                          cv2.imwrite("images/ " + str(epoch) + ".jpg", cv2.resize(complete_image * 255.0, (0, 0), fx = 3, fy = 3))
GAN_Model = GAN()
GAN_Model.train(20001)
