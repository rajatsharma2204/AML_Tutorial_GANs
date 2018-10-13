import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, concatenate, MaxoutDense, Embedding
from keras.datasets import mnist
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
(X_Train, Y_Train), _ = mnist.load_data()
Y_one_hot = to_categorical(Y_Train, 10)
X_Train = X_Train/127.5 - 1.0
class GAN:
        def __init__(self):
                self.latent_space_dim = 100
                self.batch_size = 100
                self.num_classes = 10
                        
                gen_noise_input = Input(shape=(self.latent_space_dim,))
                gen_noise_dense = Dense(256, activation='relu')(gen_noise_input)
                
                gen_label_input = Input(shape=(10,))
                gen_label_dense = Dense(1024, activation='relu')(gen_label_input)
                
                gen_merged = concatenate([gen_noise_dense, gen_label_dense]) 
                gen_combined_dense1 = Dense(784, activation='tanh')(gen_merged)
                gen_result = Reshape((28, 28))(gen_combined_dense1)
                
                self.Generator = Model(inputs=[gen_noise_input, gen_label_input], outputs=gen_result)
                
                dis_img_input = Input(shape=(28, 28))
                dis_img_flat = Flatten()(dis_img_input)
                dis_img_dense = MaxoutDense(240, 5)(dis_img_flat)
                
                dis_label_input = Input(shape=(10,))
                dis_label_dense = MaxoutDense(50, 5)(dis_label_input)
                
                dis_merged = concatenate([dis_img_dense, dis_label_dense])       
                dis_result = Dense(1, activation='sigmoid')(dis_merged)
                
                self.Discriminator = Model(inputs=[dis_img_input, dis_label_input], outputs=dis_result)
                self.Discriminator.compile(loss = "binary_crossentropy", optimizer = Adam(0.0002, 0.5), metrics = ["accuracy"])
                self.Discriminator.trainable = False
                
                noise = Input(shape=(self.latent_space_dim,))
                label = Input(shape=(10,))
                img = self.Generator([noise, label])
                score = self.Discriminator([img, label])
                self.Cascaded_model = Model([noise, label], score)                
                self.Cascaded_model.compile(loss = "binary_crossentropy", optimizer = Adam(0.0002, 0.5))
                
        def train_one_epoch(self):
                for i in range(3):
                    sampled_x_indices = np.random.randint(0, X_Train.shape[0], self.batch_size)
                    sampled_x = X_Train[sampled_x_indices, :, :]
                    sampled_y = Y_one_hot[sampled_x_indices, :]
                    sampled_Gz = self.Generator.predict([np.random.normal(0, 1, (self.batch_size, self.latent_space_dim)), sampled_y])
                    
                    #all_samples contains both real and generated samples.
                    all_samples = np.zeros((self.batch_size * 2, 28, 28))                    
                    all_samples[0:self.batch_size, :, :] = sampled_x
                    all_samples[self.batch_size:self.batch_size * 2, :, :] = sampled_Gz
                    
                    ground_truth = np.zeros((self.batch_size * 2))
                    ground_truth[0:self.batch_size] = np.ones((self.batch_size))
                    
                    (discriminator_loss, discriminator_acc) = self.Discriminator.train_on_batch([all_samples, np.concatenate((sampled_y, sampled_y))], ground_truth)
                                   
                sampled_z = np.random.normal(0, 1, (self.batch_size, self.latent_space_dim))
                sampled_labels = to_categorical(np.random.randint(0, self.num_classes, (self.batch_size, 1)), self.num_classes)
                discriminator_loss_after_train_gen = self.Cascaded_model.train_on_batch([sampled_z, sampled_labels], np.ones((self.batch_size)))
                                
        def train(self, epochs):
              for epoch in range(epochs):
                    self.train_one_epoch()
                    if epoch % 200 == 0:
                          print "Epoch: " + str(epoch) + " completed"
                          noise = np.random.normal(0, 1, (100, self.latent_space_dim))
                          labels = to_categorical(np.tile(np.arange(self.num_classes), 10), 10)
                          sampled_Gz = self.Generator.predict([noise, labels]) * 0.5 + 0.5
                          complete_image = np.ones((28 * 10 + 9, 28 * 10 + 9, 1))
                          for i in range(0, 10):
                                for j in range(0, 10):
                                      complete_image[29 * i:29 * i + 28, 29 * j:29 * j + 28, 0] = sampled_Gz[i * 10 + j, :, :]
                          cv2.imwrite("images/ " + str(epoch) + ".jpg", cv2.resize(complete_image * 255.0, (0, 0), fx = 3, fy = 3))
GAN_Model = GAN()
GAN_Model.train(30001)
