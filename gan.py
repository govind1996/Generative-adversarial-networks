# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

np.random.seed(10)

random_dim=100

def load_minst_data():
    (xtrain,ytrain),(xtest,ytest)=mnist.load_data()
    xtrain=(xtrain.astype(np.float32)-127.5)/127.5
    xtrain=xtrain.reshape(60000,784)
    return (xtrain,ytrain,xtest,ytest)
    


def get_optimiser():
    return Adam(0.0002,0.5)


def get_Generator(optimiser):
    generator=Sequential()
    generator.add(Dense(256,input_dim=100,kernel_initializer=initializers.RandomNormal(stddev=0.2)))
    generator.add(LeakyReLU(alpha=0.2))
    
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(784,activation='tanh'))
    
    generator.compile(loss='binary_crossentropy',optimizer=optimiser)
    
    return generator


def get_Discriminator(optimiser):
    discriminator=Sequential()
    discriminator.add(Dense(512,input_dim=784,kernel_initializer=initializers.RandomNormal(stddev=0.2)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(1,activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy',optimizer=optimiser)
    
    return discriminator


def get_GAN(discriminator,generator,optimiser,random_dim):
    discriminator.trainable=False
    ganInput=Input(shape=(random_dim,))
    x=generator(ganInput)
    
    ganOutput=discriminator(x)
    
    gan=Model(inputs=ganInput,outputs=ganOutput)
    
    gan.compile(loss='binary_crossentropy',optimizer=optimiser)
    
    return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)


def train(epochs=1,batchSize=128):
    xtrain,ytrain,xtest,ytest=load_minst_data()
    batch_count=xtrain.shape[0]/batchSize
    
    optimiser=get_optimiser()
    generator=get_Generator(optimiser)
    discriminator=get_Discriminator(optimiser)
    gan=get_GAN(discriminator,generator,optimiser,random_dim)

    for i in range(1,epochs+1):
        print('-'*15,'Epoch %d'%i,'-'*15)
        for _ in tqdm(range(int(batch_count))):
            noise=np.random.normal(0,1,size=[batchSize,random_dim])
            image=xtrain[np.random.randint(0,xtrain.shape[0],batchSize)]
            
            generatedImage=generator.predict(noise)
            
            X=np.concatenate([image,generatedImage])
            
            y=np.zeros(2*batchSize)
            y[:batchSize]=0.9
            
            discriminator.trainable=True
            discriminator.train_on_batch(X,y)
            
            noise=np.random.normal(0,1,size=[batchSize,random_dim])
            y=np.ones(batchSize)
            
            discriminator.trainable=False
            gan.train_on_batch(noise,y)
            
        if(i==1or i%20==0):
            plot_generated_images(i,generator)
if __name__=='__main__':
    train(400,128) 
    gan.save('gan.h5')
    
    
























