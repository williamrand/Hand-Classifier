# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras.utils  import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation

class SignLanguage:
    def __init__(self):
        self.model = None
        
        self.data = {
            "train": None,
            "test" : None
        }
        self.create_model()
    
    def create_model(self):
        """
        Create a CNN model and save it to self.model
        """
        
        # TODO: Create a Sequential model conv,pooling,dropout, softmax, dense /flatten?(start or end)
        model = Sequential() 
        model.add(Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
        model.add(Conv2D(32,(3,3),activation='relu'))  
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(.4))
        model.add(Conv2D(25,(3,3),activation='relu')) 
        model.add(Conv2D(25,(3,3),activation='relu'))  
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(.4))
        model.add(Flatten()) 
        model.add(Dense(25,activation='relu'))
        #model.add(Dropout(.4))
             
        model.add(Activation('softmax'))
        
        
        # TODO: Compile the model with categorical_crossentropy
        model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        x=model.getLayer(0)
        print(x.output)
        x=model.getLayer(2)
        print(x.output)
    def prepare_data(self, images, labels):
        """
        Use this method to normalize the dataset and split it into train/test.
        Save your data in self.data["train"] and self.data["test"] as a tuple
        of (images, labels)
        
        :param images numpy array of size (num_examples, 28*28)
        :param labels numpy array of size (num_examples, )
        """
        # TODO : split into training and validation set
        # TODO : reshape each example into a 2D image (28, 28, 1) what is the convolution method? use that to make 28,28,1? make into samples and classes
        #encode one hot array of 25 with 1 1 value?
        # Split data 9:1 training and test
        num_examples = len(images)
        training_size = int(num_examples * 0.9)
        binary_label=np.zeros((num_examples,25))
        picAr=np.zeros((num_examples,28,28,1))
        for y in range(0,num_examples):
          hold=labels[y]
          binary_label[y][hold]=1
          picAr[y]=np.reshape(images[y],(28,28,1))
        x_train = picAr[0:training_size]
        x_test = picAr[training_size:]
        y_train = binary_label[0:training_size]
        y_test = binary_label[training_size:]

        self.data = {
            "train": (x_train, y_train), # (x_train, y_train)
            "test" : (x_test, y_test), # (x_test, y_test)
        }
    
    def train(self, batch_size:int=128, epochs:int=50, verbose:int=1):
        """
        Use model.fit() to train your model. Make sure to return the history for a neat visualization.
        
        :param batch_size The batch size to use for training
        :param epochs     Number of epochs to use for training
        :param verbose    Whether or not to print training output
        """
        history = self.model.fit(self.data["train"][0], self.data["train"][1], epochs=epochs, batch_size=batch_size, verbose=verbose)
        # history = None
        return history
    
    def predict(self, data):
        """
        Use the trained model to predict labels for test data.
        
        :param data: numpy array of test images
        :return a numpy array of test labels. array size = (num_examples, )
        """
        
        # Don't forget to normalize the data in the same way as training data
        # self.model.predict() and np.argmax( , axis=1) might help
        num_examples = len(data)
        picAr=np.zeros((num_examples,28,28,1))
        for y in range(0,num_examples):
          picAr[y]=np.reshape(data[y],(28,28,1))
        arr=self.model.predict(picAr)
        fin=np.zeros((len(arr),1))
        for t in range(0,len(arr)):
          fin[t]=np.argmax(arr[t])
        
        return fin
    
    def visualize_data(self, data):
        """
        Visualizing the hand gestures
        
        :param data: numpy array of images
        """
        if data is None: return
        
        nrows, ncols = 5, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(nrows):
            for j in range(ncols):
                axs[i][j].imshow(data[0][i*ncols+j].reshape(28, 28), cmap='gray')
        plt.show()

    def visualize_accuracy(self, history):
        """
        Plots out the accuracy measures given a keras history object
        
        :param history: return value from model.fit()
        """
        if history is None: return
        
        plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        plt.title("Accuracy")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train','test'])
        plt.show()