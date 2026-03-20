from tensorflow.keras.datasets import mnist
import numpy as np
import random

class DataManager:
    def __init__(self):
        self.x_train = None 
        self.y_train = None 
        self.x_test = None 
        self.y_test = None

    def load_and_prepare(self):
        # 1. data loading
        (raw_x_train, y_train),(raw_x_test, y_test) = mnist.load_data()   # x: imgs, shape (number_img, 28, 28); y: labels, shape (number_img, ); input img 28×28 grayscale, values in [0, 255]
        # 2. reshaping data matrix
        self.x_train = self._x_reshaping(raw_x_train)
        self.y_train = self._y_reshaping(y_train)
        self.x_test = self._x_reshaping(raw_x_test)
        self.y_test = self._y_reshaping(y_test)
        print ("Data loaded successfully.")

    def generate_batch_train(self, batch_size = 64):
        return self._batch_generator(self.x_train, self.y_train, batch_size, shuffle = True) # for training, shuffle is required before feeding the model
    
    def generate_batch_test(self, batch_size = 64):
        return self._batch_generator(self.x_test, self.y_test, batch_size, shuffle = False) # for testing, no need to shuffle before feeding the model
    
    def _batch_generator(self, x, y, batch_size = 64, shuffle = True):
        '''
        x: input img matrix, shape: (n_total_input, H_input, W_input, C_input)
        y: corresponding ground_truth_labels, shape: (n_tital_lables,)
        batch_size: int, numbers of img for one batch
        '''
        N_input, H_input, W_input, C_input = x.shape[0:4]
        input_index_list = np.arange (N_input)
        if shuffle:
            random.shuffle(input_index_list) # shuffle index for img selection
        for index in range (0, N_input, batch_size):
            small_index_list = input_index_list[index: index + batch_size]
            x_batch = x[small_index_list]
            y_batch = y[small_index_list]
            yield x_batch, y_batch

    def _x_reshaping(self, x):
        # reshaping: (N_img, H_img, W_img, C_img); since inputs are grey scale, channel_input is 1
        # normalization: divided by 255, because standard 8-bit grayscale pixels go from 0 to 255.
        return x.reshape(-1, 28, 28, 1) / 255   

    def _y_reshaping(self, y):
        return y.reshape(-1, 1)   # reshaping: (N_img, 1)
    
