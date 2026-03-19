import numpy as np
import math
import random
from tensorflow.keras.datasets import mnist


# Batches creator
def batch_generator (x, y, batch_size):
    '''
    x: input img matrix, shape: (n_total_input, H_input, W_input, C_input)
    y: according labels, shape: (n_tital_lables,)
    batch_size: int, numbers of img for one batch
    '''
    N_input, H_input, W_input, C_input = x.shape[0:4]
    input_index_list = np.arange (N_input)
    random.shuffle(input_index_list) # shuffle index for img selection
    for index in range (0, N_input, batch_size):
        small_index_list = input_index_list[index: index + batch_size]
        x_batch = x[small_index_list]
        y_batch = y[small_index_list]
        yield x_batch, y_batch

# Filter Matrix Generator
def filter_generator(Num_Kernel, H_filter, W_filter, C_input):
    fan_in = int(H_filter * W_filter * C_input)
    filter = np.random.randn(Num_Kernel, H_filter, W_filter, C_input).astype(np.float32) * np.sqrt(2/fan_in)
    return np.array(filter)

# ReLu
def ReLu(x):
    a = np.array(x,copy=True)
    a[a<0] = 0
    return a

def padding(input_matrix, H_window, W_window, stride):

    H_input, W_input = input_matrix.shape[1:3]
    H_conved = math.ceil((H_input - H_window) / stride) + 1
    H_padding = (H_conved - 1) * stride + H_window - H_input
    W_conved = math.ceil((W_input - W_window) / stride) + 1
    W_padding = (W_conved - 1) * stride + W_window - W_input

    H_pad_top = H_padding //2
    H_pad_bottom = H_padding - H_pad_top
    W_pad_left = W_padding //2
    W_pad_right = W_padding - W_pad_left

    # Padding; using np.pad(array, pad_width_spec, mode); where pad_with_spec is ((imgs_front, imgs_tail)(top, bottom), (left,right), (channel_before, channel_after))
    padded = np.pad (input_matrix, ((0,0)(H_pad_top, H_pad_bottom), (W_pad_left, W_pad_right), (0,0)), mode = 'constant', constant_values = 0)

    return padded, H_conved, W_conved

# Max_pooling function
def max_pooling (input_matrix, H_pool, W_pool, stride):
    '''
    performing a max pooling on input_matrix

    Args:
    input_matrix: Array of shape (H, W, C)
    pool_size: int
    stride : int
    '''
    input_padded, H_output, W_output = padding(input_matrix, H_pool, W_pool, stride)[:3]
    C_output = input_padded.shape [-1]
    max_pooled = np.zeros((H_output, W_output, C_output), dtype = np.float32)
    for n in range (H_output):
        index_n = n * stride # locate the top index of the window in the input img
        for m in range (W_output):
            index_m = m * stride # locate the left index of the window in the input img
            partial_input = input_padded [index_n : index_n + H_pool , index_m : index_m + W_pool, :]
            # compare every element from H and W in the layer of C. To achieve that, first collapse the H and W, then compare across the different channels' value
            # since the axis 0 is H, and axis 1 is W; only need to froze the axis 2, to avoid comparism among the channels.
            max_pooled[n, m, :] = np.max(partial_input, axis = (0,1))
    return max_pooled


# Hyperparameters definition
padding = 1
conv_stride = 1
# Filter Hyperparameters; filter size (N_k , (H_filter = 3) , (W_filter = 3) , (C_filter = 3)); N_k stand for numbers of kernels
# note: normally, Kernel is a 3D matrix with Height Width and Channels; Filter is a collection of kernels, which is a 4D matrix.
N_k_layer1 = 32
N_k_layer2 = 64
N_k_layer3 = 128
H_filter = 3
W_filter = 3











# 0. load dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# input img 28×28 grayscale, values in [0, 255]
# x: imgs, shape (number_img, 28, 28); y: labels, shape (number_img, )

x_train = x_train.reshape(-1, 28, 28, 1) # reshaping: (N_p, H_input, W_input, C_input)
x_test = x_test.reshape(-1, 28, 28, 1) # reshaping: (N_p, H_input, W_input, C_input)

print ("x_train: ", x_train.shape, "y_train: ", y_train.shape)
print ("x_test: ", x_test.shape, "y_test: ", y_test.shape)


# 1. Padding raw img
# padded_img: padded input raw img; H_output, W_output, C_output: the shape of the output of filter applied padded_img
padded_img, H_conved, W_conved = padding (x_train, H_filter, W_filter, 1)

print ("padded image shape", padded_img.shape)

# 2. Patches generation, using im2col(image to column) method (based on Toepliz Matrix)
windows = np.lib.stride_tricks.sliding_window_view(
    padded_img,
    window_shape = (H_filter, W_filter),
    axis = (1, 2)
)
# shape: (N_p, H_padded - H_filter + 1, W_padded - W_filter + 1, C_input, H_filter, W_filter); consider it as stride = 1, then extract all the possible patches
# N_p: number of img in a patch

windows = windows.transpose(0, 1, 2, 4, 5, 3)
# shape: (N_p, H_patches, W_patches, H_filter, W_filter, C_input)
N_p, H_patches, W_patches, H_filter, W_filter, C_input = windows.shape()

patches = windows[:, ::conv_stride, ::conv_stride, :, :, :]

patches_im2col = patches.reshape(N_p, H_filter * W_filter * C_input) # matrix shape (number of patches, pixal values)-> pixal_value = (RGB of first patch's first pixal)(RGB of first patch's second pixal)...

print("patches shape:", patches.shape)
print("patches_col shape:", patches_im2col.shape)


# 3. filter_layer1 generation
filter_layer1 = filter_generator(N_k_layer1, H_filter, W_filter, C_input)
filter_layer1_flat = filter_layer1.reshape(-1, H_filter * W_filter * C_input) # matrix shape (number of kernels, kernel values)-> kernal_value = (kernel pixal value for RGB of first patch's first pixal)(kernal pixal value for RGB of first patch's second pixal)...
filter_layer1_col = filter_layer1_flat.T
print ("filter shape: ", filter_layer1.shape)
print ("filter_vertical shape: ", filter_layer1_col.shape)

# 4. Convolution using purly reshaping, here the Bias is not added
output_flattened = patches_im2col @ filter_layer1_col
print ("output_flattened shape:", output_flattened.shape)

output = output_flattened.reshape (H_conved, W_conved, N_k_layer1) # Reconstructing output matrix from flatten one

# 5. ReLu activate output
output_activated = ReLu (output)

# 6. Max pooling
output_maxpooled = max_pooling(output_activated, 2, 2, 2)
print(output_maxpooled)

