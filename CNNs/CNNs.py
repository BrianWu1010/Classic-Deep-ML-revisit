import numpy as np
import cv2

# input
img = cv2.imread('/Users/boyuanwu/Projects/Classic-Deep-ML-revisit/CNNs/Flower_Data/train/daisy/144076848_57e1d662e3_m.jpg')
print ("image shape:", img.shape) 
H_input, W_input, C_input = img.shape

# Hyperparameters definition
padding = 1
stride = 1
# Filter Hyperparameters; filter size (N_k X (H_filter = 3) X (W_filter = 3) X (C_filter = 3)); N_k stand for numbers of kernels
# note: normally, Kernel is a 3D matrix with Height Width and Channels; Filter is a collection of kernels, which is a 4D matrix.
N_k_layer1 = 32
N_k_layer2 = 64
N_k_layer3 = 128
H_filter = 3
W_filter = 3

# Output size = (input_size - filter_size + 2 * padding)/stride + 1; since padded_img is already padded, therefore padded_input_size = (input_size + 2 * padding)
def output_size (H_input, W_input, N_k, H_filter, W_filter, padding, stride):
    H_output = int((H_input - H_filter + 2 * padding)/stride + 1)
    W_output = int((W_input - W_filter + 2 * padding)/stride + 1)
    C_output = N_k     # Channel of output is the same as the number of kernels in a filter
    return H_output, W_output, C_output

# Filter Matrix Generator
def filter_generator(Num_Kernel, H_filter, W_filter, C_input):
    fan_in = int(H_filter * W_filter * C_input)
    filter = np.random.randn(Num_Kernel, H_filter, W_filter, C_input) * np.sqrt(2/fan_in)
    return np.array(filter)

# ReLu
def ReLu(x):
    a = np.array(x,copy=True)
    a[a<0] = 0
    return a

# Max_pooling function
def max_pooling (input_matrix, window_size):
    H_input, W_input = input_matrix.shape # How to only get the H and W, no matter what the shape is?
    H_output = int(H_input / window_size)
    W_ouutput = int(W_input / window_size)
    output = np.zeros 
    for n in range (H_input):
        for m in range (W_input):
            partial_input_img = input_matrix [n : n + window_size, m : m + window_size]
            return 


# Padding; using np.pad(array, pad_width_spec, mode); where pad_with_spec is ((top, bottom), (left,right), (channel_before, channel_after))
padded_img = np.pad (img, ((padding,padding), (padding,padding), (0,0)), mode = 'constant', constant_values = 0)
print ("padded image shape", padded_img.shape)

H_output, W_output, C_output = output_size (H_input, W_input, N_k_layer1, H_filter, W_filter, padding, stride)
output = np.zeros ((H_output, W_output, C_output))

# Patches generation, using im2col(image to column) method (based on Toepliz Matrix)
patches = []
for n in range (H_output):
    for m in range (W_output):
        partial_input_img = padded_img [n : n + H_filter, m : m + W_filter]
        patches.append(partial_input_img)
patches = np.array (patches)
print ("patches shape: ", patches.shape) # Shape: (Number of Patches, H__patch, W_patch, C_input)

# filter_layer1 generation
filter_layer1 = filter_generator(N_k_layer1, H_filter, W_filter, C_input)
print ("filter shape: ", filter_layer1.shape)

# Convolution using purly reshaping, here the Bias is not added
patches_im2col = patches.reshape(-1, 27)       # matrix shape (number of patches, pixal values)-> pixal_value = (RGB of first patch's first pixal)(RGB of first patch's second pixal)...
print ("patches_im2col shape: ", patches_im2col.shape)
filter_layer1_flat = filter_layer1.reshape(-1, 27)    # matrix shape (number of kernels, kernel values)-> kernal_value = (kernel pixal value for RGB of first patch's first pixal)(kernal pixal value for RGB of first patch's second pixal)...
filter_layer1_col = filter_layer1_flat.T
print ("filter_vertical shape: ", filter_layer1_col.shape)
output_flattened = patches_im2col @ filter_layer1_col
print ("output_flattened shape:", output_flattened.shape)

# Convolution using einsum(Einstein Summation); another way to do convolution; it is not as intuitive as the pure reshaping method.
# output_vertical = np.einsum('phwc,khwc -> pk', patches, filter_layer1)

output = output_flattened.reshape (H_output, W_output, C_output) # Reconstructing output matrix from flatten one

# ReLu activate output
output_activated = ReLu (output)

