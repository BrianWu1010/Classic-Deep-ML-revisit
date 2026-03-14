import numpy as np

import cv2

# input
img = cv2.imread('/Users/boyuanwu/Projects/Classic-Deep-ML-revisit/CNNs/Flower_Data/train/daisy/144076848_57e1d662e3_m.jpg')
print ("image shape:", img.shape) 

# hyperparameters definition
padding = 1
stride = 1

# add padding
# here using np.pad(array, pad_width_spec, mode)
# where pad_with_spec is ((top, bottom), (left,right), (channel_before, channel_after))
padded_img = np.pad (img, ((padding,padding), (padding,padding), (0,0)), mode = 'constant', constant_values = 0)
print ("padded image shape", padded_img.shape)

H_padded_input, W_padded_input, C_padded_input = padded_img.shape 

# filter size (N_k X (H_filter = 3) X (W_filter = 3) X (C_filter = 3)); N_k stand for numbers of kernels
# number of kernels: first layer = 32; second layer = 64; third layer = 128
# note: normally, Kernel is a 3D matrix with Height Width and Channels; Filter is a collection of kernels, which is a 4D matrix.
N_k_layer1 = 32
N_k_layer2 = 64
N_k_layer3 = 128
H_filter = 3
W_filter = 3

def filter_generator(Num_Kernel, C_input, H_filter, W_filter):
    fan_in = int(H_filter * W_filter * C_input)
    filter = np.random.randn(Num_Kernel, C_input, H_filter, W_filter) * np.sqrt(2/fan_in)
    return np.array(filter)

filter_layer1 = filter_generator(N_k_layer1, C_padded_input, H_filter, W_filter)

print ("filter shape: ", filter_layer1.shape)
print ("filter: ", filter_layer1)

# output size = (input_size - filter_size + 2 * padding)/stride + 1; since padded_img is already padded, therefore padded_input_size = (input_size + 2 * padding)
H_output = int((H_padded_input - H_filter)/stride + 1)
W_output = int((W_padded_input - W_filter)/stride + 1)
print ("H_output", H_output, "\n"
       "W_output", W_output )

output = np.zeros ((H_output, W_output))

'''
# Convolution calculation using traditional way, for RGB layer 1, good for understanding the mechanism
for n in range (H_output):
    for m in range (W_output):
        partial_input_img_layer1 = padded_img_layer1 [n : n + H_filter, m : m + W_filter]
        if partial_input_img.shape != filter.shape:
            print("partial_input_img shape:",partial_input_img.shape,"\n(n,m):",n,m)
        output[n,m] = np.sum(partial_input_img * filter)
'''
# prepare convolution calculation using im2col(image to column) method (based on Toepliz Matrix)
patches = []
for n in range (H_output):
    for m in range (W_output):
        partial_input_img = padded_img [n : n + H_filter, m : m + W_filter]
        patches.append(partial_input_img)
patches = np.array (patches)
print ("patches shape: ", patches.shape)

'''
# Convolution using purly reshaping, here the Bias is not added
patches_im2col = patches.reshape(-1, 27)
print ("patches_im2col shape: ", patches_im2col.shape)
filter_vertical = filter.reshape(27,1)
print ("filter_vertical shape: ", filter_vertical.shape)
output_vertical = patches_im2col @ filter_vertical
output = output_vertical.reshape (H_output, W_output)
print (output)
'''

# Convolution using einsum(Einstein Summation)
output_vertical = np.einsum('phwc,kchw -> pk', patches, filter)

'''
output = output_vertical.reshape (H_output, W_output)
print (output)
'''