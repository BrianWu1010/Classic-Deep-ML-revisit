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

# filter size (3 X 3)
filter = np.array([[1, 4, 1],
                   [4, 16, 4],
                   [1, 4, 1],
                   ]) / 36
print ("filter shape", filter.shape)
H_filter, W_filter = filter.shape

# slicing the padded_img, using nested loop
padded_img_layer1 = np.zeros((H_padded_input,W_padded_input))
for n in range (H_padded_input):
    for m in range (W_padded_input):
        padded_img_layer1[n,m] = padded_img [n, m, 0]
print("padded_img_layer1 shape", padded_img_layer1.shape)

# slicing the padded_img, using NumPy way
padded_img_layer2 = padded_img [:,:,1].copy
padded_img_layer3 = padded_img [:,:,2].copy

# output size = (input_size - filter_size + 2 * padding)/stride + 1; since padded_img is already padded, therefore padded_input_size = (input_size + 2 * padding)
H_output = int((H_padded_input - H_filter)/stride + 1)
W_output = int((W_padded_input - W_filter)/stride + 1)
print ("H_output", H_output, "\n"
       "W_output", W_output )

output = np.zeros ((H_output, W_output))

for n in range (H_output):
    for m in range (W_output):
        partial_input_img = padded_img_layer1 [n : n + H_filter, m : m + W_filter]
        if partial_input_img.shape != filter.shape:
            print("partial_input_img shape:",partial_input_img.shape,"\n(n,m):",n,m)
        output[n,m] = np.sum(partial_input_img * filter)

print ("output:", output)

continue