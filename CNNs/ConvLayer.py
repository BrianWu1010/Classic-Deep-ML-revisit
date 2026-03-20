import numpy as np

class ConvLayer:
    def __init__(self, batch, N_kernel, H_filter, W_filter):
        self.batch = batch
        self.N_kernel = N_kernel
        self.H_filter = H_filter
        self.W_filter = W_filter
        self.filter = None

    def filter_generator(N_Kernel, H_filter, W_filter, C_input):
        fan_in = int(H_filter * W_filter * C_input)
        # He(Kaiming) matrix initialization for ReLu activation
        filter = np.random.randn(N_Kernel, H_filter, W_filter, C_input).astype(np.float32) * np.sqrt(2/fan_in)
        self.filter = np.array(filter)