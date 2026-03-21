import numpy as np

class ConvLayer:
    """
    A specialist worker that handles 2D Convolutional math.
    
    Attributes:
        filters (numpy.ndarray): The learned kernels stored in the 'fridge'.
        stride (int): How many pixels the filter jumps.
        input_cache (numpy.ndarray): 'Frozen' input for backpropagation.
    """
    def __init__(self, batch, N_kernel, H_filter, W_filter, N_k_layer):
        """
        Initializes the layer and generates filters.
        
        Args:
            n_kernels: The number of unique filters to create.
            f_size: The Height/Width of the filters (assumes square).
            stride: The movement step (defaults to 1).
        """
        self.batch = batch
        self.N_kernel = N_kernel
        self.H_filter = H_filter
        self.W_filter = W_filter
        self.N_k_layer1 = N_k_layer
        self.filter = None
    
    def _filter_generator(N_Kernel, H_filter, W_filter, C_input):

        fan_in = int(H_filter * W_filter * C_input)
        # He(Kaiming) matrix initialization for ReLu activation
        self.filter = np.random.randn(N_Kernel, H_filter, W_filter, C_input).astype(np.float32) * np.sqrt(2/fan_in)

    pass

class ReLuLayer:
    """Applies the Rectified Linear Unit activation function."""
    pass

class MaxPoolingLayer:
    """
    Performs Max Pooling to reduce spatial dimensions.
    
    Attributes:
        pool_size (int): Dimensions of the pooling window.
        stride (int): Step size.
        cache (dict): Stores the 'mask' of where the max values were.
    """
    pass