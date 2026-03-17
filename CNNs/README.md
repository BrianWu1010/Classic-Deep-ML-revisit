# CNN From Scratch — Implementation Notes

This document explains the internal logic of the CNN pipeline implemented in `CNNs.py`. The goal is not performance, but structural clarity and mathematical correctness.

---

## Tensor Convention

All tensors follow:

```
(H, W, C)
```

- H: height  
- W: width  
- C: channels  

---

## Pipeline Overview

```
Input Image
    ↓
Padding
    ↓
Sliding Window Extraction
    ↓
im2col Transformation
    ↓
Matrix Multiplication (Convolution)
    ↓
Reshape to Feature Map
    ↓
ReLU Activation
    ↓
Max Pooling
```

---

## 1. Padding

### Purpose
Ensure that convolution windows fully cover the input, especially at boundaries.

### Method

Padding is computed dynamically:

```
H_out = ceil((H_in - H_filter) / stride) + 1
Padding = (H_out - 1) * stride + H_filter - H_in
```

Padding is split symmetrically:
- top / bottom
- left / right

---

## 2. Sliding Window (Patch Extraction)

Instead of nested loops, patches are extracted using:

```
np.lib.stride_tricks.sliding_window_view
```

Output shape:

```
(H_out, W_out, H_filter, W_filter, C)
```

This represents all receptive fields across the image.

Stride is applied by slicing:

```
windows[::stride, ::stride]
```

---

## 3. im2col Transformation

Each patch is flattened:

```
(H_filter, W_filter, C) → (H_filter * W_filter * C)
```

Final shape:

```
(N_patches, patch_size)
```

Where:
```
N_patches = H_out * W_out
```

---

## 4. Kernel Transformation

Filters:

```
(N_k, H_filter, W_filter, C)
```

Flattened into:

```
(patch_size, N_k)
```

---

## 5. Convolution via Matrix Multiplication

Core operation:

```
output_flat = patches_im2col @ kernel_matrix
```

Shape:

```
(N_patches, N_k)
```

This is equivalent to applying all kernels to all patches simultaneously.

---

## 6. Output Reconstruction

Reshape back to spatial structure:

```
(H_out, W_out, N_k)
```

Each channel corresponds to one kernel.

---

## 7. ReLU Activation

Element-wise:

```
max(0, x)
```

Applied directly to the feature map.

---

## 8. Max Pooling

### Operation

- Sliding window over feature map
- Take maximum over spatial dimensions
- Preserve channel dimension

```
(H_pool, W_pool, C) → (1, 1, C)
```

### Implementation Detail

```
np.max(window, axis=(0,1))
```

---

## Key Design Decisions

### 1. Vectorization over Loops
- im2col + matrix multiplication replaces nested convolution loops
- closer to how real libraries implement convolution

### 2. No Hard-Coded Dimensions
- all shapes derived from input and parameters
- improves generality

### 3. Explicit Shape Transitions
- every reshape is intentional and traceable
- avoids silent broadcasting errors

### 4. Unified Padding Logic
- same function supports both convolution and pooling

---

## What This Implementation Reveals

- Convolution is fundamentally a **dot product over flattened patches**
- CNN efficiency comes from **linear algebra optimization**, not magic
- Tensor reshaping changes **computation semantics**, not data
- Pooling is simply a structured reduction

---

## Known Gaps

- No bias term in convolution
- No gradient computation
- No batching
- No layer abstraction (everything is procedural)

---

## Next Engineering Targets

- Add bias:  
  ```
  output += bias
  ```

- Backpropagation:
  - gradient wrt filters
  - gradient wrt input

- Batch support:
  ```
  (N, H, W, C)
  ```

- Modular layers:
  - Conv2D
  - Activation
  - Pooling

---

## Positioning

This implementation sits between:

- mathematical derivation (paper-level)
- production frameworks (PyTorch)

It is intended as a **bridge layer for understanding**, not a deployment system.