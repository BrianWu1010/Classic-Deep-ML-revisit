# Classic Deep ML Revisit

A from-scratch reconstruction of classical deep learning components with explicit control over tensor transformations and numerical operations.

---

## Objective

This repository focuses on rebuilding core deep learning operations using only low-level numerical tools (NumPy), with the goal of making every transformation **traceable, verifiable, and mathematically grounded**.

The emphasis is not performance, but **mechanistic transparency**.

---

## Design Philosophy

- **First-principles implementation**  
  Every operation (convolution, pooling, activation) is derived directly from its mathematical definition.

- **Explicit tensor transformations**  
  All reshaping, flattening, and dimension transitions are written out and inspectable.

- **Minimal abstraction**  
  Avoid frameworks (PyTorch, TensorFlow) to expose underlying computation.

- **Shape discipline**  
  A consistent tensor convention is enforced across the pipeline:
  ```
  (H, W, C)
  ```

- **Vectorization over loops**  
  Prefer linear algebra formulations (im2col + GEMM) over nested iteration.

---

## Implemented Components

### 1. Convolution Pipeline (Fully Vectorized)

- Dynamic **same-style padding**
- Patch extraction via:
  - `np.lib.stride_tricks.sliding_window_view`
- im2col transformation:
  ```
  (H, W, Hf, Wf, C) → (N_patches, Hf * Wf * C)
  ```
- Kernel flattening:
  ```
  (N_k, Hf, Wf, C) → (Hf * Wf * C, N_k)
  ```
- Convolution via matrix multiplication:
  ```
  output = patches_im2col @ kernel_matrix
  ```
- Output reconstruction:
  ```
  → (H_out, W_out, N_k)
  ```

---

### 2. Initialization

- He initialization:
  ```
  W ~ N(0, 2 / fan_in)
  ```
- Stabilizes variance across layers.

---

### 3. Activation

- ReLU (element-wise thresholding at zero)

---

### 4. Max Pooling

- Sliding window with dynamic padding
- Channel-wise reduction:
  ```
  max over (H, W), preserve C
  ```
- Explicit stride handling
- Output shape aligned with convolution pipeline

---

## Current Architecture (Single Block)

```
Input Image (H, W, C)
        ↓
Padding
        ↓
Sliding Window (patch extraction)
        ↓
im2col (flatten patches)
        ↓
Matrix Multiplication (Convolution)
        ↓
Reshape → (H, W, N_k)
        ↓
ReLU
        ↓
Max Pooling
```

---

## Key Engineering Improvements

- Eliminated hard-coded dimensions (e.g., `27`)
- Replaced nested loops with vectorized window extraction
- Unified padding logic across convolution and pooling
- Standardized tensor layout `(H, W, C)`
- Enforced `float32` consistency
- Clear separation between:
  - data extraction
  - linear algebra computation
  - shape reconstruction

---

## What This Repository Demonstrates

- Convolution as matrix multiplication (im2col + GEMM)
- Tensor reshaping as a computational tool
- Effects of padding and stride on spatial resolution
- Pooling as structured reduction
- Dependence of implementation on memory layout and shape conventions

---

## Limitations

- No backpropagation
- No batching (single image only)
- No bias terms in convolution
- No modular layer abstraction
- CPU-only (NumPy)

---

## Next Steps

- Add bias to convolution
- Implement backward pass (gradients)
- Support batch input: `(N, H, W, C)`
- Refactor into reusable layers:
  - `Conv2D`
  - `ReLU`
  - `MaxPool`
- Validate against PyTorch / TensorFlow outputs
- Extend to multi-layer CNN + classifier

---

## Tools

- Python
- NumPy
- OpenCV

---

## Long-Term Goal

To build a complete deep learning stack from scratch where:

- every tensor transformation is explicit,
- every computation is verifiable,
- and no abstraction is treated as a black box.