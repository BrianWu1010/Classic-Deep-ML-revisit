# Classic Deep ML Revisit

A from-scratch implementation of core machine learning and deep learning components using only low-level numerical tools.

---

## Objective

This repository focuses on rebuilding fundamental ML/DL operations with full transparency. Every computation is explicitly implemented to expose how models work at the tensor and linear algebra level.

---

## Principles

- First-principles implementation
- Explicit tensor transformations
- Minimal abstraction (NumPy only)
- Consistent shape convention: (H, W, C)
- Vectorized computation where possible

---

## Current Progress

### CNN Pipeline (Partial)

- Dynamic padding
- im2col-based convolution (matrix multiplication)
- He initialization
- ReLU activation
- Max pooling (window-based)

---

## Why This Matters

Modern frameworks hide implementation details. This project reverses that abstraction to build a precise understanding of:

- convolution as matrix multiplication  
- tensor reshaping as computation  
- spatial transformations in CNNs  

---

## Limitations

- No backpropagation  
- No batching  
- No bias terms  
- No GPU support  

---

## Next Steps

- Add backward pass  
- Introduce batching  
- Refactor into layer abstractions  
- Validate against PyTorch  

---

## Stack

- Python  
- NumPy  
- OpenCV  