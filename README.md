# Classic Deep ML Revisit

A from-scratch implementation of core machine learning and deep learning components using only low-level numerical tools.

---

## Objective

This repository rebuilds fundamental ML and DL operations with minimal abstraction so that each computation remains visible and traceable.

---

## Principles

- First-principles implementation
- Explicit tensor transformations
- Minimal abstraction using NumPy
- Consistent shape convention: `(H, W, C)`
- Clear class structure for each layer

---

## Current Progress

### CNN Components

- Added a `ConvLayer` class structure
- Added parameter storage for batch size, kernel count, and filter dimensions
- Added He (Kaiming) initialization logic for convolution filters
- Added internal filter storage through `self.filter`
- Added docstrings and comments to clarify layer purpose and attributes
- Added placeholder class structures for `ReLuLayer` and `MaxPoolingLayer`

---

## Current Limitations

- Convolution forward pass is not implemented yet
- The convolution filter generator still needs correction before use
- `ReLuLayer` is only a placeholder
- `MaxPoolingLayer` is only a placeholder
- No backward pass
- No bias terms
- No training loop
- No validation against framework outputs

---

## Immediate Next Step

- Fix the convolution filter generator so filters are correctly initialized inside `ConvLayer` and ready for the forward pass

---

## Stack

- Python
- NumPy