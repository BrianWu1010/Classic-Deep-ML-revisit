# Classic Deep ML Revisit

This repository re-implements classic deep learning algorithms and papers **from scratch** with minimal reliance on high-level libraries.

## Purpose
The goal of this project is to expose the **mathematical mechanisms** behind classical machine learning and deep learning methods. Instead of relying on high-level frameworks, the implementations prioritize clarity, explicit tensor tracing, and direct correspondence with the underlying equations.

Each component is written to make the **math-to-code relationship explicit**, helping readers see how deep learning operations are constructed from basic numerical building blocks.

## Principles
- **Math-first implementation** — code structure follows the mathematical formulation.
- **Minimal abstractions** — avoid high-level frameworks when possible.
- **Educational clarity** — prioritize readability, tensor-shape transparency, and conceptual traceability.
- **From-scratch learning** — manually build core components to understand the mechanics.
- **Shape awareness** — carefully track how data dimensions change across each operation.

## Current Progress
The repository is still a **work in progress**, but the CNN pipeline has advanced beyond a simple convolution demo. Current implemented components include:

- Manual **input padding**
- **Output dimension calculation** for convolution / window-based operations
- **He initialization** for convolution kernels
- **Patch extraction** from padded images using an **im2col-style** method
- Convolution implemented via **matrix multiplication**
- **ReLU activation**
- A working **max pooling** function with explicit sliding-window logic
- Early handling of **dynamic padding for pooling windows**
- Shape tracing across image -> patches -> flattened patches -> convolution output -> activation -> pooled output

These components are designed to show how standard CNN operations can be reconstructed directly from array manipulation and linear algebra.

## Current Technical Focus
Right now, the project is focused on making the implementation more mathematically consistent and structurally clean. The main engineering directions are:

- Remove hard-coded values such as the flattened filter size (`27`) and replace them with dimension-driven calculations
- Enforce a single tensor layout convention throughout the script, such as either:
  - `(H, W, C)`, or
  - `(C, H, W)`
- Consolidate related convolution math so that **padding, output size, and window traversal** are derived in a unified way
- Correct tensor reshaping so convolution output and pooling input follow the same dimension standard
- Continue improving the pooling implementation so the data flow is both correct and easy to inspect

## Why This Project Matters
Many deep learning libraries hide the operational details behind optimized APIs. This project aims to reverse that abstraction by rebuilding the underlying logic step by step.

The emphasis is not just on getting the code to run, but on understanding:

- how convolution becomes matrix multiplication
- how tensor reshaping changes interpretation without changing data
- how activation and pooling transform feature maps
- how implementation choices depend on dimension conventions

## Tools
- Python
- NumPy
- OpenCV

## Goal
To develop a **deep, first-principles understanding of machine learning and deep learning systems** by rebuilding them step by step from mathematical foundations, while making every tensor transformation explicit and inspectable.


## Next Steps
- Replace hard-coded flattened dimensions with computed variables
- Standardize all tensor shapes across convolution and pooling
- Add convolution bias explicitly
- Refactor convolution and pooling utilities into reusable layer-style functions
- Compare manual implementations against library outputs for verification