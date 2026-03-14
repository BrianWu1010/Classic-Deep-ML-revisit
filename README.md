# Classic Deep ML Revisit

This repository re-implements classic deep learning algorithms and papers **from scratch** with minimal reliance on high-level libraries.

## Purpose
The goal of this project is to expose the **mathematical mechanisms** behind classical machine learning and deep learning methods. Instead of relying on high-level frameworks, the implementations prioritize clarity and direct correspondence with the underlying equations.

Each component is written to make the **math-to-code relationship explicit**, helping readers understand how the algorithms actually operate internally.

## Principles
- **Math-first implementation** – code structure follows the mathematical formulation.
- **Minimal abstractions** – avoid high-level frameworks when possible.
- **Educational clarity** – prioritize readability and conceptual transparency.
- **From-scratch learning** – build core components manually to understand the mechanics.

## Current Scope
The repository is **work in progress**. Currently implemented components include:

- Manual **image padding**
- **Output dimension calculation** for convolution layers
- **He initialization** for convolution kernels
- **Patch extraction (im2col style)** from images
- Convolution implemented via **matrix multiplication**
- Alternative convolution implementation using **Einstein Summation (`einsum`)**
- **ReLU activation**
- Initial work toward **max pooling implementation**

These components are written to explicitly show how convolution operations in CNNs can be derived from linear algebra.

## Tools
- Python
- NumPy (for numerical computation)
- OpenCV (for image loading)

## Goal
To develop a **deep, first-principles understanding of machine learning algorithms** by rebuilding them step by step directly from the mathematical foundations.