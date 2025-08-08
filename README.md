# **RCMLRS: A Low-Level Machine Learning Framework in Rust**

| Component            | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| **RAM**              | Tensor-like in-memory storage for multi-dimensional numerical data.         |
| **Compute**          | Highly optimized arithmetic and matrix operations implemented in pure Rust. |
| **Machine Learning** | Support for neural network training, evaluation, and inference.             |
| **Rust**             | Memory-safe, high-performance systems language for deterministic execution. |
| **Syntax**           | Ergonomic API design with method-chaining and operator overloading.         |

---

## **Abstract**

RCMLRS is a lightweight, low-level machine learning framework written entirely in Rust. It provides the fundamental computational building blocks required to construct and train models such as **Recurrent Neural Networks (RNNs)**, without relying on heavyweight dependencies or external bindings. By implementing core tensor operations, activation functions, optimizers, and normalization methods from scratch, RCMLRS enables full transparency of the machine learning process, critical for both research and systems-level optimization.

The framework is intentionally minimal yet extensible, serving as a testbed for experimenting with neural architectures, optimization algorithms, and memory management in high-performance environments. Unlike Python based ML frameworks, RCMLRS compiles natively to multiple platforms and offers deterministic, low-latency execution.

---

## **1. Introduction**

Machine learning frameworks in Rust remain scarce compared to Python ecosystems such as TensorFlow and PyTorch. Existing Rust options often suffer from:

1. **Installation complexity** Many require fragile bindings to C/C++ libraries, which can fail on macOS or ARM architectures.
2. **Platform limitations** Some frameworks exclude macOS or Windows support altogether.
3. **Opaque internals** High-level abstractions hide critical computational details from developers.

RCMLRS addresses these limitations by:

* **Full native Rust implementation** (no external bindings).
* **Crossplatform support**, verified on macOS and Linux.
* **Fine grained control** over every tensor operation and learning step.

---

## **2. System Architecture**

### **2.1 Tensor Engine**

At the core is the **`RamTensor`** structure, which stores numerical data in a three dimensional vector (`Vec<Vec<Vec<f32>>>`). This design supports:

* **Multilayer neural networks** with arbitrary shapes.
* **Constant time indexing** for element access.
* **Method chaining** for concise operations (for example, `tensor1 + tensor2`).

Key operations include:

* **Matrix multiplication** (`matmul`) with optional zero padding (`pad_matmul_to_another`).
* **Elementwise arithmetic** (`add`, `sub`, `powi`, `powf`).
* **Reduction functions** (`mean`, `median`, `find_min`, `find_max`).
* **Shape transformations** (`flatten`, `transpose`, `pad`).

---

### **2.2 Activation Functions**

All major activation functions are implemented with their derivatives for backpropagation:

| Function   | Equation                                     | Derivative                      | Use Case                                         |
| ---------- | -------------------------------------------- | ------------------------------- | ------------------------------------------------ |
| ReLU       | $f(x) = \max(0, x)$                          | $f'(x) = 1$ if $x > 0$ else $0$ | Sparse activations, avoiding vanishing gradients |
| Leaky ReLU | $f(x) = \max(ax, x)$                         | Same with slope $a$             | Small gradients for negative inputs              |
| Sigmoid    | $f(x) = \frac{1}{1+e^{-x}}$                  | $f'(x) = f(x)(1 - f(x))$        | Binary classification                            |
| Tanh       | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$   | $1 - f(x)^2$                    | Normalized outputs in $[-1,1]$                   |
| Softmax    | $f(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$    | Jacobian matrix                 | Multi-class classification                       |
| Swish      | $x \cdot \sigma(x)$                          | Complex, smooth gradients       | State-of-the-art deep nets                       |
| GELU       | $0.5x[1+\tanh(\sqrt{2/\pi}(x+0.044715x^3))]$ | Differentiable smooth step      | Transformer architectures                        |

---

### **2.3 Optimization Engine**

RCMLRS currently supports:

* **Adam Optimizer** Implements both default and customizable hyperparameters ($\beta_1, \beta_2, \epsilon$).
* **Loss functions** Mean Squared Error (MSE), Mean Absolute Error (MAE), with analytic derivatives for gradient-based learning.

The training loop involves:

1. **Forward pass** Sequential layer multiplications + activations.
2. **Loss computation** — `mse_loss` or `mae_loss`.
3. **Backward pass** Activation derivatives + optimizer update.

---

### **2.4 Normalization Methods**

* **Min-Max Scaling:**

$$
x' = \frac{x - \min(X)}{\max(X) - \min(X)}
$$

* **Z-Score Normalization:**

$$
x' = \frac{x - \mu}{\sigma}
$$

These are implemented as in place tensor methods for preprocessing input datasets.

---

### **2.5 Model Persistence**

RCMLRS supports **binary and JSON serialization** of trained model weights and biases, enabling reproducibility across sessions:

```rust
save_tensors_binary("model.bin", vec![weights, biases]);
let model_state = load_state_binary("model.bin");
```

Binary format offers **faster load times**, while JSON enables **human-readable inspection**.

---

## **3. Practical Example: RNN Construction**

Using RCMLRS, an RNN layer can be built from scratch:

```rust
let z1 = weights.matmul(input.clone()).unwrap();
let a1 = z1.sigmoid();
```

Here:

* `weights.matmul(input)` performs the linear transformation.
* `.sigmoid()` applies the non-linear activation.

This low-level access ensures **full control** over architectural experimentation, essential for research.

---

## **4. Discussion**

RCMLRS is not intended as a replacement for high level ML frameworks but as an **experimental platform** for:

* Understanding deep learning internals.
* Testing unconventional architectures.
* Building **Rust-native AI systems** for embedded or real-time applications.

Its transparency, portability, and deterministic execution make it a strong candidate for research projects, educational purposes, and performance critical AI systems.

---

## **6. Future Work**

* Implementation of **Convolutional Neural Networks (CNNs)**.
* GPU acceleration via Vulkan or WGPU.
* Expanded dataset loaders for CSV and image formats.
* Automatic differentiation engine.
* Keras Impl.

---
