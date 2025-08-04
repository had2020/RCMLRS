# RCMLRS
- Ram - Compute - Machine - Learning - Rust - Syntax

This was my first custom ML framework in Rust using my own tensor structure and basic gradient descent — at the time I hadn’t implemented full safety or indexing guards yet, but it marked the beginning of my exploration into safe, low-level ML in Rust.

The code is divided into modules to make it easier to optimize and maintain, for readability.

WIP, "Work in Progress"

Machine learning framework in Rust, ramless, cluster compute
Very early trials of creations...

 TODO docs
 Notes
 - Must delete txt file not text inside
 - creation does not use buf so TODO use buf

This Machine Learning libary has the following activation functions, as implimations on RamTensor struct:

Impl RamTensor

- ReLU `max(0,x)`
- Leaky ReLU `max(ax,x)`
- Sigmoid `1/1+e^-x`
- Tanh `(e^x - e^-x) / (e^x + e^-x)`
- Softmax `exp(z_i) / Σ_j exp(z_j)`
- Swish `x * (1.0 / (1.0 + e^-x))`
- GELU `0.5x(1+Tanh(2/PI(x + 0.044715x^3)))`

Main idea multi-thread completey memory safe AI
