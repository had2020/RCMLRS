# RCMLRS
| Ram | Compute | Machine | Learning | Rust | Syntax |
|-----|---------|---------|----------|------|--------|
|     |         |         |          |      |        |

## Project State
This is a very low Machine learning framework, as Keras impl is not complete and is only in a work in process state. None the less there are quite a few features already, enough to make Recurrent Neural  Networks. 

## Purpose
In Rust very few Machine Learning frameworks exist and many have predisposed software limits, or many issues. For example the Tensorflow integration with rust was a huge trouble to install and get working in rust and did not work for Macos, same story with alot of other frameworks. This is due to Rust being such a new language, which has a limited developer community. One other framework I used refused to work after I spend days reading the docs and trying to get a simple MIST example to work, just when I was finally able to compile the whole thing refused to work loggging that this was not available for Macos. 

# Features

## Activation Functions 
So far this project as all the core activation functions.

- ReLU `max(0,x)`
- Leaky ReLU `max(ax,x)`
- Sigmoid `1/1+e^-x`
- Tanh `(e^x - e^-x) / (e^x + e^-x)`
- Softmax `exp(z_i) / Σ_j exp(z_j)`
- Swish `x * (1.0 / (1.0 + e^-x))`
- GELU `0.5x(1+Tanh(2/PI(x + 0.044715x^3)))`

## Activation Derivatives 
RCMLRS also has in built functions that let you easy apply derivatives of your activation functions directly to tensors for the core activation functions, listed before.

Example Usage: 
```rust
let z1 = weights.matmul(input.clone()).unwrap();
let a1 = z1.sigmoid();
```

## Loss Optimizer
RCMLRS has a function for the Adam Optimizer ``adam_optimizer`` with some recommended hyperparameters and a ``custom_adam_optimizer`` for custom hyperparameters, if required. 

## MAE and MSE
RCMLRS has ``mse_deriv``, and ``mae_deriv`` derivatives along with ``mae_loss`` and ``mse_loss`` counterparts.

## Normalization
For methods of normalization this framework has ``min_max_norm`` and ``z_score_norm`` 

## Tensor Operations
Atherimetical operations don't need to be called by a function you can use them just by applying operations across tensors for example ``ramtensor0 + ramtensor1 = ramtensor2``

- find_min ``find_min(&self) -> f32``
- find_max ``find_max(&self) -> f32``
- scaler ``scaler(self, num: f32) -> Self``
- insert_matrix ``insert_matrix(&self, layer_index: usize, new_layer: Vec<Vec<f32>>) -> Self``
- new_random ``new_random(shape: Shape, layer_length: usize, rand_min: f32, rand_max: f32) -> Self``
- new_layer_zeros ``new_layer_zeros(shape: Shape, layer_length: usize) -> Self``
- matmul ``matmul(&self, another_tensor: RamTensor) -> Result<RamTensor, String>``
- pad_matmul_to_another ``pad_matmul_to_another(&self, another_tensor: RamTensor, zero_value: f32) -> RamTensor``
- add ``st_add(&self, another_tensor: RamTensor) -> Result<RamTensor, String>``
- sub ``sub(&self, another_tensor: RamTensor) -> Result<RamTensor, String>``
- flatten ``flatten(&self) -> RamTensor``
- to_scalar ``to_scalar(&self) -> Result<f32, String>``
- mean ``mean(&self) -> f32``
- median ``median(&self) -> f32``
- abs ``abs(&self) -> RamTensor``
- data_points ``data_points(&self) -> usize``
- from ``from<T: Into<f32>>(value: T) -> Self``
- pad ``pad(&self, to_shape: Shape, to_layer_length_shape: usize, pad_value: f32) -> RamTensor``
- retrive_matrice ``retrive_matrice(&self, matrix: usize, row: usize, col: usize) -> f32``
-  scaler_to_f32 ``scaler_to_f32(&self) -> f32``
- powi ``powi(&self, n: i32) -> RamTensor``
- powf ``powf(&self, n: f32) -> RamTensor``
- std ``(&self, sample: bool) -> f32`` aka population standard deviation
- normalize ``normalize(&self) -> RamTensor``
- transpose ``transpose(&self) -> RamTensor``
- from_scalar ``from_scalar(&self, value: f32, shape: Shape)``
- is_scalar ``is_scalar(&self) -> bool``
- outer_product ``outer_product(&self, another_tensor: RamTensor) -> RamTensor``
- parameters ``parameters(&self) -> f32`` aka Gets the number of parameters
- f32_to_scaler ``f32_to_scaler(scaler: f32) -> RamTensor``

## Saving and Loading Networks/Models to file
- save_tensors_json(filename: &str, tensors: Vec<RamTensor>)
- save_tensors_binary(filename: &str, tensors: Vec<RamTensor>

- load_state_json(filename: &str) -> Vec<RamTensor>
- load_state_binary(filename: &str) -> Vec<RamTensor>

Usage Examples:

Saving:
```rust
save_tensors_binary(
    filename,
    vec![
        weights,
        hidden_layer,
        bias1,
        RamTensor {
            shape: Shape { x: 1, y: 1 },
            layer_length: 1,
            data: vec![vec![vec![bias2]]],
        },
    ],
);
```
Loading:
```rust
 let model = load_state_binary(filename);

 let mut weights = model[0].clone();
 let mut hidden_layer = model[1].clone();
 let mut bias1 = model[2].clone();
 let mut bias2 = model[3].data[0][0][0].clone();
```
