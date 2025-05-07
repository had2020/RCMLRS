//use std::{f32::EPSILON, intrinsics::expf32};

use std::path::absolute;

use crate::{neuralnet_impls::NeuralNetwork, storage_op::Matrix, *};

// derivatives

/*
//TODO
// backpropgations
impl RamTensor {
    pub fn backpropgations_relu_derivative(&self) -> RamTensor {}

    pub fn backpropgations_sigmoid_derivative(&self) -> RamTensor {}

    pub fn backpropgations_tanh_derivative(&self) -> RamTensor {}
}
*/

// optimizers

//TODO
// Hyperparameters:
// - Learning rate (0.001)
// - First Moment Exponential Decay Rate
// - Second Moment Exponential Decay Rate
// - Epsilon (f32::EPSILON)
// - Weight Decay
// - AMSGrad
pub fn custom_adam_optimizer(
    alpha_learning_rate: f32,
    beta_1: f32,
    beta_2: f32,
    delta_timestep: usize,
) {
}

impl RamTensor {
    /// Recommanded learning rate 0.001
    /// passed in tensor which should be the backprogation deribvtive of the activiation(weights + bias)
    /// Self is the gradient passed into adam
    /// Just apply to tensor and bias in layer
    pub fn adam_optimizer(
        &mut self,
        m: &mut RamTensor,
        v: &mut RamTensor,
        learning_rate: f32,
        timestep: usize,
    ) {
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let epsilon: f32 = 1e-8;

        for matrix in 0..self.layer_length {
            for row in 0..self.shape.x {
                for col in 0..self.shape.y {
                    // gradient
                    let i: f32 = self.data[matrix][row][col];

                    // update bias first moment estimate
                    m.data[matrix][row][col] = beta1 * m.data[matrix][row][col] + (1.0 - beta1) * i;

                    // update bias second moment estimate
                    v.data[matrix][row][col] =
                        beta2 * v.data[matrix][row][col] + (1.0 - beta2) * i.powi(2);

                    // compute bias fixed estimates
                    let m_hat = m.data[matrix][row][col] / (1.0 - beta1.powi(timestep as i32));
                    let v_hat = v.data[matrix][row][col] / (1.0 - beta2.powi(timestep as i32));

                    // update parameter
                    self.data[matrix][row][col] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        }
    }

    //Activation Derivatives

    pub fn relu_deriv(&self) -> RamTensor {
        let mut new_tensor = RamTensor::new_layer_zeros(self.shape, self.layer_length);

        for matrix in 0..self.layer_length {
            new_tensor.data.push(vec![]);
            for row in 0..self.shape.y {
                for col in 0..self.shape.x {
                    let x: f32 = self.data[matrix][row][col];

                    if x > 0.0 {
                        new_tensor.data[matrix][row].push(1.0);
                    } else {
                        new_tensor.data[matrix][row].push(0.0);
                    }
                }
            }
        }

        new_tensor
    }

    pub fn lrelu_deriv(&self) -> RamTensor {
        let mut new_tensor = RamTensor::new_layer_zeros(self.shape, self.layer_length);

        for matrix in 0..self.layer_length {
            new_tensor.data.push(vec![]);
            for row in 0..self.shape.y {
                for col in 0..self.shape.x {
                    let x: f32 = self.data[matrix][row][col];

                    if x > 0.0 {
                        new_tensor.data[matrix][row].push(1.0);
                    } else {
                        new_tensor.data[matrix][row].push(0.01);
                    }
                }
            }
        }

        new_tensor
    }

    pub fn sigmoid_deriv(&self) -> RamTensor {
        self.clone() * (1.0 - self.clone())
    }

    pub fn tanh_deriv(&self) -> RamTensor {
        1.0 - (self.clone().tanh().powi(2))
    }

    pub fn swish_deriv(&self) -> RamTensor {
        let e = std::f32::consts::E; // Euler's number

        let mut new_tensor = RamTensor::new_layer_zeros(self.shape, self.layer_length);

        for matrix in 0..self.layer_length {
            new_tensor.data.push(vec![]);
            for row in 0..self.shape.y {
                for col in 0..self.shape.x {
                    let x: f32 = self.data[matrix][row][col];

                    let product = 1.0 / (1.0 + (e.powf(-x.clone())));

                    let derivative = product + x * product * (1.0 - product);
                    new_tensor.data[matrix][row].push(derivative);
                }
            }
        }

        new_tensor
    }

    pub fn gelu_deriv(&self) -> RamTensor {
        let sqrt_pi_d2 = (2.0 / std::f32::consts::PI).sqrt();
        let x_cubed = self.powi(3);
        let inner = sqrt_pi_d2 * (self.clone() + 0.044715 * x_cubed);
        let tanh_inner = inner.tanh();

        let sech_squared = 1.0 - tanh_inner.powi(2); // derivative of tanh

        let term1 = 0.5 * tanh_inner;
        let term2 = (0.5 * self.clone())
            * sech_squared
            * sqrt_pi_d2
            * (1.0 + 3.0 * 0.044715 * self.clone().powi(2));

        return term1 + term2; // full derivative
    }
}

// Loss Derivatives

/// Mean Sqareed Error Derivatives
pub fn mse_deriv(actual: RamTensor, predicted: RamTensor) -> RamTensor {
    let mut gradient_data: Vec<Vec<Vec<f32>>> = vec![];

    //TODO standardization for performance
    for matrix in 0..actual.layer_length {
        gradient_data.push(vec![]);
        for row in 0..actual.shape.x {
            gradient_data[matrix].push(vec![]);
            for col in 0..actual.shape.y {
                let difference = predicted.data[matrix][row][col] - actual.data[matrix][row][col];
                let gradient = (2 / actual.shape.x) as f32 * difference;
                gradient_data[matrix][row].push(gradient);
            }
        }
    }

    RamTensor {
        shape: actual.shape,
        layer_length: actual.layer_length,
        data: gradient_data,
    }
}

/// Mean Absolute Error Derivative
pub fn mae_deriv(actual: RamTensor, predicted: RamTensor) -> RamTensor {
    let mut error = RamTensor::new_layer_zeros(actual.shape, actual.layer_length);
    for matrix in 0..actual.layer_length {
        for row in 0..actual.shape.x {
            for col in 0..actual.shape.y {
                error += (predicted.clone() - actual.clone()).abs();
            }
        }
    }
    error
}

// loss methods

/// your error should be you actual - predicted values
/// Mean Absolute Error
/// This is applied to calulcate error gradient in Backpropagation.
pub fn mae_loss(actual: RamTensor, predicted: RamTensor) -> f32 {
    let diff = actual - predicted;
    let abs_diff = diff.abs();
    abs_diff.mean()
}

/// Mean Squared Error
/// This is applied to calulcate error gradient in Backpropagation.
pub fn mse_loss(actual: RamTensor, predicted: RamTensor) -> f32 {
    let sqared_error: RamTensor = (actual - predicted).powi(2);
    sqared_error.mean()
}

impl NeuralNetwork {
    /// Backprogation TODO mannul impl without network
    pub fn backprogation(&mut self, target: RamTensor) {
        let layers: usize = self.layers.len();
        let input_data = self.layers[0].clone();
        for layer in 0..layers {}
    }
}

// Draft
/*
L: number of layers
a[0]: input data
y: true labels aka target
For each layer l:
    W[l]: weight matrix
    b[l]: bias vector
    z[l]: weighted input
    a[l]: activation
    f: activation function
    f': derivative of activation

*/

// each z = w·x + b and applies such sigmoid(z).

// Steps
/*
Forward Pass
Loss
Backward Pass
Gradient Descent
*/

// Forward Pass
// for each layer
// added wights and bis
// then apply activiation
// DO not set yet
/*
def forward_pass(X, weights, biases):
    z1 = X @ weights["W1"] + biases["b1"]
    a1 = relu(z1)

    z2 = a1 @ weights["W2"] + biases["b2"]
    a2 = softmax(z2)

    cache = {
        "X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2
    }
    return a2, cache
*/

// backprop is what I need
// Start at the loss and move backward
// use the chain rule from calculus to compute gradients layer by layer
// Each layer get fault/error
// Adjust the weights a bit (opposite the gradient) using an optimizer like SGD or Adam
