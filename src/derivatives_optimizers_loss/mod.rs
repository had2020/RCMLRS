use std::f32::EPSILON;

use crate::{neuralnet_impls::NeuralNetwork, *};

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

    //TODO other deriv

    //pub fn relu_deriv(&self) -> RamTensor {}

    pub fn sigmoid_deriv(&self) -> RamTensor {
        self.clone() * (1.0 - self.clone())
    }

    //pub fn tanh_deriv(&self) -> RamTensor {}

    //pub fn softmax_deriv(&self) -> RamTensor {}

    //pub fn swish_deriv(&self) -> RamTensor {}

    //pub fn gelu_deriv(&self) -> RamTensor {}
}

/// your error should be you actual - predicted values
/// Mean Absolute Error
pub fn mae_loss(actual: RamTensor, predicted: RamTensor) -> f32 {
    let diff = actual - predicted;
    let abs_diff = diff.abs();
    abs_diff.mean()
}

/// Mean Squared Error
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
