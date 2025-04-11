use std::f32::EPSILON;

use crate::*;

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
                    let i: f32 = self.data[matrix][row][col];

                    m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * self.data[i];

                    v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * self.data[i].powi(2);

                    let m_hat = m.data[i] / (1.0 - beta1.powi(timestep as i32));
                    let v_hat = v.data[i] / (1.0 - beta2.powi(timestep as i32));

                    self.data[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                }
            }
        }
    }
}
