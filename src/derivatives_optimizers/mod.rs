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

/// Recommanded learning rate 0.001
pub fn adam_optimizer(learning_rate: f32, timestep: usize) {
    let alpha: f32 = learning_rate;
    let beta_1: f32 = 0.9;
    let beta_2: f32 = 0.999;
    let epsilon: f32 = EPSILON;
}
