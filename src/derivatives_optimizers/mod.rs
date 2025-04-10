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
    pub fn adam_optimizer(&mut self, learning_rate: f32, timestep: usize) {
        // hyperparameters
        let alpha: f32 = learning_rate;
        let beta_1: f32 = 0.9;
        let beta_2: f32 = 0.999;
        let epsilon: f32 = EPSILON;

        // initialize
        let mut theta = 0.0;
        let mut m = 0.0;
        let mut v = 0.0;

        //adam step
        //for i
        // set m and v next
        // m is the for each i in matrix the difference or loss from last point
        // for each parameter in matrix:
        // m: a tensor the same shape as the parameter, holding the moving average of gradients (first moment).
        // v: a tensor the same shape, holding moving average of squared gradients (second moment, like variance).
    }
}

/*
Initialize m_0 = 0, v_0 = 0, t = 0
while not converged:
    t = t + 1
    g_t = ∇_θ L(θ_t)
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    v_t = β2 * v_{t-1} + (1 - β2) * (g_t)^2
    m̂_t = m_t / (1 - β1^t)
    v̂_t = v_t / (1 - β2^t)
    θ_t+1 = θ_t - α * m̂_t / (sqrt(v̂_t) + ε)
*/
