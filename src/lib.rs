// Shared/Core RC-ML-RS script
// By: Hadrian Lazic
// Notes: main functionality is split into modules for organization
// Folders under src are each a feature
// Under: MIT

// module imports
pub mod activations;
pub mod cli;
pub mod derivatives_optimizers_loss;
pub mod neuralnet_impls;
pub mod storage_op;
pub mod tensor_op;
pub mod tensor_std_op;
pub mod wasm_features;

#[derive(Clone, Debug, PartialEq, PartialOrd, Copy)]
pub struct Shape {
    pub x: usize, // Rows →
    pub y: usize, // Columns ↓
}

pub fn is_even_usize(num: usize) -> bool {
    num % 2 == 0
}

pub fn average_2_f32(num1: f32, num2: f32) -> f32 {
    (num1 + num2) / 2.0
}

//((self.layer_length - 1) / 2) + 1; OLD idea
pub fn odd_median_usize(length: usize) -> usize {
    ((length - 1) / 2) + 1
}

// Ram tensor, TODO UPDATE STORAGE BASED
// TODO transfer to storage or some type of direct storage connection.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RamTensor {
    pub shape: Shape,
    pub layer_length: usize,
    pub data: Vec<Vec<Vec<f32>>>, // matrices, rows, cols, values
}

pub fn find_good_breakup(neural_units: usize) -> usize {
    if is_even_usize(neural_units) {
        neural_units / 2
    } else {
        neural_units
    }
}

/*
//TODO better and more Unit Tests
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ram_tensor_testing() {
        let mut weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);
        let mut bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);

        weights = cus_act!(weights, |x| x + 2.0);
        bias = weights.clone();
        let correct_array_shift2: [[[f32; 3]; 3]; 2] = [
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        ];
        assert_eq!(weights.data, correct_array_shift2);

        weights = weights.matmul(bias).unwrap();
        let correct_array_matmul: [[[f32; 3]; 3]; 2] = [
            [[12.0, 12.0, 2.0], [12.0, 12.0, 12.0], [12.0, 12.0, 12.0]],
            [[12.0, 12.0, 2.0], [12.0, 12.0, 12.0], [12.0, 12.0, 12.0]],
        ];
        assert_eq!(weights.data, correct_array_matmul);
    }
}
*/

use std::convert::From;

impl From<RamTensor> for f32 {
    fn from(tensor: RamTensor) -> Self {
        tensor
            .data
            .first()
            .unwrap()
            .first()
            .unwrap()
            .first()
            .copied()
            .unwrap_or(0.0)
    }
}
