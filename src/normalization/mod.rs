// RCMLRS normalization
// By: Hadrian Lazic
// Under MIT

use crate::RamTensor;

impl RamTensor {
    pub fn min_max_norm(&self) -> RamTensor {
        let mut new_tensor = self.clone();

        let min_value = self.find_min().clone();
        let max_value = self.find_max().clone();

        for matrix in 0..self.layer_length {
            for row in 0..self.shape.x {
                for col in 0..self.shape.y {
                    new_tensor.data[matrix][row][col] =
                        (new_tensor.data[matrix][row][col] - min_value) / (max_value - min_value)
                }
            }
        }

        new_tensor
    }

    pub fn z_score_norm(&self) -> RamTensor {
        let mean = self.mean();
        let std = self.std(true);
        (self.clone() - mean) / std
    }
}
