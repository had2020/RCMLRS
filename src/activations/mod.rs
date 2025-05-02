use crate::*;

//activation functions for non-linearity
impl RamTensor {
    ///ReLU
    pub fn relu(&self) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for (matrix_index, matrix) in self.data.iter().enumerate() {
            new_data.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                new_data[matrix_index].push(vec![]);
                for x in row {
                    if x > &0.0 {
                        new_data[matrix_index][row_index].push(x.clone());
                    } else {
                        new_data[matrix_index][row_index].push(0.0);
                    }
                }
            }
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data,
        }
    }

    ///Leaky ReLU
    pub fn lrelu(&self, negative_slope: f32) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for (matrix_index, matrix) in self.data.iter().enumerate() {
            new_data.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                new_data[matrix_index].push(vec![]);
                for x in row {
                    if x > &0.0 {
                        new_data[matrix_index][row_index].push(x.clone());
                    } else {
                        new_data[matrix_index][row_index].push(negative_slope * x);
                    }
                }
            }
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data,
        }
    }

    ///Sigmoid
    pub fn sigmoid(&self) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
        let e = std::f32::consts::E; // Euler's number

        for (matrix_index, matrix) in self.data.iter().enumerate() {
            new_data.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                new_data[matrix_index].push(vec![]);
                for x in row {
                    let denominater = 1.0 + (e.powf(-x.clone()));
                    new_data[matrix_index][row_index].push(1.0 / denominater);
                }
            }
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data,
        }
    }

    ///Tanh
    pub fn tanh(&self) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
        let e = std::f32::consts::E; // Euler's number

        for (matrix_index, matrix) in self.data.iter().enumerate() {
            new_data.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                new_data[matrix_index].push(vec![]);
                for x in row {
                    let numerator = e.powf(x.clone()) - e.powf(-x.clone());
                    let denominater = e.powf(x.clone()) + e.powf(-x.clone());
                    new_data[matrix_index][row_index].push(numerator / denominater);
                }
            }
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data,
        }
    }

    ///Softmax
    pub fn softmax(&self) -> RamTensor {
        let mut probabilities: Vec<Vec<Vec<f32>>> = vec![];
        let mut exponentials: Vec<Vec<Vec<f32>>> = vec![];
        let e = std::f32::consts::E; // Euler's number
        let mut sum_exp: f32 = 0.0;

        for (matrix_index, matrix) in self.data.iter().enumerate() {
            exponentials.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                exponentials[matrix_index].push(vec![]);

                for x in row {
                    let exp_value = e.powf(x.clone());
                    exponentials[matrix_index][row_index].push(exp_value);
                    sum_exp += exp_value;
                }
            }

            // normalize for probabilities
            for (matrix_index, matrix) in exponentials.iter().enumerate() {
                probabilities.push(vec![]);

                for (row_index, row) in matrix.iter().enumerate() {
                    probabilities[matrix_index].push(vec![]);

                    for x in row {
                        probabilities[matrix_index][row_index].push(x / sum_exp);
                    }
                }
            }
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: probabilities,
        }
    }

    ///Swish
    pub fn swish(&self) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
        let e = std::f32::consts::E; // Euler's number

        for (matrix_index, matrix) in self.data.iter().enumerate() {
            new_data.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                new_data[matrix_index].push(vec![]);
                for x in row {
                    new_data[matrix_index][row_index].push(x * (1.0 / (1.0 + e.powf(-x.clone()))));
                }
            }
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data,
        }
    }

    ///GELU (Gaussian Error Linear Unit)
    pub fn gelu(&self) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
        let frac_2_sqrt_pi = std::f32::consts::FRAC_2_SQRT_PI;
        let e = std::f32::consts::E; // Euler's number

        for (matrix_index, matrix) in self.data.iter().enumerate() {
            new_data.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                new_data[matrix_index].push(vec![]);
                for x in row {
                    let tanh_numerator = e.powf(x.clone()) - e.powf(-x.clone());
                    let tanh_denominater = e.powf(x.clone()) + e.powf(-x.clone());

                    let tanh = tanh_numerator / tanh_denominater;
                    let a1: f32 = 0.044715;

                    let value = 0.5
                        * x.clone()
                        * ((1.0 + tanh) * (frac_2_sqrt_pi as f32 * (x + (a1 * x).powf(3.0))));

                    new_data[matrix_index][row_index].push(value);
                }
            }
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data,
        }
        // TODO ELU or Mish
        // AGI: AReLU,SELU,
        // Cross-Entropy Loss
        // Mean Squared Error (MSE)
    }
}

/*
/// erf, Error Function
pub fn erf(x: f32) -> f32 {
let pi = std::f64::consts::PI;

let mut n = 1000.0;
let delta_t = x / n;
let mut sum = 0;

for i in n {
    let t = i * delta_t;
    sum += expf32(-t.pow)
}
}
*/

#[macro_export]
macro_rules! cus_act {
    ($ramtensor:expr, $code:expr) => {{
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for (matrix_index, matrix) in $ramtensor.data.iter().enumerate() {
            new_data.push(vec![]);

            for (row_index, row) in matrix.iter().enumerate() {
                new_data[matrix_index].push(vec![]);
                for &x in row {
                    let result = $code(x);
                    new_data[matrix_index][row_index].push(result);
                }
            }
        }

        RamTensor {
            shape: $ramtensor.shape,
            layer_length: $ramtensor.layer_length,
            data: new_data,
        }
    }};
}
