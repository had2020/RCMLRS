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
#[derive(Clone, Debug)]
pub struct RamTensor {
    pub shape: Shape,
    pub layer_length: usize,
    pub data: Vec<Vec<Vec<f32>>>, // matrices, rows, cols, values
}

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

///You can use this to input mannully whole tensor data.
///Notice you will need to put zeros for blank data that is made with a shape.
///It is better to enter a smaller size that is all your data, or have zero handling, and resize the tensor, or insert.
pub fn raw_input_tensor_matrices(
    input_layer_length: usize, // To break your data into smaller matrices.
    input_shape: Shape,
    input_matrices: Vec<Vec<Vec<f32>>>,
) -> RamTensor {
    RamTensor {
        shape: input_shape,
        layer_length: input_layer_length,
        data: input_matrices,
    }
}

///Prefer using raw_input_tensor_matrices, and resizing if needed.
pub fn zeroed_input_tensor_matrices(
    input_layer_length: usize, // To break your data into smaller matrices.
    input_shape: Shape,
    input_matrices: Vec<Vec<Vec<f32>>>,
) -> RamTensor {
    let zeros: f32 = 0.0; //constant
    let mut new_input_matrices: Vec<Vec<Vec<f32>>> = vec![];

    if input_matrices.len() != input_layer_length {
        let mut empty_baseline: Vec<Vec<f32>> = vec![];

        for row in 0..input_shape.x {
            empty_baseline.push(vec![]);
            for _ in 0..input_shape.y {
                empty_baseline[row].push(zeros);
            }
        }

        for _ in 0..(input_layer_length - input_matrices.len()) {
            new_input_matrices.push(empty_baseline.clone());
        }
    }

    for matrix in input_matrices {
        for (row_index, row) in matrix.iter().enumerate() {
            if row.len() < input_shape.y {
                // insert col into row
                new_input_matrices[row_index].push(vec![]);
                for _ in 0..(input_shape.x - row.len()) {
                    for _ in 0..input_shape.y {
                        new_input_matrices[row_index].push(vec![zeros]);
                    }
                }
            }
        }
    }

    RamTensor {
        shape: input_shape,
        layer_length: input_layer_length,
        data: new_input_matrices,
    }
}

use rand::prelude::*; // for random tensor gen
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign}; // for rust's operations to work on RamTensors
use std::sync::{Arc, Mutex};
use std::thread; // for effiency

impl AddAssign<RamTensor> for RamTensor {
    fn add_assign(&mut self, other: RamTensor) {
        let result = self.st_add(other).unwrap();
        self.data = result.data;
    }
}

impl Sub<RamTensor> for f32 {
    type Output = RamTensor;

    fn sub(self, rhs: RamTensor) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..rhs.layer_length {
            new_data.push(vec![]);
            for row in 0..rhs.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..rhs.shape.y {
                    new_data[matrix][row].push(self - rhs.data[matrix][row][col]);
                }
            }
        }

        RamTensor {
            shape: rhs.shape,
            layer_length: rhs.layer_length,
            data: new_data,
        }
    }
}

impl Sub<f32> for RamTensor {
    type Output = RamTensor;

    fn sub(self, num: f32) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..self.layer_length {
            new_data.push(vec![]);
            for row in 0..self.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..self.shape.y {
                    new_data[matrix][row].push(self.data[matrix][row][col] - num);
                }
            }
        }

        RamTensor {
            shape: self.shape,
            layer_length: self.layer_length,
            data: new_data,
        }
    }
}

impl Sub<RamTensor> for RamTensor {
    type Output = RamTensor;

    fn sub(self, rhs: RamTensor) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..rhs.layer_length {
            new_data.push(vec![]);
            for row in 0..rhs.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..rhs.shape.y {
                    new_data[matrix][row]
                        .push(self.data[matrix][row][col] - rhs.data[matrix][row][col]);
                }
            }
        }

        RamTensor {
            shape: rhs.shape,
            layer_length: rhs.layer_length,
            data: new_data,
        }
    }
}

impl Add<RamTensor> for f32 {
    type Output = RamTensor;

    fn add(self, rhs: RamTensor) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..rhs.layer_length {
            new_data.push(vec![]);
            for row in 0..rhs.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..rhs.shape.y {
                    new_data[matrix][row].push(self + rhs.data[matrix][row][col]);
                }
            }
        }

        RamTensor {
            shape: rhs.shape,
            layer_length: rhs.layer_length,
            data: new_data,
        }
    }
}

impl Add<f32> for RamTensor {
    type Output = RamTensor;

    fn add(self, rhs: f32) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..self.layer_length {
            new_data.push(vec![]);
            for row in 0..self.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..self.shape.y {
                    new_data[matrix][row].push(rhs + self.data[matrix][row][col]);
                }
            }
        }

        RamTensor {
            shape: self.shape,
            layer_length: self.layer_length,
            data: new_data,
        }
    }
}

impl Add<RamTensor> for RamTensor {
    type Output = RamTensor;

    fn add(self, tensor: RamTensor) -> Self::Output {
        tensor.st_add(self).unwrap()
    }
}

impl Div<RamTensor> for f32 {
    type Output = RamTensor;

    fn div(self, rhs: RamTensor) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..rhs.layer_length {
            new_data.push(vec![]);
            for row in 0..rhs.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..rhs.shape.y {
                    new_data[matrix][row].push(self / rhs.data[matrix][row][col]);
                }
            }
        }

        RamTensor {
            shape: rhs.shape,
            layer_length: rhs.layer_length,
            data: new_data,
        }
    }
}

impl Mul<f32> for RamTensor {
    type Output = RamTensor;

    fn mul(self, num: f32) -> Self::Output {
        self.scaler(num)
    }
}

impl Mul<RamTensor> for f32 {
    type Output = RamTensor;

    fn mul(self, tensor: RamTensor) -> Self::Output {
        tensor.scaler(self)
    }
}

impl Mul<RamTensor> for RamTensor {
    type Output = RamTensor;

    fn mul(self, another_tensor: RamTensor) -> Self::Output {
        let row_shape = self.shape.x;
        let col_shape = self.shape.y;
        let layer_length = self.layer_length;

        if (self.shape.x != another_tensor.shape.x)
            || (self.shape.y != another_tensor.shape.y)
            || (self.layer_length != another_tensor.layer_length)
        {
            println!("cannot multiply matrices of differing sizes");
        }

        let shared_data = Arc::new(Mutex::new(vec![
            vec![vec![0.0; col_shape]; row_shape];
            layer_length
        ]));

        let mut handles = vec![];

        for matrix_index in 0..layer_length {
            let shared_data_clone = Arc::clone(&shared_data);
            let self_matrix = self.data[matrix_index].clone();
            let another_matrix = another_tensor.data[matrix_index].clone();

            let handle = thread::spawn(move || {
                let mut data = shared_data_clone.lock().unwrap();
                for row_index in 0..row_shape {
                    for col_index in 0..col_shape {
                        data[matrix_index][row_index][col_index] = self_matrix[row_index]
                            [col_index]
                            * another_matrix[row_index][col_index];
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: Arc::try_unwrap(shared_data).unwrap().into_inner().unwrap(),
        }
    }
}

impl Neg for RamTensor {
    type Output = RamTensor;

    fn neg(self) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..self.layer_length {
            new_data.push(vec![]);
            for row in 0..self.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..self.shape.y {
                    new_data[matrix][row].push(-self.data[matrix][row][col]);
                }
            }
        }

        RamTensor {
            shape: self.shape,
            layer_length: self.layer_length,
            data: new_data,
        }
    }
}

impl RamTensor {
    pub fn scaler(self, num: f32) -> Self {
        let row_shape = self.shape.x;
        let col_shape = self.shape.y;

        let mut handles = vec![];

        for matrix in self.data {
            let handle = thread::spawn(move || {
                let mut new_matrix = vec![vec![0.0; col_shape]; row_shape];
                for row in 0..row_shape {
                    for col in 0..col_shape {
                        new_matrix[row][col] = matrix[row][col] * num;
                    }
                }
                new_matrix
            });

            handles.push(handle);
        }

        let mut result_data = vec![];
        for handle in handles {
            let result_layer = handle.join().unwrap();
            result_data.push(result_layer);
        }

        RamTensor {
            shape: self.shape,
            layer_length: result_data.len(),
            data: result_data,
        }
    }
