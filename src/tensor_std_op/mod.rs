// RCMLRS tensor-std-operations
// By: Hadrian Lazic
// Under: MIT

use crate::*;
//use rand::Rng;

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

//use rand::prelude::*; // for random tensor gen
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub}; // for rust's operations to work on RamTensors // TODO  SubAssign
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
        //println!("1:{:?}, 2:{:?}", self.shape, rhs.shape);

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

impl Div<f32> for RamTensor {
    type Output = RamTensor;

    fn div(self, rhs: f32) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..self.layer_length {
            new_data.push(vec![]);
            for row in 0..self.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..self.shape.y {
                    new_data[matrix][row].push(rhs / self.data[matrix][row][col]);
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
