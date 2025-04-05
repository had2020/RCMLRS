use crate::*;
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;

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

    /// single threaded scaler on RamTensor
    pub fn st_scaler(&self, scaler: f32) -> Self {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
        for matrix in 0..self.layer_length {
            new_data.push(vec![]);
            for row in 0..self.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..self.shape.y {
                    new_data[matrix][row].push(self.data[matrix][row][col] * scaler);
                }
            }
        }
        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data,
        }
    }

    /// Use to input inputs into a layer
    pub fn insert_matrix(&self, layer_index: usize, new_layer: Vec<Vec<f32>>) -> Self {
        let mut new_data = self.data.clone();
        new_data[layer_index] = new_layer;
        RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: new_data.clone(),
        }
    }

    pub fn new_random(shape: Shape, layer_length: usize, rand_min: f32, rand_max: f32) -> Self {
        if shape.x == 0 || shape.y == 0 || layer_length == 0 {
            eprintln!("Error, on RamTensor creation, shape and layer_length start at 1, not 0. Please select 1 or greator!");
        }

        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        let mut baseline_matrix: Vec<Vec<f32>> = vec![];

        let mut rng = rand::rng();

        for _row in 0..shape.x {
            let mut current_row: Vec<f32> = vec![];
            for _col in 0..shape.y {
                let value = rng.random_range(rand_min..rand_max);
                current_row.push(value);
            }
            baseline_matrix.push(current_row);
        }

        for _matrice in 0..layer_length {
            new_data.push(baseline_matrix.clone());
        }

        RamTensor {
            shape: shape,
            data: new_data,
            layer_length: layer_length,
        }
    }

    pub fn new_layer_zeros(shape: Shape, layer_length: usize) -> Self {
        if shape.x == 0 || shape.y == 0 || layer_length == 0 {
            eprintln!("Error, on RamTensor creation, shape and layer_length start at 1, not 0. Please select 1 or greator!");
        }

        let zero_value: f32 = 0.0;
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
        let mut baseline_matrix: Vec<Vec<f32>> = vec![];

        for _row in 0..shape.x {
            let mut current_row: Vec<f32> = vec![];
            for _col in 0..shape.y {
                current_row.push(zero_value);
            }
            baseline_matrix.push(current_row);
        }

        for _matrice in 0..layer_length {
            new_data.push(baseline_matrix.clone());
        }

        RamTensor {
            shape: shape,
            data: new_data,
            layer_length: layer_length,
        }
    }

    // ram based Matrix Multiplication, single threaded
    pub fn st_matmul(&self, another_tensor: RamTensor) -> Result<RamTensor, String> {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
        //let shared_matrix: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
        if (self.shape.x == another_tensor.shape.x)
            && (self.shape.y == another_tensor.shape.y)
            && (self.layer_length == another_tensor.layer_length)
        {
            // rows times columns
            for (matrix_index, matrix) in self.data.iter().enumerate() {
                new_data.push(vec![]);
                for (row_index, _row) in matrix.iter().enumerate() {
                    new_data[matrix_index].push(vec![]);
                    for col_index in 0..self.shape.y {
                        new_data[matrix_index][row_index].push(
                            self.data[matrix_index][row_index][col_index]
                                * another_tensor.data[matrix_index][row_index][col_index],
                        );
                    }
                }
            }
            Ok(RamTensor {
                shape: self.shape.clone(),
                layer_length: self.layer_length,
                data: new_data,
            })
        } else {
            Err(String::from("Cannot multiply matrixs of differing sizes"))
        }
    }

    /// multi threaed, matrix multiplcation
    pub fn matmul(&self, another_tensor: RamTensor) -> Result<RamTensor, String> {
        let row_shape = self.shape.x;
        let col_shape = self.shape.y;
        let layer_length = self.layer_length;

        if (self.shape.x != another_tensor.shape.x)
            || (self.shape.y != another_tensor.shape.y)
            || (self.layer_length != another_tensor.layer_length)
        {
            return Err(String::from("cannot multiply matrices of differing sizes"));
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

        Ok(RamTensor {
            shape: self.shape.clone(),
            layer_length: self.layer_length,
            data: Arc::try_unwrap(shared_data).unwrap().into_inner().unwrap(),
        })
    }

    pub fn pad_matmul_to_another(&self, another_tensor: RamTensor) -> RamTensor {
        if self.shape != another_tensor.shape {
            self.pad(another_tensor.shape, another_tensor.layer_length, 0.0)
                .matmul(another_tensor)
                .unwrap()
        } else {
            self.matmul(another_tensor).unwrap()
        }
    }

    pub fn sum(&self) -> f32 {
        let mut sum: f32 = 0.0;
        for matrix in 0..self.layer_length {
            for row in 0..self.shape.x {
                for col in 0..self.shape.y {
                    sum += self.data[matrix][row][col];
                }
            }
        }
        sum
    }

    /// single threaded matrix addition
    pub fn st_add(&self, another_tensor: RamTensor) -> Result<RamTensor, String> {
        //let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        if (self.shape.x == another_tensor.shape.x)
            && (self.shape.y == another_tensor.shape.y)
            && (self.layer_length == another_tensor.layer_length)
        {
            let mut new_data: Vec<Vec<Vec<f32>>> = vec![];
            for matrix in 0..self.layer_length {
                new_data.push(vec![]);
                for row in 0..self.shape.x {
                    new_data[matrix].push(vec![]);
                    for col in 0..self.shape.y {
                        new_data[matrix][row].push(
                            self.data[matrix][row][col] + another_tensor.data[matrix][row][col],
                        );
                    }
                }
            }

            Ok(RamTensor {
                shape: self.shape.clone(),
                layer_length: self.layer_length,
                data: new_data,
            })
        } else {
            Err(String::from("Cannot add matrixs of differing sizes"))
        }
    }

    pub fn sub(&self, another_tensor: RamTensor) -> Result<RamTensor, String> {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        if (self.shape.x == another_tensor.shape.x)
            && (self.shape.y == another_tensor.shape.y)
            && (self.layer_length == another_tensor.layer_length)
        {
            for matrix in 0..self.layer_length {
                new_data.push(vec![]);
                for row in 0..self.shape.x {
                    new_data[matrix].push(vec![]);
                    for col in 0..self.shape.y {
                        let result: f32 =
                            self.data[matrix][row][col] - another_tensor.data[matrix][row][col];

                        new_data[matrix][row][col] = result;
                    }
                }
            }

            Ok(RamTensor {
                shape: self.shape.clone(),
                layer_length: self.layer_length,
                data: new_data,
            })
        } else {
            Err(String::from("Cannot subtract matrixs of differing sizes"))
        }
    }

    pub fn flatten(&self) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..self.layer_length {
            new_data.push(vec![vec![]]);
            for row in 0..self.shape.x {
                for col in 0..self.shape.y {
                    new_data[matrix][0].push(self.data[matrix][row][col]);
                }
            }
        }

        RamTensor {
            shape: Shape {
                x: new_data[0].len(),
                y: 1,
            },
            layer_length: new_data.len(),
            data: new_data,
        }
    }

    pub fn to_scalar(&self) -> Result<f32, String> {
        if self.shape.x == 1 && self.shape.y == 1 && self.layer_length == 1 {
            Ok(self.data[0][0][0])
        } else {
            Err(String::from(
                "Tensor must be of dims x:1,y:1, and have a layer_length of 1",
            ))
        }
    }

    pub fn mean(&self) -> f32 {
        let mut data_sum: f32 = 0.0;
        for matrix in 0..self.layer_length {
            for row in 0..self.shape.x {
                for col in 0..self.shape.y {
                    data_sum += self.data[matrix][row][col];
                }
            }
        }
        let dataset_indexs = (self.shape.x * self.shape.y) * self.layer_length;
        data_sum / dataset_indexs as f32
    }

    pub fn median(&self) -> f32 {
        let mut returned_median: f32 = 0.0;
        // cases
        let matrix_even = is_even_usize(self.layer_length);
        let row_even = is_even_usize(self.shape.x);
        let col_even = is_even_usize(self.shape.y);

        match (matrix_even, row_even, col_even) {
            (true, true, true) => {
                let mi1 = self.layer_length / 2;
                let mi2 = (self.layer_length / 2) + 1;
                let ri1 = self.shape.x / 2;
                let ri2 = (self.shape.x / 2) + 1;
                let col1 = self.shape.y / 2;
                let col2 = (self.shape.y / 2) + 1;

                let first_p = self.data[mi1][ri1][col1];
                let second_p = self.data[mi2][ri2][col2];

                returned_median = average_2_f32(first_p, second_p);
            }
            (false, true, true) => {
                let mi = odd_median_usize(self.layer_length);
                let ri1 = self.shape.x / 2;
                let ri2 = (self.shape.x / 2) + 1;
                let col1 = self.shape.y / 2;
                let col2 = (self.shape.y / 2) + 1;

                let first_p = self.data[mi][ri1][col1];
                let second_p = self.data[mi][ri2][col2];

                returned_median = average_2_f32(first_p, second_p);
            }
            (false, false, true) => {
                let mi = odd_median_usize(self.layer_length);
                let ri = odd_median_usize(self.shape.x);
                let col1 = self.shape.y / 2;
                let col2 = (self.shape.y / 2) + 1;

                let first_p = self.data[mi][ri][col1];
                let second_p = self.data[mi][ri][col2];

                returned_median = average_2_f32(first_p, second_p);
            }
            (false, false, false) => {
                let mi = odd_median_usize(self.layer_length);
                let ri = odd_median_usize(self.shape.x);
                let col = odd_median_usize(self.shape.y);

                returned_median = self.data[mi][ri][col];
            }
            (false, true, false) => {
                let mi = odd_median_usize(self.layer_length);
                let ri1 = self.shape.x / 2;
                let ri2 = (self.shape.x / 2) + 1;
                let col = odd_median_usize(self.shape.y);

                let first_p = self.data[mi][ri1][col];
                let second_p = self.data[mi][ri2][col];

                returned_median = average_2_f32(first_p, second_p);
            }
            (true, true, false) => {
                let mi1 = self.layer_length / 2;
                let mi2 = (self.layer_length / 2) + 1;
                let ri1 = self.shape.x / 2;
                let ri2 = (self.shape.x / 2) + 1;
                let col = odd_median_usize(self.shape.y);

                let first_p = self.data[mi1][ri1][col];
                let second_p = self.data[mi2][ri2][col];

                returned_median = average_2_f32(first_p, second_p);
            }
            _ => (),
        };
        returned_median
    }

    /// pads tensor based on shape, and layer_length shape
    /// will cut data useful for padding.
    /// It is prefered to use Dense in ML
    pub fn pad(&self, to_shape: Shape, to_layer_length_shape: usize, pad_value: f32) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix_index in 0..to_layer_length_shape {
            new_data.push(vec![]);

            if matrix_index >= self.data.len() {
                continue;
            }

            for row in 0..to_shape.x {
                new_data[matrix_index].push(vec![]);

                if self.shape.x <= row {
                    for _col in 0..to_shape.y {
                        new_data[matrix_index][row].push(pad_value);
                    }
                } else {
                    for col in 0..to_shape.y {
                        if self.shape.y <= col {
                            new_data[matrix_index][row].push(pad_value);
                        } else {
                            if row >= self.data[matrix_index].len() {
                                continue;
                            }
                            if col >= self.data[matrix_index][row].len() {
                                continue;
                            }

                            new_data[matrix_index][row].push(self.data[matrix_index][row][col]);
                        }
                    }
                }
            }
        }

        RamTensor {
            shape: to_shape,
            layer_length: to_layer_length_shape,
            data: new_data,
        }
    }

    // turn one number into a tensor
    pub fn from<T: Into<f32>>(value: T) -> Self {
        Self {
            data: vec![vec![vec![value.into()]]],
            shape: Shape { x: 1, y: 1 },
            layer_length: 1,
        }
    }

    pub fn retrive_matrice(&self, matrix: usize, row: usize, col: usize) -> f32 {
        self.data[matrix][row][col]
    }

    pub fn scaler_to_f32(&self) -> f32 {
        self.data[0][0][0]
    }

    //TODO transpose aka rotate matrix

    /// population standard deviation or sample, and outputs variance
    pub fn std(&self, sample: bool) -> f32 {
        let mut variance: f32 = 0.0;
        let flattened_self = self.flatten();
        let n = flattened_self.data.len();

        if n == 0 {
            0.0
        } else {
            let mean = self.sum() / n as f32;

            let mut squared_differnces: Vec<f32> = vec![];
            for x in 0..flattened_self.shape.y {
                squared_differnces.push((x as f32 - mean).powf(2.0));
            }

            // returning variance
            if sample {
                variance = squared_differnces.iter().copied().sum::<f32>() / (n as f32 - 1.0);
            } else {
                variance = squared_differnces.iter().copied().sum::<f32>() / n as f32;
            }

            variance
        }
    }

    pub fn normalize(&self) -> RamTensor {
        let mean: f32 = self.mean();
        let std: f32 = self.std(true);
        (self.clone() - mean) / std
    }
}
