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

        for row in 0..shape.x {
            let mut current_row: Vec<f32> = vec![];
            for col in 0..shape.y {
                let value = rng.random_range(rand_min..rand_max);
                current_row.push(value);
            }
            baseline_matrix.push(current_row);
        }

        for matrice in 0..layer_length {
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

        for row in 0..shape.x {
            let mut current_row: Vec<f32> = vec![];
            for col in 0..shape.y {
                current_row.push(zero_value);
            }
            baseline_matrix.push(current_row);
        }

        for matrice in 0..layer_length {
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
                for (row_index, row) in matrix.iter().enumerate() {
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
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

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

    /// resizes tensor based on shape, and layer_length shape
    /// will cut data useful for padding.
    /// It is prefered to use Dense in ML
    pub fn resize(
        &self,
        to_shape: Shape,
        to_layer_length_shape: usize,
        pad_value: f32,
    ) -> RamTensor {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix_index in 0..to_layer_length_shape {
            new_data.push(vec![]);
            for row in 0..to_shape.x {
                new_data[matrix_index].push(vec![]);

                if self.shape.x < row {
                    for col in 0..to_shape.y {
                        new_data[matrix_index][row].push(pad_value);
                    }
                } else {
                    for col in 0..to_shape.y {
                        if self.shape.y > col {
                            new_data[matrix_index][row].push(pad_value);
                        } else {
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

    //activation functions for non-linearity

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
        let e = std::f32::consts::E; // Eular's number

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
        let e = std::f32::consts::E; // Eular's number

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
        let e = std::f32::consts::E; // Eular's number
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
        let e = std::f32::consts::E; // Eular's number

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
        let e = std::f32::consts::E; // Eular's number

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
