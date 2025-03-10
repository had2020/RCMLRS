// RCMLRS
// By: Hadrian Lazic

#[macro_export] // TODO delete
macro_rules! init_memory {
    ($name:expr) => {
        println!("{$name}")
    };
} // delete later

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

pub struct Memory {
    pub dir_name: String,
    pub current_layer: usize,
}

use std::f32::consts::E;
use std::fs;

pub fn dir_exists(path: &str) -> std::io::Result<()> {
    match fs::metadata(path) {
        Ok(_) => println!("File exists! ✅"),
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                println!("File does not exist.");
            } else {
                println!("An error occurred: {}", e);
            }
        }
    }
    Ok(())
}

impl Memory {
    pub fn new(dir: &str) -> Self {
        match fs::create_dir(dir) {
            Ok(_) => println!("Memory dir created ✅"),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    println!("Using existing Memory dir ✅");
                } else {
                    println!("Using existing Memory dir! ✅");
                }
            }
        }

        Memory {
            dir_name: dir.to_string(),
            current_layer: 0,
        }
    }
}

pub fn matrix_print(matrix: Matrix) {
    for row in matrix.data {
        println!(">");
        for cols in row {
            let col = format!("{cols}");
            println!("{col}");
        }
    }
}

pub fn matrix_into_string(matrix: Matrix) -> String {
    let mut matrix_string: String = String::new();
    for row in matrix.data {
        let mut first_b: bool = true; // the 'b' after the a is not needed
        matrix_string.push_str("a"); // x, row's length
        for cols in row {
            // y, Columns
            let col = format!("{cols}");
            if !first_b {
                matrix_string.push_str("b");
            } else {
                first_b = false
            }
            matrix_string.push_str(&col);
        }
    }
    matrix_string
}

use std::fs::OpenOptions;
//use std::intrinsics::expf32;
use std::io::Write;

pub fn save_matrix(memory: &mut Memory, matrix: Matrix) {
    let file_path = format!(
        "{}/saved/{}_layer.txt",
        memory.dir_name, memory.current_layer
    );

    // encode matrix to a string
    let infomation_to_write = matrix_into_string(matrix);

    // file then write
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(file_path)
        .unwrap();

    writeln!(file, "{}", infomation_to_write).unwrap();
}

use std::fs::File;
use std::io::Read;
use std::io::{BufRead, BufReader};
use std::num::FpCategory;
use std::os::unix::fs::MetadataExt;

/// Warning will load each whole line into memory in one block!
pub fn print_tensor(memory: &Memory, tensor: Tensor) -> std::io::Result<()> {
    let file_path: String;
    if tensor.saved {
        file_path = format!("{}/saved/{}_layer.txt", memory.dir_name, tensor.id);
    } else {
        file_path = format!("{}/loaded/{}_layer.txt", memory.dir_name, tensor.id);
    }

    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        println!("{}", line?); // each line is a matrix
    }
    Ok(())
}

pub fn find_point_matrix(
    tensor_file_path: String,
    stop_shape_counting_at: Shape,
    stop_at_new_line: usize,
) -> f64 {
    let file = File::open(tensor_file_path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buffer = [0; 1];

    // loop counter items
    let mut shape_counter = Shape { x: 0, y: 0 };
    let mut index_f64_value: f64 = 0.0;
    let mut index_string_value: String = "".to_string();
    let mut new_line_counter: usize = 0;

    while reader.read(&mut buffer).unwrap() > 0 {
        // needs to compare with each number
        let char = buffer[0] as char;
        // store temp operations in .temp
        println!("{}", char); // only for debug

        match char {
            'a' => {
                // new row
                shape_counter.x += 1;
                index_string_value = "".to_string(); //TODO write to temp or loaded
                index_f64_value = 0.0;
            }
            'b' => {
                // new column
                shape_counter.y += 1;
                // share if the right index
                if stop_shape_counting_at.x == shape_counter.x
                    && stop_shape_counting_at.y == shape_counter.y
                    && new_line_counter == stop_at_new_line
                {
                    break;
                }
                index_string_value = "".to_string();

                index_f64_value = 0.0;
            }
            '\n' => {
                // another whole matrix
                println!("new line");
                new_line_counter += 1;
            }
            _ => {
                index_string_value.push_str(&char.to_string());
                if index_string_value.len() > 0 {
                    index_f64_value = index_string_value.parse().unwrap();
                }
            }
        }
    }
    index_f64_value
}

// TODO single operations / and or scaler
/// Multiples a whole layer, by another, each layer stands as a Tensor.
pub fn matrix_multiplication(memory: &Memory, tensor_1: Tensor, tensor_2: Tensor) {
    if tensor_1.shape.x == tensor_2.shape.x && tensor_1.shape.y == tensor_2.shape.y {
        let read_file_path = format!("{}/saved/{}_layer.txt", &memory.dir_name, tensor_1.id);
        let file = File::open(read_file_path).unwrap();
        let mut reader = BufReader::new(file);
        let mut buffer = [0; 1];

        // loop counter items
        let mut shape_counter = Shape { x: 0, y: 0 };
        let mut index_f64_value: f64 = 0.0;
        let mut index_string_value: String = "".to_string();
        let mut new_line_counter: usize = 0;

        while reader.read(&mut buffer).unwrap() > 0 {
            // needs to compare with each number
            let char = buffer[0] as char;
            // store temp operations in .temp
            println!("{}", char); // only for debug

            match char {
                'a' => {
                    // new row
                    shape_counter.x += 1;
                    index_string_value = "".to_string(); //TODO write to temp or loaded
                    index_f64_value = 0.0;
                }
                'b' => {
                    // new column
                    shape_counter.y += 1;
                    index_string_value = "".to_string(); // TODO multiply and temp
                    let other_file_path =
                        format!("{}/saved/{}_layer.txt", &memory.dir_name, tensor_2.id);
                    find_point_matrix(
                        // TODO multiply
                        // TODO save in load file
                        other_file_path,
                        Shape {
                            x: shape_counter.x,
                            y: shape_counter.y,
                        },
                        new_line_counter,
                    );
                    // return something from find_point_matrix and use mult
                    index_f64_value = 0.0;
                    index_string_value = "".to_string();
                    index_f64_value = 0.0;
                }
                '\n' => {
                    // another whole matrix
                    println!("new line");
                    new_line_counter += 1;
                }
                _ => {
                    index_string_value.push_str(&char.to_string());
                    if index_string_value.len() > 0 {
                        index_f64_value = index_string_value.parse().unwrap();
                    }
                }
            }
            // DEBUG, TODO remove!
            println!("scanned: x:{}, y:{}", shape_counter.x, shape_counter.y);
            println!("num: {}", index_string_value);
            println!("floated: {}", index_f64_value);
            println!("__");
            // DEBUG
        }
    } else {
        eprintln!("Tensors Multipled of Differing Shapes! 🙅"); //TODO fix shape? first x and second y make new shape
    }
}

pub fn clear_all_memory(memory: &Memory) {
    let file_path = format!("{}/saved", memory.dir_name);
    fs::remove_dir_all(file_path).unwrap();
    let file_path = format!("{}/loaded", memory.dir_name);
    fs::remove_dir_all(file_path).unwrap();
}

pub fn clear_save(memory: &Memory) {
    let file_path = format!("{}/saved", memory.dir_name);
    match fs::remove_dir_all(file_path) {
        Ok(_) => println!("Cleared Saved folder. ✅"),
        Err(e) => eprintln!("Could not clear Saved folder for: {}", e),
    }
}

pub fn clear_load(memory: &Memory) {
    let file_path = format!("{}/loaded", memory.dir_name);
    fs::remove_dir_all(file_path).unwrap();
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub x: usize, // Rows →
    pub y: usize, // Columns ↓
}

/// holds single layer tensor operation qualites
#[derive(Clone)]
pub struct Tensor {
    //pub matrices: Vec<Matrix>, //TODO ram tensor
    pub saved: bool,
    pub id: usize,
    pub shape: Shape,
}

impl Tensor {
    //TODO more D then 2D, or less
    //TODO random new
    pub fn new_layer_zeros(memory: &mut Memory, shape: Shape, layer_length: usize) -> Self {
        //let mut matrices: Vec<Matrix> = vec![]; // TODO with ram tensor
        //matrices.push(matrix);

        /* // non-automatic example
        let matrix_data = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        */

        memory.current_layer = memory.current_layer + 1;

        let zero_value: f64 = 0.1;

        for layer in 0..layer_length {
            let mut matrix_data: Vec<Vec<f64>> = vec![];
            for row in 0..shape.y {
                let col: Vec<f64> = vec![zero_value; shape.x as usize];
                matrix_data.push(col);
            }

            let matrix = Matrix {
                rows: matrix_data.len(),
                cols: matrix_data[0].len(),
                data: matrix_data,
            };

            let save_path: String = format!("{}/saved", memory.dir_name);
            match fs::create_dir(save_path) {
                Ok(_) => (),
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        ();
                    } else {
                        ();
                    }
                }
            }

            let file_path = format!(
                "{}/saved/{}_layer.txt",
                memory.dir_name, memory.current_layer
            );

            // encode matrix to a string
            let infomation_to_write = matrix_into_string(matrix);

            // file then write
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(file_path)
                .unwrap();

            writeln!(file, "{}", infomation_to_write).unwrap();
        }

        Tensor {
            id: memory.current_layer,
            shape: shape,
            saved: true,
        }
    }
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
use std::ops::{Add, Div, Mul, Neg, Sub}; // for rust's operations to work on RamTensors
use std::sync::{Arc, Mutex};
use std::thread; // for effiency

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

impl Add<RamTensor> for RamTensor {
    type Output = RamTensor;

    fn add(self, rhs: RamTensor) -> Self::Output {
        let mut new_data: Vec<Vec<Vec<f32>>> = vec![];

        for matrix in 0..rhs.layer_length {
            new_data.push(vec![]);
            for row in 0..rhs.shape.x {
                new_data[matrix].push(vec![]);
                for col in 0..rhs.shape.y {
                    new_data[matrix][row]
                        .push(self.data[matrix][row][col] + rhs.data[matrix][row][col]);
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
        let row_shape = self.shape.x;
        let col_shape = self.shape.y;
        let layer_length = self.layer_length;

        let shared_data = Arc::new(Mutex::new(vec![
            vec![vec![0.0; col_shape]; row_shape];
            layer_length
        ]));

        let mut handles = vec![];

        for matrix in 0..self.layer_length {
            let shared_data_clone = Arc::clone(&shared_data);
            let self_matrix = self.data[matrix].clone();

            let handle = thread::spawn(move || {
                let mut data = shared_data_clone.lock().unwrap();
                for row_index in 0..row_shape - 1 {
                    for col_index in 0..col_shape - 1 {
                        data[matrix][row_index]
                            .push(num * self_matrix[row_index + 1][col_index + 1]);
                        // error here
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        RamTensor {
            shape: self.shape,
            layer_length: self.layer_length,
            data: Arc::try_unwrap(shared_data).unwrap().into_inner().unwrap(),
        }
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
    pub fn scaler(&self, scaler: f32) -> Self {
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

    // ram based Matrix Multiplication
    pub fn matmul(&self, another_tensor: RamTensor) -> Result<RamTensor, String> {
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

    pub fn multi_threaded_matmul(&self, another_tensor: RamTensor) -> Result<RamTensor, String> {
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

    /*
    pub fn add(&self, another_tensor: RamTensor) -> Result<RamTensor, String> {
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
                            self.data[matrix][row][col] + another_tensor.data[matrix][row][col];

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
            Err(String::from("Cannot add matrixs of differing sizes"))
        }
    }
    */

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

/// Custom activation function for ram tensor, for each element/float in matrix
/// operations must be done to x get and set a variable named "x"
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

/*
pub struct Model {

}

pub struct Network {
    pub
}
*/

// Should be like
/*
# Define a simple feedforward neural network
class NeuralNet:
    def __init__(self, input_shape, output_units):
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_units, activation='softmax')
        ])

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
        metrics = metrics or ["accuracy"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

# Example usage
if __name__ == "__main__":
    net = NeuralNet(input_shape=(784,), output_units=10)
    net.compile().train(x_train, y_train, epochs=5)
*/
