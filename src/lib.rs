#[macro_export]
macro_rules! init_memory {
    ($name:expr) => {
        println!("{$name}")
    };
} // delete later

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

pub struct Memory {
    pub dir_name: String,
    pub current_layer: usize,
}

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
        matrix_string.push_str("a");
        for cols in row {
            let col = format!("{cols}");
            matrix_string.push_str("b");
            /*
            if col.len() != 3 {
                matrix_string.push_str(&col);
            } else {
                matrix_string.push_str(&col);
            }
            */
            matrix_string.push_str(&col);
        }
    }
    //matrix_string.push_str("c");
    matrix_string
}

use std::fs::OpenOptions;
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

// TODO single operations / and or scaler
/// Multiples a whole layer, by another, each layer stands as a Tensor.
pub fn matrix_multiplication(memory: &Memory, tensor_1: Tensor, tensor_2: Tensor) {
    if tensor_1.shape.x == tensor_2.shape.x && tensor_1.shape.y == tensor_2.shape.y {
        let read_file_path = format!("{}/saved/{}_layer.txt", &memory.dir_name, tensor_1.id);
        let file = File::open(read_file_path).unwrap();
        let mut reader = BufReader::new(file);
        let mut buffer = [0; 1];

        while reader.read(&mut buffer).unwrap() > 0 {
            // needs to compare with each number
            let char = buffer[0] as char;
            if char == '\n' {
                println!("new line");
            }
            println!("{}", char);
        }
    } else {
        eprintln!("Tensors Multipled of Differing Shapes! 🙅");
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
    fs::remove_dir_all(file_path).unwrap();
}

pub fn clear_load(memory: &Memory) {
    let file_path = format!("{}/loaded", memory.dir_name);
    fs::remove_dir_all(file_path).unwrap();
}

#[derive(Clone)]
pub struct Shape {
    pub x: usize,
    pub y: usize,
}

// representants single layer tensor operation needs
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

        let zero_value: f64 = 0.0;

        // incap inside for loop for layer_length?
        for layer in 0..layer_length {
            let mut matrix_data: Vec<Vec<f64>> = vec![];
            for row in 0..shape.y {
                let col: Vec<f64> = vec![zero_value; shape.x as usize + 1];
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
