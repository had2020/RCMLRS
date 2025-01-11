pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
use std::fs::{self};

// TODO documation like code on top

#[macro_export]
macro_rules! init_memory {
    ($name:expr) => {
        println!("{$name}")
    };
}

pub struct Matrix {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

pub struct Shape {
    pub x: usize,
    pub y: usize,
}

pub struct Memory {
    pub dir_name: String,
    pub current_layer: usize,
}

pub fn dir_exists(path: &str) -> std::io::Result<()> {
    match fs::metadata(path) {
        Ok(_) => println!("File exists!"),
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
            Ok(_) => println!("Memory dir created"),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    println!("Using old Memory dir");
                } else {
                    println!("An error occurred: {}", e);
                }
            }
        }

        Memory {
            dir_name: dir.to_string(),
            current_layer: 0,
        }
    }
}

/*
pub fn save_tensor(memory: &mut Memory, shape: Shape) {
    let file_path = memory.path.to_string() + ".txt";
    fs::write(file_path, b"Lorem ipsum").unwrap();
    //memory.current_layer
}
*/

pub struct Tensor {
    pub id: usize,
    pub shape: Shape,
}

pub fn load_tensor(file_path: String) {
    //file_contents = std::fs::read(file_path);
}

pub fn matrix_to_string(matrix: Matrix) {
    for row in matrix.data {
        println!(">");
        for cols in row {
            println!("X");
        }
    }
}

pub fn save_tensor(memory: &mut Memory, shape: Shape) {
    let file_path = format!("{}/{}_layer.txt", memory.dir_name, memory.current_layer);

    //let infomation_to_write()

    match std::fs::write(file_path, "shape_string") {
        Ok(_) => println!("write"),
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                println!("Not found");
            } else {
                println!("An error occurred: {}", e);
            }
        }
    }
    //.expect("Failed to write file. Check the file path and permissions.");
    //println!("{file_path}"); // TODO create file to avoid no file err
    memory.current_layer += 1;
}
