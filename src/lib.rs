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
use std::fs;

// TODO documation like code on top
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

pub fn save_tensor(memory: &mut Memory, shape: Shape) {
    match fs::create_dir("") {
        Ok(_) => println!("Memory dir created"),
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                println!("Using old Memory dir");
            } else {
                println!("An error occurred: {}", e);
            }
        }
    }

    let file_path = format!("/{}/{}_layer.txt", memory.dir_name, memory.current_layer);
    //let file_path = "{memory.dir_name}/{memory.current_layer}_layer.txt";
    std::fs::write(file_path, "shape_string").unwrap();
    //println!("{file_path}"); // TODO create file to avoid no file err
    memory.current_layer += 1;
}

#[macro_export]
macro_rules! init_memory {
    ($name:expr) => {
        println!("{$name}")
    };
}
