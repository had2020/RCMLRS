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
    pub path: String,
    pub current_layer: usize,
}

impl Memory {
    pub fn new(path: String) -> Self {
        Memory {
            path: path,
            current_layer: 0,
        }
    }
    pub fn save_tensor(&mut self, shape: Shape) -> Result<(), std::io::Error> {
        let file_path = format!("{}.layer_{}.txt", self.path, self.current_layer); // More descriptive file names
        fs::write(file_path, "shape_string")?; // Use ? for error propagation
        self.current_layer += 1;
        Ok(())
    }
}

/*
pub fn save_tensor(memory: &mut Memory, shape: Shape) {
    let file_path = memory.path.to_string() + ".txt";
    fs::write(file_path, b"Lorem ipsum").unwrap();
    //memory.current_layer
}
*/
