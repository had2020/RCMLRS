use RCMLRS::*;

fn main() {
    // TODO possibly replace with macros for easier use?
    let mut memory = MemoryDatabase::open_create("database.db");

    create_tensors(&mut memory, Shape { x: 2, y: 4 });
}
