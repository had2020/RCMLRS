use RCMLRS::*;

fn main() {
    let mut memory = MemoryDatabase::open_create("database.db");

    create_tensors(&mut memory, Shape { x: 2, y: 4 });
}
