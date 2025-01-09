use RCMLRS::*;

fn main() {
    let mut memory = Memory::new("test");

    //Memory::save_tensor(&mut memory, Shape { x: 3, y: 3 });
    save_tensor(&mut memory, Shape { x: 3, y: 3 });
}
