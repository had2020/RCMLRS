use rcmlrs::*;

fn main() {
    let mut memory = Memory::new("test");

    ////Memory::save_tensor(&mut memory, Shape { x: 3, y: 3 });
    //save_tensor(&mut memory, Shape { x: 3, y: 3 });
    let matrix_data = vec![
        vec![1.5, 2.1, 3.2],
        vec![4.0, 5.2, 6.4],
        vec![7.6, 8.7, 9.9],
    ];

    let matrix = Matrix {
        name: "MyMatrix".to_string(),
        rows: matrix_data.len(),
        cols: matrix_data[0].len(),
        data: matrix_data,
    };

    matrix_print(matrix);
    memory.current_layer += 1;
}
