use rcmlrs::*;

fn main() {
    //let mut memory = Memory::new("test");

    ////Memory::save_tensor(&mut memory, Shape { x: 3, y: 3 });
    //save_tensor(&mut memory, Shape { x: 3, y: 3 });
    let matrix_data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    let matrix = Matrix {
        name: "MyMatrix".to_string(),
        rows: matrix_data.len(),
        cols: matrix_data[0].len(),
        data: matrix_data,
    };

    matrix_to_string(matrix);
}
