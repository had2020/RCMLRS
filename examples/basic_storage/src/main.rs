use rcmlrs::*;

/*
fn forward(tensor: Tensor) -> Tensor {
}
*/

fn main() {
    let mut memory = Memory::new("test");

    // TODO input layer
    let weights: Tensor = Tensor::new_layer_zeros(&mut memory, Shape { x: 2, y: 2 }, 1);
    // Note: that a very high number of tensors, may not go will, it is best to layer, for larger processes
    let bias: Tensor = Tensor::new_layer_zeros(&mut memory, Shape { x: 2, y: 2 }, 1);

    matrix_multiplication(&memory, weights, bias);

    clear_save(&memory);
}
