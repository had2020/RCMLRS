use rcmlrs::*;

/*
fn forward(tensor: Tensor) -> Tensor {
}
*/

fn main() {
    let mut memory = Memory::new("test");

    // TODO input layer
    let weights: Tensor = Tensor::new_layer_zeros(&mut memory, Shape { x: 2, y: 2 }, 1);
    let bias: Tensor = Tensor::new_layer_zeros(&mut memory, Shape { x: 2, y: 2 }, 1);

    /*
    println!("WEIGHTS:");
    print_tensor(&memory, weights);
    println!("BIAS:");
    print_tensor(&memory, bias);


    //clear_load(&memory);
    clear_save(&memory);
    */

    //TODO Relu
    matrix_multiplication(&memory, weights, bias);

    clear_save(&memory)
}
