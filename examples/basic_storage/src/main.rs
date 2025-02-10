use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(&mut memory, Shape { x: 2, y: 2 }, 1);
    let bias: RamTensor = RamTensor::new_layer_zeros(&mut memory, Shape { x: 2, y: 2 }, 1);

    //matrix_multiplication(&memory, weights, bias);
}
