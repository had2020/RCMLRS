use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 2, y: 2 }, 3);
    //let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 2, y: 2 }, 1);
    //matrix_multiplication(&memory, weights, bias);
    println!("{:?}", weights.matrix);
}
