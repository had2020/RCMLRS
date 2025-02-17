use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);
    //let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 2, y: 2 }, 1);
    //matrix_multiplication(&memory, weights, bias);
    println!("{:?}", weights.data);
    let weights2: RamTensor = weights.matmul(weights.clone()).unwrap();
    println!("{:?}", weights2.data);
}
