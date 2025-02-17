use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);
    //let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 2, y: 2 }, 1);
    //matrix_multiplication(&memory, weights, bias);
    println!(" before: {:?}", weights.data);
    let weights2: RamTensor = weights.matmul(weights.clone()).unwrap();
    println!("after: {:?}", weights2.data);

    // Relu
    let relued: RamTensor = weights2.relu();
    println!("Relued: {:?}", relued.data);
}
