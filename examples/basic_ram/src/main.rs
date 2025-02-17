use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);
    let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);
    //matrix_multiplication(&memory, weights, bias);
    println!(" before: {:?}", weights.data);
    let weights2: RamTensor = weights.matmul(bias).unwrap();
    println!("after: {:?}", weights2.data);

    // Relu
    let relued: RamTensor = cus_act!(weights2, |x| x * 2.0);
    println!("Relued: {:?}", relued.data);
}
