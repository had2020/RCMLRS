use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 3);
    let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 3);

    println!("Before: {:?}", bias.data);
    let weights2: RamTensor = weights.multi_threaded_matmul(bias).unwrap();
    println!("After: {:?}", weights2.data);
}
