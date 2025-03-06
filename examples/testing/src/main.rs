use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_random(Shape { x: 50, y: 150 }, 1, 1.0, 10.0);
    let bias: RamTensor = RamTensor::new_random(Shape { x: 50, y: 150 }, 1, 1.0, 10.0);

    println!("Before: {:?}", bias.data);
    let weights2: RamTensor = weights.multi_threaded_matmul(bias).unwrap();
    println!("After: {:?}", weights2.data);
}
