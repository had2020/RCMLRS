use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_random(Shape { x: 3, y: 3 }, 2, 1.0, 10.0);
    let bias: RamTensor = RamTensor::new_random(Shape { x: 3, y: 3 }, 2, 1.0, 10.0);

    println!("Before: {:?}", weights.data);
    //let weights2: RamTensor = weights.multi_threaded_matmul(bias).unwrap();
    let weights2: RamTensor = weights * 1.0; //TODO  fix zeros
    println!("After: {:?}", weights2.data);
}
