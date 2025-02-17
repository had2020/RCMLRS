use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);
    let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);

    println!("Before: {:?}", weights.data);
    let weights2: RamTensor = weights.matmul(bias).unwrap();
    println!("After: {:?}", weights2.data);

    // activation functions
    let cus_act_res: RamTensor = cus_act!(weights2, |x| x + 2.0); // custom example, shift by 2
    println!("Cus_act: {:?}", cus_act_res.data);

    // Relu
    let relu_result = cus_act_res.relu();
    println!("Relued: {:?}", relu_result.data);

    // Sigmoid
    let sigmoid_result = Sigmoid.relu();
    println!("Sigmoided: {:?}", sigmoid_result.data);
}
