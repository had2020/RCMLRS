use rcmlrs::*;

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 5);
    let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 3);

    println!("Before: {:?}", bias.data);
    let resized_bias = bias.resize_tensor(Shape { x: 3, y: 3 }, 5, 3.0);
    let weights2: RamTensor = weights.matmul(resized_bias).unwrap();
    println!("After: {:?}", weights2.data);

    // activation functions
    let cus_act_res: RamTensor = cus_act!(weights2, |x| x + 2.0); // custom example, shift by 2
    println!("Cus_act: {:?}", cus_act_res.data);

    // Relu
    let relu_result = cus_act_res.relu();
    println!("ReLU: {:?}", relu_result.data);

    // Sigmoid
    let sigmoid_result = relu_result.sigmoid();
    println!("Sigmoid: {:?}", sigmoid_result.data);

    // Tanh
    let tanh_result = sigmoid_result.tanh();
    println!("Tanh: {:?}", tanh_result.data);
}
