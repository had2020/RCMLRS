use image::{DynamicImage, ImageFormat, ImageReader, Pixel, Rgb};
use rcmlrs::*;

fn scan_image(image_name: &str) -> RamTensor {
    let path = format!("../Datasets/blackwhite/{}.png", image_name);
    let img = ImageReader::open(path)
        .expect("No File at path!")
        .decode()
        .expect("Failed to decode image")
        .into_rgb8();

    let mut collected_pixels: Vec<Vec<f32>> = vec![];

    for row in 0..50 {
        collected_pixels.push(vec![]);

        for col in 0..50 {
            let pixel = img.get_pixel(row, col); //Rgb<u8>
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            collected_pixels[row as usize].push(red as f32);
            collected_pixels[row as usize].push(green as f32);
            collected_pixels[row as usize].push(blue as f32);
        }
    }

    RamTensor {
        shape: Shape { x: 50, y: 150 },
        layer_length: 1,
        data: vec![collected_pixels],
    }
}

fn main() {
    let input: RamTensor = scan_image("white");

    let mut weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
    let mut hidden_layer: RamTensor = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
    //let mut bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
    let mut bias: f32 = 0.0;

    let target: f32 = 0.0;
    let max_epochs = 1000;
    let stopping_threshold: f32 = 1e-10;
    let learning_rate: f32 = 0.01;

    for epoch in 0..max_epochs {
        let mut output: RamTensor = weights.clone();

        // fowardfeed
        output = weights.matmul(input.clone()).unwrap();
        output = hidden_layer.matmul(output.clone()).unwrap();
        output = output.sigmoid();

        // compute error
        let error = target - output.mean();

        // gradent decent
        // let gradient = output * (1.0 - output) * error; // For Sigmoid
        //let gradient = output.scaler((1.0 - output) * error);
        //let gradient = output.matmul((1.0 - output.clone())).unwrap() * error;
        let gradient = output.clone() * (1.0 - output.clone()) * error;

        // backpropgation
        //weights = weights + (gradient * learning_rate);
        //println!("{:?}", gradient.clone() * learning_rate);
        //weights = weights.add(gradient * learning_rate).unwrap();
        //weights = weights + (gradient * learning_rate);
        weights = weights + (gradient.clone() * input.clone() * learning_rate);
        hidden_layer = hidden_layer + (gradient.clone() * output.clone() * learning_rate);

        // Update bias
        bias -= learning_rate * error;
        // bias = output.mean();

        println!("Epoch {}: Bias: {}, Error: {}", epoch, bias, error);
    }
}
