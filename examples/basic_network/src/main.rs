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

    let mut bias: f32 = 0.0;

    let target: f32 = 0.0;
    let max_epochs = 1000;
    let stopping_threshold: f32 = 1e-10;
    let learning_rate: f32 = 0.01;

    for epoch in 0..max_epochs {
        // fowardfeed
        let z1 = weights.matmul(input.clone()).unwrap() + bias;
        let a1 = z1.sigmoid();
        let z2 = hidden_layer.matmul(a1.clone()).unwrap();
        let a2 = z2.sigmoid();

        // compute error/ loss
        let error = target - a2.mean();

        // gradent decent
        let d_output = error * a2.clone() * (1.0 - a2); // sigmoid based

        // backprogation
        //weights = weights + (d_output.clone() * input.clone() * learning_rate);
        //hidden_layer = hidden_layer + (d_output.clone() * a2.clone() * learning_rate);
        let d_hidden_layer = d_output.matmul(a1.clone()).unwrap();
        let d_weights = d_output.matmul(input.clone()).unwrap();

        // update weights
        weights += d_weights * learning_rate;
        hidden_layer += d_hidden_layer * learning_rate;

        // bias
        bias -= learning_rate * error;

        println!("Epoch {}: Bias: {}, Error: {}", epoch, bias, error);

        if error.abs() < stopping_threshold {
            break;
        }
    }
}
