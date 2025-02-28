use image::{DynamicImage, ImageFormat, ImageReader, Pixel, Rgb};
use rcmlrs::*;

fn scan_image(image_name: &str) -> RamTensor {
    let path = format!("../Datasets/blackwhite/{}.png", image_name);
    let img = ImageReader::open(path)
        .expect("No File at path!")
        .decode()
        .expect("Failed to decode image")
        .into_rgb8();

    let mut collected_pixels: Vec<Vec<Vec<f32>>> = vec![];
    collected_pixels.push(vec![]); // first matrix

    for row in 0..50 {
        collected_pixels[0].push(vec![]);

        for col in 0..50 {
            let pixel = img.get_pixel(row, col); //Rgb<u8>
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            collected_pixels[0][row as usize].push(vec![red as f32, green as f32, blue as f32]);
        }
    }

    RamTensor {
        shape: Shape { x: 50, y: 150 },
        layer_length: 1,
        data: collected_pixels,
    }
}

fn main() {
    let input: RamTensor = scan_image("white");

    let mut weights: RamTensor = RamTensor::new_random(Shape { x: 50, y: 150 }, 1, 0.0, 100.0);
    let mut bias: RamTensor = RamTensor::new_random(Shape { x: 50, y: 150 }, 1, 0.0, 100.0);

    let target: f32 = 1.0;
    let max_epochs = 1000;
    let stopping_threshold: f32 = 1e-10;
    let learning_rate: f32 = 0.01;

    for epoch in 0..max_epochs {
        let test = weights.matmul(input.clone()).unwrap();
    }

    /*
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2); // TODO fix panic when this tensor is bigger
    let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);

    println!("Before: {:?}", weights.data);
    let weights2: RamTensor = weights.matmul(bias).unwrap();
    println!("After: {:?}", weights2.data);

    // Relu
    let relu_result = weights2.relu();
    println!("ReLU: {:?}", relu_result.data);
    */
}
