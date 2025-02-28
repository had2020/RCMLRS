use image::{DynamicImage, ImageFormat, ImageReader, Pixel, Rgb};
use rcmlrs::*;

fn scan_image(image_name: &str) -> RamTensor {
    let path = format!("../Datasets/blackwhite/{}.png", image_name);
    let img = ImageReader::open(path).unwrap().decode().unwrap();

    let mut collected_pixels: Vec = Vec![];

    for row in 0..50 {
        collected_pixels.push(Vec![]);

        for col in 0..50 {
            let pixel: Rgb<u8> = img.get_pixel(row, col).unwrap();
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            collected_pixels[row].push(Vec![red as f32, green as f32, blue as f32]);
        }
    }

    RamTensor {
        shape: Shape { x: 50, y: 150 },
        layer_length: 1,
        data: collected_pixels,
    }
}

fn main() {
    let weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2); // TODO fix panic when this tensor is bigger
    let bias: RamTensor = RamTensor::new_layer_zeros(Shape { x: 3, y: 3 }, 2);

    println!("Before: {:?}", weights.data);
    let weights2: RamTensor = weights.matmul(bias).unwrap();
    println!("After: {:?}", weights2.data);

    // Relu
    let relu_result = weights2.relu();
    println!("ReLU: {:?}", relu_result.data);
}
