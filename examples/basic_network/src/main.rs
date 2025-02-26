use image::{DynamicImage, ImageFormat, ImageReader, Pixel, Rgb};
use rcmlrs::*;

fn scane(image_name: &str) -> Result<(), std::io::Error> {
    let path = format!("../Datasets/blackwhite/{}.png", image_name);
    let img = ImageReader::open(path)?.decode()?;

    let pixel: Rgb<u8> = img.get_pixel(10, 5).unwrap();

    let red = pixel[0];
    let green = pixel[1];
    let blue = pixel[2];

    println!("Pixel at (10, 5): R={}, G={}, B={}", red, green, blue);

    Ok(())
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
