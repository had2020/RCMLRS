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
    train("white", 0.0);
    train("black", 1.0);
    // took roughly 18.78 secs last tests
}

fn train(file_name: &str, output_target: f32) {
    println!("Training on ->, {}", file_name);

    let input: RamTensor = scan_image(file_name);

    let mut weights: RamTensor = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
    let mut hidden_layer: RamTensor = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
    let mut bias1: RamTensor = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
    let mut bias2: f32 = 0.0;

    let target: f32 = output_target; //
    let max_epochs = 1000;
    let stopping_threshold: f32 = 0.08; //1e-10 for most precise training
    let learning_rate: f32 = -0.01;

    for epoch in 0..max_epochs {
        // fowardfeed
        let z1 = weights.matmul(input.clone()).unwrap() + bias1.clone(); // bias1 tensor broadcasted
        let a1 = z1.sigmoid();
        let z2 = hidden_layer.matmul(a1.clone()).unwrap() + bias2;
        let a2 = z2.sigmoid();

        // output mean
        let output_mean = a2.mean();

        // compute error/ loss
        let error = target - output_mean;

        // gradent decent
        let d_output = error * a2.clone() * (1.0 - a2); // sigmoid based

        // backprogation
        let d_hidden_layer = d_output.matmul(a1.clone()).unwrap();
        let d_weights = d_output.matmul(input.clone()).unwrap();

        // update weights
        weights += d_weights * learning_rate;
        hidden_layer += d_hidden_layer.clone() * learning_rate;
        bias1 += d_hidden_layer.clone() * learning_rate;
        bias2 -= learning_rate * error;

        // bias
        bias2 -= learning_rate * error;

        if epoch % 10 == 0 {
            println!(
                "Epoch {}: Bias2: {}, Error: {}, Output: {}",
                epoch, bias2, error, output_mean,
            );
        }

        if error.abs() < stopping_threshold {
            break;
        }
    }
}
