use image::ImageReader;
use rand::Rng;
use rcmlrs::*;

fn scan_image(image_name: &str, variance: f32) -> RamTensor {
    let path = format!("../Datasets/blackwhite/{}.png", image_name);
    let img = ImageReader::open(path)
        .expect("No File at path!")
        .decode()
        .expect("Failed to decode image")
        .into_rgb8();

    let mut rng = rand::thread_rng();
    let mut collected_pixels: Vec<Vec<f32>> = vec![];

    for row in 0..50 {
        collected_pixels.push(vec![]);

        for col in 0..50 {
            let pixel = img.get_pixel(row, col);
            let red = pixel[0] as f32;
            let green = pixel[1] as f32;
            let blue = pixel[2] as f32;

            // small noise between -2.0 and +2.0
            let noise_r: f32 = rng.gen_range(-variance..variance);
            let noise_g: f32 = rng.gen_range(-variance..variance);
            let noise_b: f32 = rng.gen_range(-variance..variance);

            collected_pixels[row as usize].push(red + noise_r);
            collected_pixels[row as usize].push(green + noise_g);
            collected_pixels[row as usize].push(blue + noise_b);
        }
    }

    RamTensor {
        shape: Shape { x: 50, y: 150 },
        layer_length: 1,
        data: vec![collected_pixels],
    }
}

fn main() {
    // training
    let filename: &str = "grayscale_model";
    let reruns = 1;
    train("white", 0.95, false, filename, true); // init
    for run in 0..reruns {
        println!("rerun: {}", run);
        train("white", 0.05, true, filename, true);
        train("black", 0.95, true, filename, false);
    }

    // usage
    let response0 = run("white", filename);
    let response1 = run("black", filename);
    println!(
        "ran model with white:{}, and black:{}",
        response0, response1
    );
}

fn train(
    input_file_name: &str,
    output_target: f32,
    read_first: bool,
    filename: &str,
    sample_a: bool,
) {
    println!("Training on ->, {}", input_file_name);

    let input: RamTensor = scan_image(input_file_name, 10.0);

    let mut weights: RamTensor;
    let mut hidden_layer: RamTensor;
    let mut hidden_layer1: RamTensor;
    let mut bias1: RamTensor;
    let mut bias2: f32 = 0.0;
    let mut bias3: f32 = 0.0;

    if !read_first {
        weights = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
        hidden_layer = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
        hidden_layer1 = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
        bias1 = RamTensor::new_layer_zeros(Shape { x: 50, y: 150 }, 1);
        // Scaler per class
        bias2 = 0.0; // sample_a
        bias3 = 0.0;
    } else {
        let model = load_state_binary(filename);

        weights = model[0].clone();
        hidden_layer = model[1].clone();
        hidden_layer1 = model[2].clone();
        bias1 = model[3].clone();
        bias2 = model[4].data[0][0][0].clone();
        bias3 = model[5].data[0][0][0].clone();
    }

    let target: f32 = output_target; //
    let max_epochs = 1000;
    let stopping_threshold: f32 = 0.08; //1e-10 for most precise training
    let learning_rate: f32 = -0.05; // needs to be negative

    let start = std::time::Instant::now();

    let mut bias_frozen: bool = false;

    for epoch in 0..max_epochs {
        // fowardfeed
        let z1 = weights.matmul(input.clone()).unwrap() + bias1.clone();
        let a1 = z1.sigmoid();
        let z2 = hidden_layer.matmul(a1.clone()).unwrap() + bias2.clone();
        let a1b = z2.sigmoid();
        let z3 = hidden_layer1.matmul(a1b.clone()).unwrap() + bias3.clone();
        let a2 = z3.sigmoid();

        // output mean
        let output_mean = a2.mean();

        // compute error/ loss
        let mut error = target - output_mean;

        if error < 0.0 {
            error = error.abs();
        }

        // gradent decent
        let d_output = error * a2.clone() * (1.0 - a2); // sigmoid based

        // backprogation
        let d_hidden_layer = d_output.matmul(a1.clone()).unwrap();
        let d_hidden_layer1 = d_output.matmul(a1b.clone()).unwrap();
        let d_weights = d_output.matmul(input.clone()).unwrap();

        // update weights
        weights += d_weights * learning_rate;
        hidden_layer += d_hidden_layer.clone() * learning_rate;
        hidden_layer1 += d_hidden_layer1.clone() * learning_rate;
        bias1 += d_hidden_layer.clone() * learning_rate;

        // bias updating in accordance to class sample
        if !bias_frozen {
            if sample_a == true {
                bias2 -= learning_rate * error;
            } else {
                bias3 -= learning_rate * error;
            }
        }

        // preventing bias freeze
        if epoch % 5 == 0 {
            bias_frozen = !bias_frozen;
        }

        if epoch % 10 == 0 {
            println!(
                "Epoch {}: Bias2: {}, Bias3: {}, Error: {}, Output: {}, Elapsed: {:?}",
                epoch,
                bias2,
                bias3,
                error,
                output_mean,
                start.elapsed()
            );
        }

        if error.abs() < stopping_threshold {
            break;
        }
    }

    save_tensors_binary(
        filename,
        vec![
            weights,
            hidden_layer,
            hidden_layer1,
            bias1,
            RamTensor {
                shape: Shape { x: 1, y: 1 },
                layer_length: 1,
                data: vec![vec![vec![bias2]]],
            },
            RamTensor {
                shape: Shape { x: 1, y: 1 },
                layer_length: 1,
                data: vec![vec![vec![bias3]]],
            },
        ],
    );
}

fn run(input_file_name: &str, filename: &str) -> f32 {
    let input: RamTensor = scan_image(input_file_name, 10.0);

    let model = load_state_binary(filename);

    let weights = model[0].clone();
    let hidden_layer = model[1].clone();
    let hidden_layer1 = model[2].clone();
    let bias1 = model[3].clone();
    let bias2 = model[4].data[0][0][0].clone();
    let bias3 = model[5].data[0][0][0].clone();

    // fowardfeed
    let z1 = weights.matmul(input.clone()).unwrap() + bias1.clone();
    let a1 = z1.sigmoid();
    let z2 = hidden_layer.matmul(a1.clone()).unwrap() + bias2.clone();
    let a1b = z2.sigmoid();
    let z3 = hidden_layer1.matmul(a1b.clone()).unwrap() + bias3.clone();
    let a2 = z3.sigmoid();

    // output mean
    a2.mean()
}
