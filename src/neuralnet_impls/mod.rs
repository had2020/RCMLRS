// RCMLRS neuralnet-impls
// By: Hadrian Lazic
// Under MIT

use crate::*;

#[derive(Clone, Debug)]
pub struct Layer {
    pub activation: String,
    pub tensor: RamTensor,
    pub bias: RamTensor,
    pub neural_units: usize,
    pub target_bias: RamTensor, // aka bias2 in earlyer protypes TODO
}

/// Main class used for easy NeuralNetworks
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>, // Layer 0, is always input layer
    pub rand_min_max: (f32, f32),
}

/// each dense will create a new layer on layers of NeuralNetwork
impl NeuralNetwork {
    pub fn new(
        input_shape: Shape,
        input_layer_length: usize,
        random_range_min_max: (f32, f32),
    ) -> Self {
        let input_layer_init = Layer {
            activation: "None".to_string(),
            bias: RamTensor::new_layer_zeros(input_shape.clone(), input_layer_length),
            tensor: RamTensor {
                shape: input_shape.clone(),
                layer_length: input_layer_length,
                data: vec![],
            },
            neural_units: input_shape.x * input_shape.y,
            target_bias: RamTensor::new_layer_zeros(input_shape.clone(), input_layer_length),
        };

        NeuralNetwork {
            layers: vec![input_layer_init],
            rand_min_max: random_range_min_max,
        }
    }

    /// This function basicly records and sets up the network structure, it will not run any ML calulcations yet
    /// Each dense will create a flattend layer with a size of the neural_units units and split it down into batchs.
    pub fn dense(&mut self, neural_units: usize, activation: &str) {
        //let last_layer_index = self.layers.len() - 1; //TODO in training
        //let input_units = last_tensor.shape.x * last_tensor.shape.y * last_tensor.layer_length;
        //let last_tensor = &self.layers[last_layer_index].tensor; //TODO in training

        let new_shape = Shape {
            x: 1,
            y: neural_units,
        };

        let layer_tensor = RamTensor::new_random(
            new_shape,
            1, // layer length
            self.rand_min_max.0,
            self.rand_min_max.1,
        );

        self.layers.push(Layer {
            activation: activation.to_string(),
            tensor: layer_tensor.clone(),
            bias: RamTensor::new_layer_zeros(layer_tensor.shape, layer_tensor.layer_length),
            neural_units,
            target_bias: RamTensor::new_layer_zeros(
                layer_tensor.shape.clone(),
                layer_tensor.layer_length,
            ),
        });
    }

    /// Use this to train your model, use a for loop for a larger dataset.
    pub fn input(&mut self, input: RamTensor, neural_units: usize) {
        let new_shape = Shape {
            x: 1,
            y: neural_units,
        };

        let layer_tensor = input.flatten();

        self.layers[0] = (Layer {
            activation: "None".to_string(),
            tensor: layer_tensor.clone(),
            bias: RamTensor::new_layer_zeros(layer_tensor.shape, layer_tensor.layer_length),
            neural_units,
        });
    }

    /// x_train, sets input shape.
    /// y_train, sets output shape.
    pub fn train(
        &mut self,
        //x_train: Shape, /*Shape { x: 50, y: 150 }, Shape { x: 1, y: 1 },*/
        //y_train: Shape,
        max_epochs: usize,
        target: RamTensor,
        learning_rate: f32,
        stopping_threshold: f32,
    ) {
        let last_id = self.layers.len();
        let mut layer_id: usize = 0; // this is the next matrix, when layer starts from zero

        for epoch in 0..max_epochs {
            for layer in 0..self.layers.len() {
                layer_id += 1;

                // to match size etheir zero pad or Linear projection
                // matmul first than activation

                let mut tensor_layer: RamTensor = RamTensor {
                    shape: Shape { x: 1, y: 1 },
                    layer_length: 1,
                    data: vec![vec![vec![0.0]]],
                };

                if last_id - 1 != layer {
                    if self.layers[layer].tensor.shape > self.layers[layer_id].tensor.shape {
                        tensor_layer = self.layers[layer].tensor.flatten().pad(
                            self.layers[layer_id].tensor.shape.clone(),
                            self.layers[layer].tensor.layer_length,
                            0.0,
                        );
                        tensor_layer = tensor_layer
                            .matmul(self.layers[layer_id].tensor.clone())
                            .unwrap();
                    } else if self.layers[layer].tensor.shape < self.layers[layer_id].tensor.shape {
                        tensor_layer = self.layers[layer_id].tensor.flatten().pad(
                            self.layers[layer].tensor.shape.clone(),
                            self.layers[layer_id].tensor.layer_length,
                            0.0,
                        );
                        tensor_layer = tensor_layer
                            .matmul(self.layers[layer].tensor.clone())
                            .unwrap();
                    }
                }

                let activation = self.layers[layer].activation.as_str();
                match activation {
                    "ReLU" => {
                        self.layers[layer].tensor = tensor_layer.relu();
                    }
                    //"Leaky ReLU" => (),  # requires negiative slope
                    "Sigmoid" => {
                        self.layers[layer].tensor = tensor_layer.sigmoid();
                    }
                    "Tanh" => {
                        self.layers[layer].tensor = tensor_layer.tanh();
                    }
                    "Softmax" => {
                        self.layers[layer].tensor = tensor_layer.softmax();
                    }
                    "Swish" => {
                        self.layers[layer].tensor = tensor_layer.swish();
                    }
                    "GELU" => {
                        self.layers[layer].tensor = tensor_layer.gelu();
                    }
                    _ => {
                        if layer != 0 {
                            println!("No activiation on layer: {}", layer)
                        }
                    }
                }
                // layer loop
            }
            // epochs loop
            layer_id = 0; // reset loop state

            // TODO handling if it is a full tensor and not a scaler

            let output_mean = self.layers[last_id - 1].tensor.mean();

            let error: f32 = (target.clone() - RamTensor::from(output_mean)).into(); // computer error

            // gradent decent
            let d_output = error
                * self.layers[last_id - 1].tensor.clone()
                * (1.0 - self.layers[last_id - 1].tensor.clone()); // For sigmoid

            // backprogation
            for layer in 0..self.layers.len() {
                // check to see if their really is something to backprogate
                if layer != 0 && layer != last_id {
                    // gradent decent for each layer
                    let d_layer = d_output
                        .pad(
                            self.layers[layer - 1].tensor.shape,
                            self.layers[layer - 1].tensor.layer_length,
                            0.0,
                        )
                        .matmul(self.layers[layer - 1].tensor.clone())
                        .unwrap();

                    // weight updates
                    self.layers[layer].tensor = d_layer.pad(
                        self.layers[layer].tensor.shape.clone(),
                        self.layers[layer].tensor.layer_length.clone(),
                        0.0,
                    ) + self.layers[layer].tensor.clone()
                        * learning_rate;

                    self.layers[layer].bias = d_layer * learning_rate;
                }
            }

            // bias updates TODO

            if error.abs() < stopping_threshold {
                println!("Training safely ending early!");
                break;
            }
            if epoch % 10 == 0 {
                println!(
                    "Epoch {:?}, Error: {:?}, Output: {:?}",
                    epoch, error, output_mean,
                );
            }
        }
        println!("Max epochs reached!")
    }
}

// Desired
/*
# Define a simple feedforward neural network
class NeuralNet:
    def __init__(self, input_shape, output_units):
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_units, activation='softmax')
        ])

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
        metrics = metrics or ["accuracy"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

# Example usage
if __name__ == "__main__":
    net = NeuralNet(input_shape=(784,), output_units=10)
    net.compile().train(x_train, y_train, epochs=5)
*/
