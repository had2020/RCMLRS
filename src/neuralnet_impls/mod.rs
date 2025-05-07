// RCMLRS neuralnet-impls
// By: Hadrian Lazic
// Under MIT

//use std::intrinsics::logf32; TODO debug to find use

use std::string;

use crate::{
    derivatives_optimizers_loss::{mae_loss, mse_loss},
    *,
};

#[derive(Clone, Debug)]
pub struct Layer {
    pub activation: String,
    pub tensor: RamTensor,
    pub bias: RamTensor,
    pub neural_units: usize,
}

/// Main class used for easy NeuralNetworks
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>, // Layer 0, is always input layer
    pub rand_min_max: (f32, f32),
    pub optimizer: String,
    pub loss: String,
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
        };

        NeuralNetwork {
            layers: vec![input_layer_init],
            rand_min_max: random_range_min_max,
            optimizer: "None".to_string(),
            loss: "None".to_string(),
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
            //bias: RamTensor::new_layer_zeros(new_shape, layer_tensor.layer_length),
            bias: RamTensor::new_layer_zeros(Shape { x: 1, y: 1 }, 1),
            neural_units,
        });
    }

    pub fn normalize_layer(&mut self, layer_index: usize) {
        self.layers[layer_index].tensor.normalize();
        self.layers[layer_index].bias.normalize();
    }

    pub fn normalize_input(&mut self) {
        self.normalize_layer(0);
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
        let mut next_id: usize = 0; // this is the next matrix, when layer starts from zero

        let mut fowardfeed_copy: Vec<RamTensor> = vec![];
        fowardfeed_copy.push(self.layers[0].tensor.clone()); // add input layer to stack

        for epoch in 0..max_epochs {
            for layer in 0..self.layers.len() {
                next_id += 1;

                // to match size etheir zero pad or Linear projection
                // matmul first than activation

                let mut tensor_layer: RamTensor = RamTensor {
                    shape: Shape {
                        x: 1,
                        y: self.layers[layer].neural_units.clone(),
                    },
                    layer_length: 1,
                    data: vec![vec![vec![0.0]]],
                };

                // fowardfeed
                if last_id - 1 != layer && layer != 0 {
                    // not last layer
                    tensor_layer.shape = self.layers[layer].tensor.shape.clone();
                    tensor_layer = self.layers[next_id]
                        .tensor
                        .pad_matmul_to_another(self.layers[layer].tensor.clone());
                    /*
                    .matmul(self.layers[next_id].tensor.clone().pad(
                        self.layers[layer].tensor.shape,
                        self.layers[layer].tensor.layer_length,
                        0.0,
                    ))
                    .unwrap();
                    */
                    // add bias
                    /*
                    println!(
                        "t1{:?} b2{:?} Layer: {:?}",
                        tensor_layer.shape, self.layers[layer].bias.shape, layer
                    );
                    */
                    tensor_layer = tensor_layer.clone() + self.layers[layer].bias.clone()
                }

                let activation = self.layers[layer].activation.as_str();
                match activation {
                    "ReLU" => {
                        tensor_layer = tensor_layer.relu();
                    }
                    //"Leaky ReLU" => (),  # requires negiative slope
                    "Sigmoid" => {
                        tensor_layer = tensor_layer.sigmoid();
                    }
                    "Tanh" => {
                        tensor_layer = tensor_layer.tanh();
                    }
                    "Softmax" => {
                        tensor_layer = tensor_layer.softmax();
                    }
                    "Swish" => {
                        tensor_layer = tensor_layer.swish();
                    }
                    "GELU" => {
                        tensor_layer = tensor_layer.gelu();
                    }
                    _ => {
                        if layer != 0 {
                            tensor_layer = tensor_layer.clone();
                        }
                    }
                }

                fowardfeed_copy.push(tensor_layer);

                // inside layer loop
            }
            // epochs loop
            next_id = 0; // reset loop state

            // TODO handling if it is a full tensor and not a scaler

            let output_mean = self.layers[last_id - 1].tensor.mean();

            let error: RamTensor = target.clone() - self.layers[last_id - 1].tensor;

            // incorrect applied theory
            // gradent decent
            /*
            let d_output: f32 = error
                * self.layers[last_id - 1].tensor.clone().scaler_to_f32()
                * (1.0 - self.layers[last_id - 1].tensor.clone().scaler_to_f32()); // For sigmoid
            */
            let d_output: f32 = error
                * self.layers[last_id - 1].tensor.scaler_to_f32()
                * (1.0 - self.layers[last_id - 1].tensor.scaler_to_f32());

            let loss: f32 = match self.loss.clone().as_str() {
                //TODO "SCCE" => -logf32(output_mean), // Sparse Categorical Crossentropy
                "MAE" => mae_loss(target, self.layers[last_id].tensor), // Mean Absolute Error
                "MSE" => mse_loss(target, self.layers[last_id].tensor), // Mean Squared Error
                _ => mse_loss(target, self.layers[last_id].tensor),     //fallback to MSE
            };

            // Incorrect applied theory
            // backprogation //TODO correct for vanishing gradients and adams
            //for layer in 0..self.layers.len() {
            for layer in (1..last_id).rev() {
                // check to see if their really is something to backprogate
                if layer != 0 && layer != last_id {
                    // gradent decent for each layer

                    let d_layer = d_output * self.layers[layer - 1].tensor.clone();

                    // weight updates
                    self.layers[layer].tensor = (d_layer.scaler_to_f32()
                        + self.layers[layer].tensor.clone())
                        * learning_rate;

                    // bias updates
                    self.layers[layer].bias = (self.layers[layer].bias.clone()
                        + d_layer.clone().scaler_to_f32())
                        * learning_rate;
                    //self.layers[layer].bias.clone() - learning_rate * error;

                    //layer_2_delta = layer_2_error * sigmoid_deriv(layer_2)
                    let layer_2_error = target - layer_2_output;
                    let layer_2_delta = layer_2_error * layer_2_output.sigmoid_deriv();
                }
            }

            // Backpropagation
            for layer in (1..last_id).rev() {
                if layer != 0 && layer != last_id {
                    // loss derivative
                    for matrix in 0..self.layers[layer].tensor.layer_length {
                        for row in 0..self.layers[layer].tensor.shape.x {
                            for col in 0..self.layers[layer].tensor.shape.y {
                                let error_gradient = match "{self.loss}" {
                                    "MSE" => mse_loss(target, self.layers[last_id - 1].tensor),
                                    "MAE" => mae_loss(target, self.layers[last_id - 1].tensor),
                                    _ => mse_loss(target, self.layers[last_id - 1].tensor),
                                };

                                let activation_gradient = match "{self.}" {

                                }

                            }
                        }
                    }
                }
            }

            if error.mean().abs() < stopping_threshold {
                println!("Training safely ending early!");
                break;
            }

            if epoch % 10 == 0 {
                println!(
                    "🔁Epoch {:?}, 🛸Loss: {:?} ❎Error: {:?}, 📤Output: {:?}, 🎯Target: {:?}, 📝bias: {:?}, first layer: {:?}",
                    epoch,
                    loss,
                    error,
                    output_mean,
                    target.scaler_to_f32(),
                    self.layers[last_id - 1].bias.mean(),
                    println!("Lay: {:?}", self.layers[1].tensor.data) // TODO remove in final
                );
                //println!("{:?}", self.layers[1]); // layer debug
            }
        }
        println!("Max epochs reached!")
    }

    //TODO metrics
    /// current optimizers "adam" or "" and loss or ""
    pub fn compile(&mut self, optimizer: &str, loss: &str) {
        self.optimizer = optimizer.to_string();
        self.loss = loss.to_string();
    }

    /// Applys normalize to the wights and bias of the previous layer
    pub fn normalize(&mut self) {
        let last_layer = self.layers.len().clone() - 1;

        self.layers[last_layer].tensor = self.layers[last_layer].tensor.normalize();

        self.layers[last_layer].bias = self.layers[last_layer].bias.normalize();
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
