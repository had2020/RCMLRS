// RCMLRS neuralnet-impls
// By: Hadrian Lazic
// Under MIT

#[derive(Clone, Debug)]
pub struct Layer {
    pub activation: String,
    pub tensor: RamTensor,
    pub bias: Vec<f32>,
    pub neural_units: usize,
}

/// Main class used for easy NeuralNetworks
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>, // Layer 0, is always input layer
    pub rand_min_max: (f32, f32),
    // training metrics
    /*
    pub target_f32: f32,
    pub matrix_target_f32: RamTensor,
    //pub max_epochs: f32,
    pub stopping_threshold: f32,
    pub learning_rate: f32,
    */
}

pub fn find_good_breakup(neural_units: usize) -> usize {
    if is_even_usize(neural_units) {
        neural_units / 2
    } else {
        neural_units
    }
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
            bias: vec![],
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
        }
    }

    /// This function basicly records and sets up the network structure, it will not run any ML calulcations yet
    /// Each dense will create a flattend layer with a size of the neural_units units and split it down into batchs.
    pub fn dense(&mut self, neural_units: usize, activation: &str) {
        let last_layer_index = self.layers.len() - 1;
        let last_tensor = &self.layers[last_layer_index].tensor;
        //let input_units = last_tensor.shape.x * last_tensor.shape.y * last_tensor.layer_length;

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
            tensor: layer_tensor,
            bias: vec![0.0; neural_units],
            neural_units,
        });
    }

    /// x_train, sets input shape.
    /// y_train, sets output shape.
    pub fn train(&self, x_train: Shape, y_train: Shape) {
        for layer in &self.layers {
            let activation = layer.activation.as_str();
            match activation {
                "ReLU" => {
                    println!("ReLU");
                }
                "Leaky ReLU" => (),
                "Sigmoid" => (),
                "Tanh" => (),
                "Softmax" => (),
                "Swish" => (),
                "GELU" => (),
                _ => (),
            }
        }
    }
}

// Example
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
