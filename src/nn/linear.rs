use crate::nn::Module;
use crate::{Matrix, MatrixError};

pub struct Linear {
    weights: Matrix,
    biases: Vec<f32>,
}

impl Linear {
    pub fn new(
        input_size: usize,
        output_size: usize,
        seed: &mut u32,
    ) -> Result<Linear, MatrixError> {
        let mut weights = Matrix::new(input_size, output_size, 0.0);
        let biases = vec![0.0; output_size];

        weights.randomize(seed)?;

        Ok(Linear { weights, biases })
    }
}

impl Module for Linear {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        let mut res = input.mul(&self.weights)?;
        res.add_vector_to_rows(&self.biases)?;
        Ok(res)
    }

    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError> {
        let weights_t = self.weights.transpose()?;
        let grad_input = grad_output.mul(&weights_t)?;

        let input_t = input.transpose()?;
        let grad_weights = input_t.mul(grad_output)?;

        let learning_rate = 0.01;
        for i in 0..self.weights.data.len() {
            self.weights.data[i] -= learning_rate * grad_weights.data[i];
        }
        for i in 0..self.biases.len() {
            self.biases[i] -= learning_rate * grad_output.data[i];
        }
        Ok(grad_input)
    }
}
