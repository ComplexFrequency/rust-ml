use crate::nn::Module;
use crate::{Matrix, MatrixError};

pub struct Linear {
    weights: Matrix,
    biases: Matrix,
    grad_weights: Matrix,
    grad_biases: Matrix,
}

impl Linear {
    pub fn new(
        input_size: usize,
        output_size: usize,
        seed: &mut u32,
    ) -> Result<Linear, MatrixError> {
        let mut weights = Matrix::new(input_size, output_size, 0.0);
        let grad_weights = Matrix::new(input_size, output_size, 0.0);
        let biases = Matrix::new(1, output_size, 0.0);
        let grad_biases = Matrix::new(1, output_size, 0.0);

        weights.randomize(seed)?;

        Ok(Linear {
            weights,
            biases,
            grad_weights,
            grad_biases,
        })
    }
}

impl Module for Linear {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        let mut res = input.mul(&self.weights)?;
        res.add_vector_to_rows(&self.biases.data)?;
        Ok(res)
    }

    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError> {
        let weights_t = self.weights.transpose()?;
        let grad_input = grad_output.mul(&weights_t)?;

        let input_t = input.transpose()?;
        self.grad_weights = self.grad_weights.add(&input_t.mul(grad_output)?)?;
        self.grad_biases = grad_output.clone();

        Ok(grad_input)
    }

    fn get_params_and_gradients(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        vec![
            (&mut self.weights, &mut self.grad_weights),
            (&mut self.biases, &mut self.grad_biases),
        ]
    }

    fn zero_grad(&mut self) {
        self.grad_weights.apply(|_| 0.0);
        self.grad_biases.apply(|_| 0.0);
    }
}
