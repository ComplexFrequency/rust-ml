use crate::core::error::MatrixError;
use crate::core::matrix::Matrix;
use crate::nn::Module;

pub struct ReLU;
pub struct LeakyReLU {
    alpha: f32,
}
pub struct Softmax;
pub struct Identity;

impl Module for ReLU {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        let mut res = Matrix {
            rows: input.rows,
            cols: input.cols,
            data: input.data.clone(),
        };
        res.apply(|x| if x > 0.0 { x } else { 0.0 });
        Ok(res)
    }

    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError> {
        let data = input
            .data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(x, g)| if *x > 0.0 { *g } else { 0.0 })
            .collect();
        Ok(Matrix {
            rows: input.rows,
            cols: input.cols,
            data,
        })
    }
}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        let mut res = Matrix {
            rows: input.rows,
            cols: input.cols,
            data: input.data.clone(),
        };
        res.apply(|x| if x > 0.0 { x } else { self.alpha * x });
        Ok(res)
    }

    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError> {
        let data = input
            .data
            .iter()
            .zip(grad_output.data.iter())
            .map(|(x, g)| if *x > 0.0 { *g } else { self.alpha * g })
            .collect();
        Ok(Matrix {
            rows: input.rows,
            cols: input.cols,
            data,
        })
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        let mut result = Matrix::new(input.rows, input.cols, 0.0);

        for r in 0..input.rows {
            let start = r * input.cols;
            let end = start + input.cols;
            let row = &input.data[start..end];

            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = row.iter().map(|x| f32::exp(x - max_val)).sum();

            for (c, val) in row.iter().enumerate() {
                result.set(r, c, f32::exp(val - max_val) / sum);
            }
        }

        Ok(result)
    }

    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError> {
        let output = self.forward(input)?;
        let mut grad_input = Matrix::new(input.rows, input.cols, 0.0);

        for r in 0..input.rows {
            for i in 0..input.cols {
                let mut sum = 0.0;
                let yi = output.get(r, i)?;
                for j in 0..input.cols {
                    let yj = output.get(r, j)?;
                    let jacobian = if i == j { yi * (1.0 - yi) } else { -yi * yj };
                    sum += jacobian * grad_output.get(r, j)?;
                }
                grad_input.set(r, i, sum);
            }
        }
        Ok(grad_input)
    }
}

impl Module for Identity {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        Ok(input.clone())
    }

    fn backward(&mut self, _input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError> {
        Ok(grad_output.clone())
    }
}
