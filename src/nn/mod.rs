use crate::core::error::MatrixError;
use crate::core::matrix::Matrix;

pub mod activation;
pub mod linear;
pub mod loss;
pub mod sequential;

pub trait Module {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError>;
    fn backward(&mut self, input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError>;

    fn get_params_and_gradients(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        vec![]
    }
}
