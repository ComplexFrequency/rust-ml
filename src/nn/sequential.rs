use crate::nn::Module;
use crate::{Matrix, MatrixError};
use std::cell::RefCell;

pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
    forward_cache: RefCell<Vec<Matrix>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential {
            layers,
            forward_cache: RefCell::new(Vec::new()),
        }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        let mut cache = self.forward_cache.borrow_mut();
        cache.clear();
        cache.push(input.clone());

        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
            cache.push(current.clone());
        }

        Ok(current)
    }

    fn backward(&mut self, _input: &Matrix, grad_output: &Matrix) -> Result<Matrix, MatrixError> {
        let mut current_grad = grad_output.clone();
        let cache = self.forward_cache.borrow();

        for (index, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = &cache[index];
            current_grad = layer.backward(&layer_input, &current_grad)?;
        }

        Ok(current_grad)
    }

    fn get_params_and_gradients(&mut self) -> Vec<(&mut Matrix, &mut Matrix)> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.get_params_and_gradients())
            .collect()
    }
}
