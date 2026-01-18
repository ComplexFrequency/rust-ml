use crate::nn::Module;

pub trait Optimizer {
    fn step(&self, model: &mut dyn Module);
}

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }

    pub fn step(&self, model: &mut dyn Module) {
        let params_and_gradients = model.get_params_and_gradients();
        for (p, g) in params_and_gradients.into_iter() {
            for i in 0..p.data.len() {
                p.data[i] -= self.learning_rate * g.data[i];
            }
        }
    }
}
