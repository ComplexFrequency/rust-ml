use crate::core::error::MatrixError;
use crate::core::matrix::Matrix;

pub struct MSE;
pub struct CrossEntropy;

impl MSE {
    pub fn loss(prediction: &Matrix, target: &Matrix) -> f32 {
        let n = (prediction.rows * prediction.cols) as f32;
        let s = prediction
            .data
            .iter()
            .zip(target.data.iter())
            .fold(0.0, |acc, (pred, targ)| acc + (pred - targ) * (pred - targ));
        s / n
    }

    pub fn grad(prediction: &Matrix, target: &Matrix) -> Result<Matrix, MatrixError> {
        let n = (prediction.rows * prediction.cols) as f32;
        let mut res = prediction.sub(target)?;
        res.apply(|x| 2.0 * x / n);
        Ok(res)
    }
}

impl CrossEntropy {
    pub fn loss(prediction: &Matrix, target: &Matrix) -> f32 {
        let mut loss: f32 = 0.0;
        for r in 0..prediction.rows {
            let start = r * prediction.cols;
            let end = start + prediction.cols;
            let row_pred = &prediction.data[start..end];
            let row_trgt = &target.data[start..end];

            let max_val = row_pred.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = row_pred.iter().map(|x| f32::exp(x - max_val)).sum();

            let target_index = row_trgt.iter().position(|&x| x == 1.0).unwrap_or(0);
            loss += f32::ln(sum) + max_val - row_pred[target_index]
        }

        loss / prediction.rows as f32
    }

    pub fn grad(prediction: &Matrix, target: &Matrix) -> Result<Matrix, MatrixError> {
        Self::softmax(prediction)?.sub(target)
    }

    fn softmax(input: &Matrix) -> Result<Matrix, MatrixError> {
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
}
