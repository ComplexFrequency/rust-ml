use rust_ml::{Matrix, MatrixError};

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::process::Command;

fn read_u32(file: &mut File) -> io::Result<u32> {
    let mut buffer = [0u8; 4];
    file.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn download_file(path: &Path, url: &str) {
    if path.exists() {
        return;
    }
    let status = Command::new("curl")
        .arg("-o")
        .arg(path)
        .arg(url)
        .status()
        .expect("Failed to execute curl");

    if !status.success() {
        panic!("Curl failed to download the data!");
    }
}

fn unzip_file(path: &Path) {
    let mut target = path.to_path_buf();
    target.set_extension("");

    if target.exists() {
        return;
    }

    if !path.exists() {
        panic!("Attempted to unzip {:?}, but the file is missing!", path);
    }

    let status = Command::new("gzip")
        .arg("-d")
        .arg(path)
        .status()
        .expect("Failed to execute gzip");

    if !status.success() {
        panic!("Gzip failed!");
    }
}

fn load_mnist_images(path: &Path) -> Result<Matrix, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;

    let magic = read_u32(&mut file)?;
    let num_images = read_u32(&mut file)? as usize;
    let rows = read_u32(&mut file)? as usize;
    let cols = read_u32(&mut file)? as usize;

    if magic != 2051 {
        return Err("Invalid magic number for MNIST images".into());
    }

    let pixels_per_image = rows * cols;
    let mut matrix = Matrix::new(num_images, pixels_per_image, 0.0);

    let mut raw_pixels = Vec::new();
    file.read_to_end(&mut raw_pixels)?;

    matrix.data = raw_pixels.iter().map(|x| (*x as f32) / 255.0).collect();

    Ok(matrix)
}

fn load_mnist_labels(path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;

    let magic = read_u32(&mut file)?;
    let _num_items = read_u32(&mut file)?;

    if magic != 2049 {
        return Err("Invalid magic number".into());
    }

    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    Ok(data)
}

fn one_hot_encode(labels: &[u8]) -> Matrix {
    let mut matrix = Matrix::new(labels.len(), 10, 0.0);

    for (i, &label) in labels.iter().enumerate() {
        matrix.set(i, label as usize, 1.0);
    }

    matrix
}

struct Layer {
    weights: Matrix,
    biases: Vec<f32>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize, seed: &mut u32) -> Result<Layer, MatrixError> {
        let mut weights = Matrix::new(input_size, output_size, 0.0);
        let biases = vec![0.0; output_size];

        weights.randomize(seed)?;

        Ok(Layer { weights, biases })
    }

    pub fn forward(&self, input: &Matrix) -> Result<Matrix, MatrixError> {
        let mut res = input.mul(&self.weights)?;
        res.add_vector_to_rows(&self.biases)?;
        Ok(res)
    }

    pub fn update_weights(&mut self, input: &Matrix, grad_output: &Matrix, learning_rate: f32) {
        let input_t = input.transpose().unwrap();
        let delta_w = input_t.mul(grad_output).unwrap();

        for i in 0..self.weights.data.len() {
            self.weights.data[i] -= learning_rate * delta_w.data[i];
        }

        for i in 0..self.biases.len() {
            self.biases[i] -= learning_rate * grad_output.data[i];
        }
    }
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

pub fn softmax(matrix: &Matrix) -> Matrix {
    let mut result = Matrix::new(matrix.rows, matrix.cols, 0.0);

    for r in 0..matrix.rows {
        let start = r * matrix.cols;
        let end = start + matrix.cols;
        let row = &matrix.data[start..end];

        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = row.iter().map(|x| f32::exp(x - max_val)).sum();

        for c in 0..row.len() {
            result.set(r, c, f32::exp(row[c] - max_val) / sum);
        }
    }

    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img_gz = Path::new("/tmp/train-images-idx3-ubyte.gz");
    let img_raw = Path::new("/tmp/train-images-idx3-ubyte");
    let img_url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";

    let lbl_gz = Path::new("/tmp/train-labels-idx1-ubyte.gz");
    let lbl_raw = Path::new("/tmp/train-labels-idx1-ubyte");
    let lbl_url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";

    download_file(img_gz, img_url);
    unzip_file(img_gz);
    download_file(lbl_gz, lbl_url);
    unzip_file(lbl_gz);

    let matrix = load_mnist_images(img_raw)?;
    let labels = load_mnist_labels(lbl_raw)?;

    println!("Dataset Loaded. Images: {}x{}", matrix.rows, matrix.cols);
    println!("Starting Training...");

    let learning_rate = 0.01;
    let mut seed = 12345;
    let mut layer = Layer::new(784, 10, &mut seed)?;

    let mut correct_predictions = 0;
    let mut running_loss = 0.0;
    let window_size = 1000;

    for i in 0..matrix.rows {
        let row_start = i * 784;
        let mut input_row = Matrix::new(1, 784, 0.0);
        input_row.data.copy_from_slice(&matrix.data[row_start..row_start + 784]);

        let logits = layer.forward(&input_row)?;
        let probs = softmax(&logits);

        let label = labels[i] as usize;

        let mut predicted_digit = 0;
        let mut max_prob = -1.0;
        for (idx, &p) in probs.data.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                predicted_digit = idx;
            }
        }

        if predicted_digit == label {
            correct_predictions += 1;
        }

        running_loss -= (probs.data[label] + 1e-10).ln();

        let mut error = probs;
        let current_val = error.get(0, label)?;
        error.set(0, label, current_val - 1.0);

        layer.update_weights(&input_row, &error, learning_rate);

        if i % window_size == 0 && i > 0 {
            let accuracy = (correct_predictions as f32 / window_size as f32) * 100.0;
            let avg_loss = running_loss / window_size as f32;
            println!(
                "Sample: {:5} | Accuracy: {:>5.2}% | Avg Loss: {:.4}",
                i, accuracy, avg_loss
            );

            correct_predictions = 0;
            running_loss = 0.0;
        }
    }

    println!("Training complete!");
    Ok(())
}
