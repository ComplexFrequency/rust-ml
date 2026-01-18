use rust_ml::Matrix;
use rust_ml::nn::Module;
use rust_ml::nn::activation::ReLU;
use rust_ml::nn::linear::Linear;
use rust_ml::nn::loss::CrossEntropy;
use rust_ml::nn::sequential::Sequential;
use rust_ml::optim::{Optimizer, SGD};

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

    let images = load_mnist_images(img_raw)?;
    let labels = load_mnist_labels(lbl_raw)?;
    let mut seed = 12345;

    const IMAGE_SIZE: usize = 784;
    const BATCH_SIZE: usize = 64;

    let mut model = Sequential::new(vec![
        Box::new(Linear::new(IMAGE_SIZE, 128, &mut seed)?),
        Box::new(ReLU),
        Box::new(Linear::new(128, 10, &mut seed)?),
    ]);

    let optimizer = SGD::new(0.005);

    println!("Dataset Loaded. Images: {}x{}", images.rows, images.cols);
    println!("Starting Training with Batch Size: {}...", BATCH_SIZE);

    let num_samples = images.rows;
    let mut running_loss = 0.0;
    let mut correct_predictions = 0;
    let mut samples_processed = 0;
    let report_interval = 1000;

    for i in (0..num_samples).step_by(BATCH_SIZE) {
        let current_batch_size = if i + BATCH_SIZE > num_samples {
            num_samples - i
        } else {
            BATCH_SIZE
        };

        let row_start = i * IMAGE_SIZE;
        let row_end = row_start + (current_batch_size * IMAGE_SIZE);
        let mut input_matrix = Matrix::new(current_batch_size, IMAGE_SIZE, 0.0);
        input_matrix
            .data
            .copy_from_slice(&images.data[row_start..row_end]);

        let mut target_matrix = Matrix::new(current_batch_size, 10, 0.0);
        for b in 0..current_batch_size {
            target_matrix.set(b, labels[i + b] as usize, 1.0);
        }

        let prediction = model.forward(&input_matrix)?;

        for b in 0..current_batch_size {
            let start = b * 10;
            let end = start + 10;
            let row_pred = &prediction.data[start..end];

            let predicted_digit = row_pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted_digit == labels[i + b] as usize {
                correct_predictions += 1;
            }
        }

        let loss_val = CrossEntropy::loss(&prediction, &target_matrix);
        running_loss += loss_val * current_batch_size as f32; // Scale by batch

        let gradient = CrossEntropy::grad(&prediction, &target_matrix)?;
        model.backward(&input_matrix, &gradient)?;

        optimizer.step(&mut model);

        samples_processed += current_batch_size;

        if samples_processed >= report_interval {
            let accuracy = (correct_predictions as f32 / samples_processed as f32) * 100.0;
            println!(
                "Sample: {:5} | Accuracy: {:>5.2}% | Avg Loss: {:.4}",
                i,
                accuracy,
                running_loss / samples_processed as f32
            );

            running_loss = 0.0;
            correct_predictions = 0;
            samples_processed = 0;
        }
    }

    println!("Training complete!");
    Ok(())
}
