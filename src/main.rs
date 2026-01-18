use std::path::Path;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
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
        return Err("Invalid magic number for MNIST labels".into());
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

#[derive(Debug)]
enum MatrixError {
    DimensionMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    IndexOutOfBounds {
        index: usize,
        rows: usize,
        cols: usize,
    },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch { expected, actual } => {
                write!(f, "Expected {:?}, got {:?}.", expected, actual)
            }
            MatrixError::IndexOutOfBounds { index, rows, cols } => {
                write!(
                    f,
                    "Index out of Bounds! Tried to access index {:?} of {:}x{:} matrix.",
                    index, rows, cols
                )
            }
        }
    }
}

impl std::error::Error for MatrixError {}

struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, val: f32) -> Matrix {
        let data = vec![val; rows * cols];
        Matrix { rows, cols, data }
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f32, MatrixError> {
        let index = row * self.cols + col;
        let val: f32 = *self.data.get(index).ok_or(MatrixError::IndexOutOfBounds {
            index,
            rows: self.rows,
            cols: self.cols,
        })?;
        Ok(val)
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        let index = row * self.cols + col;
        self.data[index] = val;
    }

    pub fn add(&mut self, b: &Matrix) -> Result<(), MatrixError> {
        if self.rows != b.rows || self.cols != b.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                actual: (b.rows, b.cols),
            });
        }
        self.data
            .iter_mut()
            .zip(&b.data)
            .for_each(|(a_val, b_val)| *a_val += *b_val);
        Ok(())
    }

    pub fn add_vector_to_rows(&mut self, b: &Vec<f32>) -> Result<(), MatrixError> {
        if self.cols != b.len() {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.cols, 1),
                actual: (b.len(), 1),
            });
        }
        self.data.chunks_mut(self.cols).for_each(|chunk| {
            chunk
                .iter_mut()
                .zip(b)
                .for_each(|(a_val, b_val)| *a_val += *b_val);
        });
        Ok(())
    }

    pub fn sub(&mut self, b: &Matrix) -> Result<(), MatrixError> {
        if self.rows != b.rows || self.cols != b.cols {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.rows, self.cols),
                actual: (b.rows, b.cols),
            });
        }
        self.data
            .iter_mut()
            .zip(&b.data)
            .for_each(|(a_val, b_val)| *a_val -= *b_val);
        Ok(())
    }

    pub fn mul(&self, b: &Matrix) -> Result<Matrix, MatrixError> {
        if self.cols != b.rows {
            return Err(MatrixError::DimensionMismatch {
                expected: (self.cols, b.cols),
                actual: (b.rows, b.cols),
            });
        }
        let b_t = b.transpose()?;
        let mut c = Matrix::new(self.rows, b.cols, 0.0);
        for i in 0..self.rows {
            for j in 0..b.cols {
                let mut s: f32 = 0.0;
                for k in 0..self.cols {
                    s += self.data[i * self.cols + k] * b_t.data[j * b.cols + k];
                }
                c.set(i, j, s);
            }
        }
        Ok(c)
    }

    pub fn transpose(&self) -> Result<Matrix, MatrixError> {
        let mut result = Matrix::new(self.cols, self.rows, 0.0);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = i * self.cols + j;
                result.set(j, i, self.data[index]);
            }
        }
        Ok(result)
    }

    pub fn randomize(&mut self, seed: &mut u32) -> Result<(), MatrixError> {
        self.data.iter_mut().for_each(|v| {
            *seed = (*seed as u64 * 1103515245 + 12345) as u32 % 2147483648;
            *v = (*seed as f32) / 2147483648.0 * 2.0 - 1.0;
        });
        Ok(())
    }

    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f32) -> f32,
    {
        self.data.iter_mut().for_each(|x| {
            *x = f(*x);
        });
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix ({} x {})\n{:?}\n",
            self.rows, self.cols, self.data
        )
    }
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
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img_gz = Path::new("/tmp/train-images-idx3-ubyte.gz");
    let img_raw = Path::new("/tmp/train-images-idx3-ubyte");
    let img_url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
    download_file(img_gz, img_url);
    unzip_file(img_gz);
    let matrix = load_mnist_images(img_raw)?;

    let lbl_gz = Path::new("/tmp/train-labels-idx1-ubyte.gz");
    let lbl_raw = Path::new("/tmp/train-labels-idx1-ubyte");
    let lbl_url = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
    download_file(lbl_gz, lbl_url);
    unzip_file(lbl_gz);
    let labels = load_mnist_labels(lbl_raw)?;

    println!("Images: {}x{}", matrix.rows, matrix.cols);
    println!("First 10 labels: {:?}", &labels[..10]);
    Ok(())
}
