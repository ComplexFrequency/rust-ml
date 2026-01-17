use std::fmt;

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
        self.data
            .chunks_mut(self.cols)
            .for_each(|chunk| {
                chunk.iter_mut().zip(b).for_each(|(a_val, b_val)| *a_val += *b_val);
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

    pub fn mul(&mut self, b: &Matrix) -> Result<Matrix, MatrixError> {
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

fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing Matrices.");
    let mut a = Matrix {
        rows: 2,
        cols: 2,
        data: vec![0.0, 1.0, 2.0, 3.0],
    };
    let b = Matrix {
        rows: 2,
        cols: 2,
        data: vec![4.0, 5.0, 6.0, 7.0],
    };
    let mut c = Matrix::new(2, 2, 0.0);
    let mut seed: u32 = 42;

    println!("{a}");
    println!("{b}");
    println!("{c}");

    println!("Adding Matrices. (C = A + B)");
    let _ = a.add(&b);
    println!("{a}");

    println!("Subtracting Matrices. (A = C - B)");
    let _ = a.sub(&b);
    println!("{a}");

    println!("Setting value at pos (0, 0).");
    a.set(0, 0, -1.0);
    println!("{a}");

    println!("Getting value at pos (0, 1)");
    let s = a.get(0, 1)?;
    println!("{s}");

    println!("Matrix Multiplication. (C = A x B)");
    c = a.mul(&b)?;
    println!("{c}");

    println!("Randomizing matrix.");
    let _ = c.randomize(&mut seed);
    println!("{c}");

    println!("Applying ReLU.");
    c.apply(relu);
    println!("{c}");

    println!("Applying sigmoid.");
    c.apply(sigmoid);
    println!("{c}");

    println!("Applying vector to rows.");
    let d = vec![5.0, -5.0];
    let _ = c.add_vector_to_rows(&d);
    println!("{c}");

    Ok(())
}
