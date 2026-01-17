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
    fn new(rows: usize, cols: usize) -> Matrix {
        let data = vec![0.0; rows * cols];
        Matrix { rows, cols, data }
    }

    fn get(&self, row: usize, col: usize) -> Result<f32, MatrixError> {
        let index = row * self.cols + col;
        let val: f32 = *self
            .data
            .get(index)
            .ok_or(MatrixError::IndexOutOfBounds {
                index,
                rows: self.rows,
                cols: self.cols,
            })?;
        Ok(val)
    }

    fn set(&mut self, row: usize, col: usize, val: f32) {
        let index = row * self.cols + col;
        self.data[index] = val;
    }

    fn add(&mut self, b: &Matrix) -> Result<(), MatrixError> {
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

    fn sub(&mut self, b: &Matrix) -> Result<(), MatrixError> {
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
    let c = Matrix::new(2, 2);
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

    Ok(())
}
