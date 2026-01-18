use std::fmt;

#[derive(Debug)]
pub enum MatrixError {
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

