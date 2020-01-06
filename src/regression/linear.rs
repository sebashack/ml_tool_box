extern crate nalgebra;

use nalgebra::{DMatrix, DVector, Matrix, Dynamic, U1, SliceStorage};

pub fn compute_cost(x: &DMatrix<f64>, y: &DVector<f64>, theta: &DVector<f64>) -> f64 {
    let m = y.len() as f64;
    let a = (x * theta) - y;
    let product = a.transpose() * a;
    let r = product.index((0, 0));

    *r / (2.0 * m)
}

pub fn gradient_descent(x: &DMatrix<f64>,
                        y: &DVector<f64>,
                        theta: &DVector<f64>,
                        alpha: f64,
                        num_iters: u32) -> DVector<f64> {
    let m = y.len();
    let theta_len = theta.len();
    let x_tr = x.transpose();

    let mut new_theta: DVector<f64> = DVector::from_element(theta_len, 0f64);
    new_theta.copy_from(theta);

    for k in 0..num_iters {
        let mut delta = DVector::from_element(theta_len, 0f64);

        for i in 0..m {
            let x_vals = x_tr.column(i);
            let v = ((new_theta.transpose() * x_vals).index(0) - *(y.index(i))) * x_vals;

            delta = delta + v;
        }

        delta = delta / (m as f64);

        new_theta = new_theta - (alpha * delta);
    }

    new_theta
}

pub fn featureNormalize(x: &DMatrix<f64>) -> DMatrix<f64> {
    let num_rows = x.nrows();
    let num_cols = x.ncols();

    let mut normalized_x = DMatrix::from_element(num_rows, num_cols, 0f64);

    for i in 0..num_cols {
        let vals = x.column(i);
        let mean = computeMxSliceMean(num_cols, &vals);
        let std = computeMxSliceStd(num_cols, mean, &vals);

        let new_col = DVector::from_fn(num_cols, |j, _| {
            (x.index((j, i)) - mean) / std
        });

        normalized_x.set_column(i, &new_col);
    }

    normalized_x
}

fn computeMxSliceMean(n: usize, x: &Matrix<f64, Dynamic, U1, SliceStorage<'_, f64, Dynamic, U1, U1, Dynamic>>) -> f64 {
    let mut sum: f64 = 0.0;

    for v in x.iter() {
        sum = sum + v;
    }

    sum / (n as f64)
}

fn computeMxSliceStd(n: usize, mean: f64, x: &Matrix<f64, Dynamic, U1, SliceStorage<'_, f64, Dynamic, U1, U1, Dynamic>>) -> f64 {
    let mut sum_of_squares: f64 = 0.0;

    for v in x.iter() {
        sum_of_squares = sum_of_squares + (v - mean).powf(2.0);
    }

    (sum_of_squares / ((n - 1) as f64)).sqrt()
}
