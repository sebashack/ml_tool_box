extern crate nalgebra;

use nalgebra::{DMatrix, DVector};

pub fn compute_cost(x: &DMatrix<f64>, y: &DVector<f64>, theta: &DVector<f64>) -> f64 {
    let m = y.len() as f64;
    let a = (x * theta) - y;
    let atr = a.transpose();
    let atr_by_a = atr * a;
    let r = atr_by_a.index((0, 0));

    *r / (2.0 * m)
}
