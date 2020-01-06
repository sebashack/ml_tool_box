extern crate nalgebra;

use nalgebra::{DMatrix, DVector};

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
