use ml_tool_box::regression::linear::{compute_cost, gradient_descent, featureNormalize};

use nalgebra::{DMatrix, DVector};

fn main() {
    let x = DMatrix::from_iterator(3, 3, vec![1000.0, 400.0, 70.0,
                                              2000.0, 500.0, 80.0,
                                              3000.0, 600.0, 90.0]);
    let y = DVector::from_iterator(3, vec![4000.0, 600.0, 80.0]);
    let theta = DVector::from_iterator(3, vec![0.0, 0.0, 0.0]);

    let normalized_x = featureNormalize(&x);

    println!("Result {}", normalized_x);
}
