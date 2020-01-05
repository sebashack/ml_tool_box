use ml_tool_box::regression::linear::{compute_cost};

use nalgebra::{DMatrix, DVector};

fn main() {
    let x = DMatrix::from_iterator(3, 3, vec![1.0, 4.0, 7.0,
                                              2.0, 5.0, 8.0,
                                              3.0, 6.0, 9.0]);
    let y = DVector::from_iterator(3, vec![4.0, 6.0, 8.0]);
    let theta = DVector::from_iterator(3, vec![4.0, 6.0, 8.0]);

    println!("Result {}", compute_cost(&x, &y, &theta));
}
