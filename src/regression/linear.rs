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
    let mut new_theta: DVector<f64> = DVector::<f64>::zeros(theta.len());

    new_theta.copy_from(theta);

    let alpha_div_m = alpha / (m as f64);

    for _ in 0..num_iters {
        unsafe {
            let mut x_by_theta = DVector::new_uninitialized(x.nrows());

            x.mul_to(&new_theta, &mut x_by_theta);

            let delta = x.transpose() * (x_by_theta - y);

            new_theta = new_theta - (alpha_div_m * delta);
        }
    }

    new_theta
}

pub fn feature_normalize(x: &DMatrix<f64>) -> DMatrix<f64> {
    let num_rows = x.nrows();
    let num_cols = x.ncols();

    let mut normalized_x = DMatrix::from_element(num_rows, num_cols, 0f64);

    for i in 0..num_cols {
        let vals = x.column(i);
        let mean = compute_mx_slice_mean(num_rows, &vals);
        let std = compute_mx_slice_std(num_rows, mean, &vals);

        let new_col = DVector::from_fn(num_rows, |j, _| {
            (x.index((j, i)) - mean) / std
        });

        normalized_x.set_column(i, &new_col);
    }

    normalized_x
}

fn compute_mx_slice_mean(n: usize, x: &Matrix<f64, Dynamic, U1, SliceStorage<'_, f64, Dynamic, U1, U1, Dynamic>>) -> f64 {
    let mut sum: f64 = 0.0;

    for v in x.iter() {
        sum = sum + v;
    }

    sum / (n as f64)
}

fn compute_mx_slice_std(n: usize, mean: f64, x: &Matrix<f64, Dynamic, U1, SliceStorage<'_, f64, Dynamic, U1, U1, Dynamic>>) -> f64 {
    let mut sum_of_squares: f64 = 0.0;

    for v in x.iter() {
        sum_of_squares = sum_of_squares + (v - mean).powf(2.0);
    }

    (sum_of_squares / ((n - 1) as f64)).sqrt()
}

#[cfg(test)]
mod tests {
    extern crate csv;
    extern crate nalgebra;

    use std::env::current_dir;
    use csv::{ReaderBuilder, StringRecord};
    use nalgebra::{DMatrix, DVector, Matrix, Dim};
    use nalgebra::base::storage::Storage;

    use crate::regression::linear::{compute_cost, feature_normalize, gradient_descent};

    fn read_sample_data() -> (DMatrix<f64>, DMatrix<f64>) {
        let mut test_data_dir = current_dir().unwrap();
        test_data_dir.push("test_data");

        let mut linear_regresion_file_path = test_data_dir.clone();
        linear_regresion_file_path.push("linearRegression.csv");

        let mut feature_normalize_file_path = test_data_dir.clone();
        feature_normalize_file_path.push("lrFeatureNormalize.csv");


        let linear_regression_data_reader = ReaderBuilder::new()
            .has_headers(false)
            .from_path(linear_regresion_file_path.as_path()).unwrap();

        let feature_normalize_data_reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b',')
            .from_path(feature_normalize_file_path.as_path()).unwrap();

        let mut linear_regression_data: Vec<f64> = Vec::new();
        for record in linear_regression_data_reader.into_records() {
            let record: StringRecord = record.unwrap();

            for field in record.iter() {
                let value: f64 = field.parse().unwrap();
                let i = linear_regression_data.len();

                linear_regression_data.insert(i, value);
            }
        }

        let mut feature_normalize_data: Vec<f64> = Vec::new();
        for record in feature_normalize_data_reader.into_records() {
            let record: StringRecord = record.unwrap();

            for field in record.iter() {
                let value: f64 = field.parse().unwrap();
                let i = feature_normalize_data.len();

                feature_normalize_data.insert(i, value);
            }
        }

        let lr_data_rows = linear_regression_data.len() / 3;
        let fn_data_rows = feature_normalize_data.len() / 2;

        let features = DMatrix::from_vec(3, lr_data_rows, linear_regression_data).transpose();
        let normalized_features = DMatrix::from_vec(2, fn_data_rows, feature_normalize_data).transpose();

        (features, normalized_features)
    }

    fn float_eq(x0: f64, x1: f64) -> bool {
        (x0 - x1).abs() < 0.0001
    }

    fn matrix_eq<R2, C2, SB>(m0: &Matrix<f64, R2, C2, SB>, m1: &Matrix<f64, R2, C2, SB>) -> bool
    where
        R2: Dim,
        C2: Dim,
        SB: Storage<f64, R2, C2>
    {
        if m0.len() != m1.len() {
            false
        } else {
            let mut m1_iter = m1.iter();
            let mut are_equal = true;

            for v0 in m0.iter() {
                if !float_eq(*v0, *(m1_iter.next().unwrap())) {
                    are_equal = false;
                    break;
                }
            }

            are_equal
        }
    }

    fn is_descending(vec: &Vec<f64>) -> bool {
        if vec.len() == 0 || vec.len() == 1 {
            true
        } else {
            let mut is_desc = true;
            let mut vec_iter = vec.iter();
            let mut cur_v = vec_iter.next().unwrap();

            for v in vec_iter {
                if cur_v >= v {
                    cur_v = v;
                } else {
                    is_desc = false;
                    break;
                }
            }

            is_desc
        }
    }

    #[test]
    fn cost_computes_expected_result() {
        let (features, _) = read_sample_data();
        let results_col = features.column(2);

        let mut results = DVector::from_element(features.nrows(), 0f64);
        results.copy_from(&results_col);

        let features: DMatrix<f64> = features.remove_column(2);
        let features: DMatrix<f64> = features.insert_column(0, 1.0);

        let theta0 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let r0 = compute_cost(&features, &results, &theta0);
        let expected_value0 = 65591548106.45744;

        assert!(float_eq(r0, expected_value0));

        let theta1 = DVector::from_vec(vec![25.0, 26.0, 27.0]);
        let r1 = compute_cost(&features, &results, &theta1);
        let expected_value1 = 47251185844.64893;

        assert!(float_eq(r1, expected_value1));

        let theta2 = DVector::from_vec(vec![1500.0, 227.0, 230.0]);
        let r2 = compute_cost(&features, &results, &theta2);
        let expected_value2 = 11433546085.01064;

        assert!(float_eq(r2, expected_value2));

        let theta3 = DVector::from_vec(vec![-15.03, -27.123, -59.675]);
        let r3 = compute_cost(&features, &results, &theta3);
        let expected_value3 = 88102482793.02190;

        assert!(float_eq(r3, expected_value3));
    }

    #[test]
    fn features_normalize_computes_expected_matrix() {
        let (features, expected_normalized_features) = read_sample_data();
        let features: DMatrix<f64> = features.remove_column(2);
        let normalized_features = feature_normalize(&features);

        assert!(matrix_eq(&normalized_features, &expected_normalized_features));
    }

    #[test]
    fn gradient_descent_computes_expected_thetas() {
        let (features, _) = read_sample_data();
        let results_col = features.column(2);

        let mut results = DVector::from_element(features.nrows(), 0f64);
        results.copy_from(&results_col);

        let features: DMatrix<f64> = features.remove_column(2);
        let features: DMatrix<f64> = feature_normalize(&features);
        let features: DMatrix<f64> = features.insert_column(0, 1.0);

        let theta0 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let new_theta0 = gradient_descent(&features, &results, &theta0, 0.1, 50);
        let expected_theta0 = DVector::from_vec(vec![338658.24925, 104127.51560, -172.20533]);

        assert!(matrix_eq(&new_theta0, &expected_theta0));

        let theta1 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let new_theta1 = gradient_descent(&features, &results, &theta1, 0.01, 500);
        let expected_theta1 = DVector::from_vec(vec![338175.98397, 103831.11737, 103.03073]);

        assert!(matrix_eq(&new_theta1, &expected_theta1));

        let theta2 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let new_theta2 = gradient_descent(&features, &results, &theta2, 0.001, 1000);
        let expected_theta2 = DVector::from_vec(vec![215244.48211, 61233.08697, 20186.40938]);

        assert!(matrix_eq(&new_theta2, &expected_theta2));
    }

    #[test]
    fn cost_function_decreases_the_more_iterations_of_gradient_descent() {
        let (features, _) = read_sample_data();
        let results_col = features.column(2);

        let mut results = DVector::from_element(features.nrows(), 0f64);
        results.copy_from(&results_col);

        let features: DMatrix<f64> = features.remove_column(2);
        let features: DMatrix<f64> = feature_normalize(&features);
        let features: DMatrix<f64> = features.insert_column(0, 1.0);

        let theta = DVector::from_vec(vec![0.0, 0.0, 0.0]);

        let iter_vals = (1..101).map(|n| 10 * n);

        let mut cost_function_vals: Vec<f64> = Vec::new();

        for i in iter_vals {
            let new_theta = gradient_descent(&features, &results, &theta, 0.01, i);
            let r0 = compute_cost(&features, &results, &new_theta);

            cost_function_vals.insert(cost_function_vals.len(), r0);
        }

        assert!(is_descending(&cost_function_vals));
    }
}
