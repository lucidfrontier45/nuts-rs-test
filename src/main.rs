use std::f64::consts::PI;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Normal, Uniform},
};
use nuts_rs::DiagGradNutsSettings;
use nutsrs_test::ProbabilisticModel;

// loglikelihood function for 1d normal distribution
fn normal_logp(z: f64, mean: f64, sd: f64) -> f64 {
    let diff = z - mean;
    -0.5 * ((sd.ln() + (PI * 2.0).ln()) + (diff / sd).powi(2))
}

struct LinearRegressionModel {
    x: Array2<f64>,
    y: Array1<f64>,
}

impl LinearRegressionModel {
    fn new(x: Array2<f64>, y: Array1<f64>) -> Self {
        Self { x, y }
    }
}

impl ProbabilisticModel for LinearRegressionModel {
    fn dim(&self) -> usize {
        self.x.shape()[1]
    }

    fn value(&self, position: &Array1<f64>) -> f64 {
        let y_hat = self.x.dot(position);
        y_hat
            .into_iter()
            .zip(self.y.iter())
            .map(|(y_hat, y)| normal_logp(y_hat, *y, 1.0))
            .sum()
    }

    fn grad(&self, position: &Array1<f64>) -> Array1<f64> {
        let y_hat = self.x.dot(position);
        self.x.t().dot(&(&self.y - y_hat))
    }
}

fn main() {
    // current ndarray requires rand 0.8 rng
    let mut rng = ndarray_rand::rand::thread_rng();
    let n = 100;

    let w = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let d = w.len();

    // sample X from 3d uniform [-2.0, 2.0]
    let x = Array2::random_using((n, d), Uniform::new(-2.0, 2.0), &mut rng);

    // sample e from 1d normal(0, 1.5)
    let e = Array1::random_using(n, Normal::new(0.0, 1.5).unwrap(), &mut rng);

    // construct y = X * w + e
    let y = x.dot(&w) + e;

    // We get the default sampler arguments
    let mut settings = DiagGradNutsSettings::default();

    // and modify as we like
    settings.num_tune = 1000;
    settings.maxdepth = 10;

    // We instanciate our posterior density function
    let model = LinearRegressionModel::new(x, y);
    let trace = nutsrs_test::run_nuts(model, settings, &vec![0.0; d], 1000);

    let w_mean = trace.mean_axis(Axis(0)).unwrap();
    dbg!(w_mean);
}
