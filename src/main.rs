use std::f64::consts::PI;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Normal, Uniform},
};
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, DiagGradNutsSettings, LogpError, Settings};
use thiserror::Error;

trait ProbabilisticModel {
    fn value(&self, position: &Array1<f64>) -> f64;
    fn grad(&self, position: &Array1<f64>) -> Array1<f64>;
}

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

// The density might fail in a recoverable or non-recoverable manner...
#[derive(Debug, Error)]
enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

// Define a function that computes the unnormalized posterior density
// and its gradient.
impl CpuLogpFunc for LinearRegressionModel {
    type LogpError = PosteriorLogpError;

    // Only used for transforming adaptation.
    type TransformParams = ();

    fn dim(&self) -> usize {
        self.x.shape()[1]
    }

    // The normal likelihood with mean 3 and its gradient.
    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        // convert position to Array1
        let w = Array1::from_shape_vec(self.dim(), position.to_vec()).unwrap();
        let logp = self.value(&w);
        grad.copy_from_slice(self.grad(&w).as_slice().unwrap());
        Ok(logp)
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
    let logp_func = LinearRegressionModel::new(x, y);
    let math = CpuMath::new(logp_func);

    // normal rand 0.9 rng
    let mut rng = rand::rng();
    let mut sampler = settings.new_chain(0, math, &mut rng);

    // Set to some initial position and start drawing samples.
    sampler
        .set_position(&vec![0f64; d])
        .expect("Unrecoverable error during init");
    let mut trace = vec![]; // Collection of all draws
    for _ in 0..2000 {
        let (draw, _info) = sampler.draw().expect("Unrecoverable error during sampling");
        trace.push(draw);
    }
    // convert trace to Array2
    let trace =
        Array2::from_shape_vec((trace.len(), d), trace.into_iter().flatten().collect()).unwrap();

    let w_mean = trace.mean_axis(Axis(0)).unwrap();
    dbg!(w_mean);
}
