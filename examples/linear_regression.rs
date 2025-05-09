use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::{
    RandomExt,
    rand::Rng,
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
    sigma: f64,
}

impl LinearRegressionModel {
    fn new(x: Array2<f64>, y: Array1<f64>, sigma: f64) -> Self {
        Self { x, y, sigma }
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
            .map(|(y_hat, y)| normal_logp(y_hat, *y, self.sigma))
            .sum()
    }

    fn grad(&self, position: &Array1<f64>) -> Array1<f64> {
        let y_hat = self.x.dot(position);
        self.x.t().dot(&(&self.y - y_hat))
    }
}

// given one sample of w, predict y from the distribution
// y ~ N(X * w, sigma^2)
fn predict(w: &[f64], sigma: f64, x: &[f64], n_samples: usize) -> Array1<f64> {
    let mut rng = ndarray_rand::rand::thread_rng();
    let position = ArrayView1::from(w);
    let x = ArrayView1::from(x);
    let z = x.dot(&position);
    let e = Array1::random_using(n_samples, Normal::new(0.0, sigma).unwrap(), &mut rng);
    z + e
}

// given multiple samples of w, predict y from multiple distributions
// y ~ 1/N sum_w N(X * w, sigma^2)
fn posterior_predictive(ws: &Array2<f64>, sigma: f64, x: &[f64], n_samples: usize) -> Array1<f64> {
    let k = ws.shape()[0];
    let res = (0..k)
        .flat_map(|i| predict(ws.row(i).as_slice().unwrap(), sigma, x, n_samples).to_vec())
        .collect::<Vec<_>>();
    Array1::from_vec(res)
}

fn prepare_input<T: Rng>(n: usize, rng: &mut T) -> (Array2<f64>, Array1<f64>, f64, usize) {
    let w = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let d = w.len();
    let x = Array2::random_using((n, d), Uniform::new(-2.0, 2.0), rng);
    let sigma = 0.1;
    let e = Array1::random_using(n, Normal::new(0.0, sigma).unwrap(), rng);
    let y = x.dot(&w) + e;
    (x, y, sigma, d)
}

fn main() {
    // current ndarray requires rand 0.8 rng
    let mut rng = ndarray_rand::rand::thread_rng();
    let n = 100;

    let (x, y, sigma, d) = prepare_input(n, &mut rng);

    // We get the default sampler arguments
    let mut settings = DiagGradNutsSettings::default();

    // and modify as we like
    settings.num_tune = 1000;
    settings.maxdepth = 10;

    // create the model
    let model = LinearRegressionModel::new(x, y, sigma);
    // draw posterior samples
    let trace = nutsrs_test::run_nuts(model, settings, &vec![0.0; d], 1000);

    // analyze the posterior mean
    let w_mean = trace.mean_axis(ndarray::Axis(0)).unwrap();
    println!("Posterior mean: {:?}", w_mean);

    // test posterior predictive
    let x_new = [2.5, -1.1, 0.3];
    let y_preds = posterior_predictive(&trace, sigma, &x_new, 1);
    let mut y_preds = y_preds.to_vec();
    // calculate the 5%, 50%, and 95% quantiles
    y_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q5 = y_preds[y_preds.len() * 10 / 100];
    let q50 = y_preds[y_preds.len() * 50 / 100];
    let q95 = y_preds[y_preds.len() * 90 / 100];
    println!(
        "Posterior predictive 10%: {:.2}, 50%: {:.2}, 90%: {:.2}",
        q5, q50, q95
    );
}
