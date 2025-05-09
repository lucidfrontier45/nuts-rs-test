use std::fmt::Debug;

use ndarray::{Array1, Array2};
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, LogpError, Settings};
use thiserror::Error;

// Trait for probabilistic models
pub trait ProbabilisticModel {
    // returns the dimension of the model parameters
    fn dim(&self) -> usize;
    // returns the log-probability of the model at a given position (parameter)
    fn value(&self, position: &Array1<f64>) -> f64;
    // returns the gradient of the log-probability at a given position
    fn grad(&self, position: &Array1<f64>) -> Array1<f64>;
}

pub fn run_nuts<M: ProbabilisticModel, S: Settings>(
    model: M,
    setting: S,
    initial_position: &[f64],
    n_samples: usize,
) -> Array2<f64> {
    // construct nuts-rs's Kernel, Math, and Chain objects
    // Kernel -> Math -> Chain
    let dim = model.dim();
    let kernel = CPUKernel::new(model);
    let math = CpuMath::new(kernel);
    let mut chain = setting.new_chain(0, math, &mut rand::rng());

    // execute the NUTS algorithm, iteratively drawing samples
    chain
        .set_position(initial_position)
        .expect("Unrecoverable error during init");
    let mut trace = vec![]; // Collection of all draws
    for _ in 0..n_samples {
        let (draw, _info) = chain.draw().expect("Unrecoverable error during sampling");
        trace.push(draw);
    }

    // convert trace to Array2
    Array2::from_shape_vec((trace.len(), dim), trace.into_iter().flatten().collect()).unwrap()
}

// internal structs to interpolate ProbabilisticModel and nuts-rs's own mechanism

#[derive(Debug, Error)]
enum PosteriorLogpError {}
impl LogpError for PosteriorLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

struct CPUKernel<M: ProbabilisticModel> {
    model: M,
}

impl<M: ProbabilisticModel> CPUKernel<M> {
    fn new(model: M) -> Self {
        Self { model }
    }
}

impl<M: ProbabilisticModel> CpuLogpFunc for CPUKernel<M> {
    type LogpError = PosteriorLogpError;

    // Only used for transforming adaptation.
    type TransformParams = ();

    fn dim(&self) -> usize {
        self.model.dim()
    }

    // input: position (parameter)
    // output:
    //    - log-probability as return
    //    - gradient of log-probability stored to mutable reference grad
    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        // convert position to Array1
        let w = Array1::from_shape_vec(self.dim(), position.to_vec()).unwrap();
        let logp = self.model.value(&w);
        grad.copy_from_slice(self.model.grad(&w).as_slice().unwrap());
        Ok(logp)
    }
}
