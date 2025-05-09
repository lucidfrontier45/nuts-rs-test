# How to use nuts-rs

## Basic Concept of PPL

Probabilistic Programming Language (PPL) is a framework to construct probabilistic models and execute their parameter estimation. Probabilistic models are characterized by those three components.

1. Prior Distribution, $\theta \sim P(\theta)$
2. Deterministic Model, $z = f(x, \theta)$
3. Observation Likelihood $y \sim P(y|z)$

The complete likelihood is given as follows.

```math
P(y|x) = \int d\theta 
```

By using PPL, all we have to do are specifying those three components, or alternatively the complete likelihood function and the framework will take care of parameter estimation by HMC, VB and so on.

## About nuts-rs

`nuts-rs` is a Rust library which implements widely used No U-turn Sampler a.k.a NUTS. It is part of PyMC project and mostly used from nutpie, a Python wrapper library. The main purpose of this project is to provide a complete Rust example to directly use nuts-rs.