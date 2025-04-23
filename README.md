# Numerical Poisson Distribution Generation

This repository contains Python code for numerically generating Poisson random variables using exponential random variables. The implementation follows a step-by-step approach to generate Poisson distributed samples and compares them with the theoretical Poisson distribution.

## Method

The implementation uses the following approach:

1. **Generate Exponential Random Variables** using the inverse function method:
   - Generate a uniform random variable u from [0, 1]
   - Calculate x = -ln(1-u)/lambda to get an exponential random variable

2. **Generate Poisson Random Variables**:
   - Generate exponential random variables E1, E2, E3, ...
   - Sum them until the sum exceeds 1
   - Count the number of variables needed (minus 1) to get a Poisson random variable

3. **Create Probability Distribution**:
   - Generate multiple Poisson random variables
   - Calculate the frequency of each value
   - Compare with the theoretical Poisson distribution

## Usage

To run the code:

```bash
python poisson_generator.py
```

This will:
1. Generate 10,000 Poisson random variables with lambda = 5
2. Calculate the empirical distribution
3. Compare with the theoretical distribution
4. Plot both distributions and save the result as 'poisson_distribution_comparison.png'

## Requirements

- NumPy
- Matplotlib
- SciPy

## Mathematical Background

The Poisson distribution with parameter lambda gives the probability of observing k events in a fixed interval:

P(k; lambda) = (lambda^k * e^(-lambda)) / k!

This implementation uses the relationship between Poisson and exponential distributions to generate Poisson random variables.
