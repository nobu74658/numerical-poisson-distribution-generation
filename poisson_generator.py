#!/usr/bin/env python3
"""
Numerical Generation of Poisson Distribution

This script implements a method to generate Poisson random variables using
exponential random variables, and compares the generated distribution with
the theoretical Poisson distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from collections import Counter
import math

def generate_exponential_random_variable(lambda_val):
    """
    Generate an exponential random variable using the inverse function method.
    
    Args:
        lambda_val (float): The rate parameter of the exponential distribution.
        
    Returns:
        float: A random variable following the exponential distribution.
    """
    u = np.random.uniform(0, 1)
    return -np.log(1 - u) / lambda_val

def generate_poisson_random_variable(lambda_val):
    """
    Generate a Poisson random variable using exponential random variables.
    
    Args:
        lambda_val (float): The mean parameter of the Poisson distribution.
        
    Returns:
        int: A random variable following the Poisson distribution.
    """
    count = 0
    sum_exp = 0
    
    while sum_exp < 1:
        exp_var = generate_exponential_random_variable(lambda_val)
        sum_exp += exp_var
        count += 1
    
    # We count the number of exponential variables needed before the sum exceeds 1
    # The last one pushes it over 1, so we subtract 1 from the count
    return count - 1

def generate_poisson_samples(lambda_val, num_samples):
    """
    Generate multiple Poisson random variables.
    
    Args:
        lambda_val (float): The mean parameter of the Poisson distribution.
        num_samples (int): The number of samples to generate.
        
    Returns:
        list: A list of Poisson random variables.
    """
    samples = []
    for _ in range(num_samples):
        samples.append(generate_poisson_random_variable(lambda_val))
    return samples

def calculate_empirical_distribution(samples):
    """
    Calculate the empirical probability distribution from the samples.
    
    Args:
        samples (list): A list of Poisson random variables.
        
    Returns:
        tuple: A tuple containing (values, probabilities).
    """
    counter = Counter(samples)
    total = len(samples)
    
    # Get the range of values
    min_val = min(counter.keys())
    max_val = max(counter.keys())
    
    values = list(range(min_val, max_val + 1))
    probabilities = [counter.get(val, 0) / total for val in values]
    
    return values, probabilities

def calculate_theoretical_distribution(lambda_val, values):
    """
    Calculate the theoretical Poisson distribution.
    
    Args:
        lambda_val (float): The mean parameter of the Poisson distribution.
        values (list): The values for which to calculate the probabilities.
        
    Returns:
        list: The theoretical probabilities for each value.
    """
    return [poisson.pmf(val, lambda_val) for val in values]

def plot_distributions(values, empirical_probs, theoretical_probs, lambda_val, num_samples):
    """
    Plot the empirical and theoretical distributions.
    
    Args:
        values (list): The values for which the probabilities are calculated.
        empirical_probs (list): The empirical probabilities.
        theoretical_probs (list): The theoretical probabilities.
        lambda_val (float): The mean parameter of the Poisson distribution.
        num_samples (int): The number of samples used.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot empirical distribution
    plt.bar(values, empirical_probs, alpha=0.5, label='Empirical Distribution')
    
    # Plot theoretical distribution
    plt.plot(values, theoretical_probs, 'ro-', label='Theoretical Distribution')
    
    plt.title(f'Poisson Distribution (λ = {lambda_val}, Samples = {num_samples})')
    plt.xlabel('k')
    plt.ylabel('Probability P(X = k)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig('poisson_distribution_comparison.png')
    plt.show()

def calculate_mean_variance(samples):
    """
    Calculate the mean and variance of the samples.
    
    Args:
        samples (list): A list of Poisson random variables.
        
    Returns:
        tuple: A tuple containing (mean, variance).
    """
    mean = np.mean(samples)
    variance = np.var(samples)
    return mean, variance

def main():
    # Parameters
    lambda_val = 5.0  # Mean of the Poisson distribution
    num_samples = 10000  # Number of samples to generate
    
    print(f"Generating {num_samples} Poisson random variables with λ = {lambda_val}...")
    
    # Generate Poisson samples
    samples = generate_poisson_samples(lambda_val, num_samples)
    
    # Calculate empirical distribution
    values, empirical_probs = calculate_empirical_distribution(samples)
    
    # Calculate theoretical distribution
    theoretical_probs = calculate_theoretical_distribution(lambda_val, values)
    
    # Calculate mean and variance
    mean, variance = calculate_mean_variance(samples)
    print(f"Empirical Mean: {mean:.4f} (Theoretical: {lambda_val})")
    print(f"Empirical Variance: {variance:.4f} (Theoretical: {lambda_val})")
    
    # Plot the distributions
    plot_distributions(values, empirical_probs, theoretical_probs, lambda_val, num_samples)
    
    print("Done! The comparison plot has been saved as 'poisson_distribution_comparison.png'.")

if __name__ == "__main__":
    main()