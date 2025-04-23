#!/usr/bin/env python3
"""
Example script demonstrating how to use the Poisson generator with different parameters.
"""

from poisson_generator import (
    generate_poisson_samples,
    calculate_empirical_distribution,
    calculate_theoretical_distribution,
    calculate_mean_variance,
    plot_distributions
)

def run_example(lambda_val, num_samples, filename_prefix):
    """
    Run an example with the given parameters.
    
    Args:
        lambda_val (float): The mean parameter of the Poisson distribution.
        num_samples (int): The number of samples to generate.
        filename_prefix (str): Prefix for the output filename.
    """
    print(f"\nRunning example with λ = {lambda_val}, samples = {num_samples}")
    
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
    filename = f"{filename_prefix}_lambda_{lambda_val}_samples_{num_samples}.png"
    
    # Create a custom plot function that saves with the specific filename
    import matplotlib.pyplot as plt
    
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
    
    # Save the figure with the specific filename
    plt.savefig(filename)
    plt.close()
    
    print(f"Plot saved as '{filename}'")

def main():
    """
    Run examples with different parameters.
    """
    print("Running examples with different parameters...")
    
    # Example 1: Low lambda, moderate sample size
    run_example(lambda_val=1.0, num_samples=5000, filename_prefix="poisson_low")
    
    # Example 2: Medium lambda, moderate sample size
    run_example(lambda_val=5.0, num_samples=5000, filename_prefix="poisson_medium")
    
    # Example 3: High lambda, moderate sample size
    run_example(lambda_val=10.0, num_samples=5000, filename_prefix="poisson_high")
    
    # Example 4: Medium lambda, large sample size
    run_example(lambda_val=5.0, num_samples=20000, filename_prefix="poisson_large_sample")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main()