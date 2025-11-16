"""
Utility functions module.
Common helper functions used across the project.
"""

import numpy as np


def format_beta(beta, precision=4):
    """
    Pretty print beta vector.

    Args:
        beta: Parameter vector of shape (3,) with [β0, β1, β2]
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    if len(beta) != 3:
        return f"β = {np.array2string(beta, precision=precision)}"

    return f"β = [{beta[0]:.{precision}f}, {beta[1]:.{precision}f}, {beta[2]:.{precision}f}]"


def create_summary_dict(dataset, beta, error, likelihood):
    """
    Create summary dictionary of results.

    Args:
        dataset: Dataset dictionary
        beta: Final beta vector
        error: Final error value
        likelihood: Final log-likelihood value

    Returns:
        Dictionary with summary information
    """
    return {
        'num_points': len(dataset['y']),
        'final_beta': beta,
        'final_error': error,
        'final_likelihood': likelihood,
        'beta_formatted': format_beta(beta)
    }


def print_summary(summary):
    """
    Print formatted summary of results.

    Args:
        summary: Summary dictionary from create_summary_dict
    """
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"Number of points: {summary['num_points']}")
    print(f"Final beta: {summary['beta_formatted']}")
    print(f"Final error: {summary['final_error']:.6f}")
    print(f"Final log-likelihood: {summary['final_likelihood']:.4f}")
    print("="*60 + "\n")
