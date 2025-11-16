"""
Gradient descent optimization module.
Implements gradient ascent for maximum log-likelihood estimation.
"""

import numpy as np
try:
    from .likelihood import log_likelihood, log_likelihood_gradient
    from ..config import parameters as params
except ImportError:
    from optimization.likelihood import log_likelihood, log_likelihood_gradient
    from config import parameters as params


def gradient_descent(X, y, initial_beta=None, learning_rate=None, num_iterations=None):
    """
    Perform gradient ascent to maximize log-likelihood.

    Updates: beta_new = beta_old + learning_rate * gradient

    Args:
        X: Feature matrix of shape (n, 3)
        y: Response vector of shape (n,)
        initial_beta: Starting parameter vector (default from params)
        learning_rate: Step size (default from params)
        num_iterations: Number of iterations (default from params)

    Returns:
        Dictionary containing:
            - 'final_beta': Optimized beta vector
            - 'beta_history': Dictionary of {iteration: beta}
            - 'likelihood_history': List of log-likelihood values
    """
    # Use default parameters if not provided
    if initial_beta is None:
        initial_beta = params.INITIAL_BETA.copy()
    if learning_rate is None:
        learning_rate = params.LEARNING_RATE
    if num_iterations is None:
        num_iterations = params.NUM_ITERATIONS

    # Initialize
    beta = initial_beta.copy()
    beta_history = {0: beta.copy()}
    likelihood_history = [log_likelihood(X, y, beta)]

    # Gradient ascent iterations
    for iteration in range(1, num_iterations + 1):
        # Calculate gradient
        gradient = log_likelihood_gradient(X, y, beta)

        # Update beta (gradient ascent)
        beta = beta + learning_rate * gradient

        # Store beta and likelihood
        beta_history[iteration] = beta.copy()
        ll = log_likelihood(X, y, beta)
        likelihood_history.append(ll)

    return {
        'final_beta': beta,
        'beta_history': beta_history,
        'likelihood_history': likelihood_history
    }


def store_beta_snapshots(beta_history, step=None):
    """
    Extract beta snapshots at regular intervals.

    Keeps betas at iterations: 5*step, 4*step, 3*step, 2*step, step, 0

    Args:
        beta_history: Dictionary of {iteration: beta} from gradient_descent
        step: Interval for storing betas (default from params)

    Returns:
        Dictionary of {iteration: beta} for selected iterations
    """
    if step is None:
        step = params.BETA_HISTORY_STEP

    max_iteration = max(beta_history.keys())

    # Calculate snapshot iterations
    snapshot_iterations = []
    for multiplier in [5, 4, 3, 2, 1]:
        iteration = multiplier * step
        if iteration <= max_iteration:
            snapshot_iterations.append(iteration)

    # Always include iteration 0 and final iteration
    snapshot_iterations.append(0)
    if max_iteration not in snapshot_iterations:
        snapshot_iterations.append(max_iteration)

    # Extract betas for snapshot iterations
    snapshots = {}
    for iteration in sorted(set(snapshot_iterations)):
        if iteration in beta_history:
            snapshots[iteration] = beta_history[iteration]

    return snapshots
