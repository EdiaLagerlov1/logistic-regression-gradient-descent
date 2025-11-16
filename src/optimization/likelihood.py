"""
Log-likelihood module for logistic regression.
Calculates likelihood and gradient for optimization.
"""

import numpy as np
try:
    from ..models.sigmoid import predict_probability, stable_log_sigmoid, stable_log_one_minus_sigmoid
    from ..config import parameters as params
except ImportError:
    from models.sigmoid import predict_probability, stable_log_sigmoid, stable_log_one_minus_sigmoid
    from config import parameters as params


def log_likelihood(X, y, beta):
    """
    Calculate log-likelihood for logistic regression.

    Log-likelihood: L(β) = Σ[y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
    where p_i = sigmoid(X_i @ β)

    Args:
        X: Feature matrix of shape (n, 3) with [x0=1, x1, x2]
        y: Response vector of shape (n,) with binary labels {0, 1}
        beta: Parameter vector of shape (3,) with [β0, β1, β2]

    Returns:
        Scalar log-likelihood value
    """
    # Calculate z = X @ beta
    z = np.dot(X, beta)

    # Use numerically stable log-sigmoid functions
    log_p = stable_log_sigmoid(z)
    log_one_minus_p = stable_log_one_minus_sigmoid(z)

    # Calculate log-likelihood
    ll = np.sum(y * log_p + (1 - y) * log_one_minus_p)

    return ll


def log_likelihood_gradient(X, y, beta):
    """
    Calculate gradient of log-likelihood with respect to beta.

    Gradient: ∇L(β) = X^T @ (y - p)
    where p = sigmoid(X @ β)

    This is the direction of maximum increase in log-likelihood.

    Args:
        X: Feature matrix of shape (n, 3)
        y: Response vector of shape (n,)
        beta: Parameter vector of shape (3,)

    Returns:
        Gradient vector of shape (3,)
    """
    # Calculate probabilities
    probabilities = predict_probability(X, beta)

    # Calculate gradient: X^T @ (y - p)
    gradient = np.dot(X.T, y - probabilities)

    return gradient


def negative_log_likelihood(X, y, beta):
    """
    Calculate negative log-likelihood (for minimization).

    Some optimization algorithms minimize, so we provide the negative.

    Args:
        X: Feature matrix of shape (n, 3)
        y: Response vector of shape (n,)
        beta: Parameter vector of shape (3,)

    Returns:
        Negative log-likelihood value
    """
    return -log_likelihood(X, y, beta)


def negative_log_likelihood_gradient(X, y, beta):
    """
    Calculate gradient of negative log-likelihood.

    Args:
        X: Feature matrix of shape (n, 3)
        y: Response vector of shape (n,)
        beta: Parameter vector of shape (3,)

    Returns:
        Negative gradient vector of shape (3,)
    """
    return -log_likelihood_gradient(X, y, beta)


def average_log_likelihood(X, y, beta):
    """
    Calculate average log-likelihood per sample.

    Args:
        X: Feature matrix of shape (n, 3)
        y: Response vector of shape (n,)
        beta: Parameter vector of shape (3,)

    Returns:
        Average log-likelihood value
    """
    n = len(y)
    return log_likelihood(X, y, beta) / n


def calculate_likelihood_components(X, y, beta):
    """
    Calculate individual components of the log-likelihood.

    Useful for debugging and analysis.

    Args:
        X: Feature matrix of shape (n, 3)
        y: Response vector of shape (n,)
        beta: Parameter vector of shape (3,)

    Returns:
        Dictionary with likelihood components
    """
    z = np.dot(X, beta)
    probabilities = predict_probability(X, beta)

    # Add small epsilon to prevent log(0)
    p_safe = np.clip(probabilities, params.EPSILON, 1 - params.EPSILON)

    # Individual log-likelihood contributions
    ll_positive = y * np.log(p_safe)
    ll_negative = (1 - y) * np.log(1 - p_safe)
    ll_individual = ll_positive + ll_negative

    return {
        'total_log_likelihood': np.sum(ll_individual),
        'average_log_likelihood': np.mean(ll_individual),
        'positive_contribution': np.sum(ll_positive),
        'negative_contribution': np.sum(ll_negative),
        'individual_ll': ll_individual,
        'probabilities': probabilities
    }
