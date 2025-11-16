"""
Error metrics module for evaluation.
Calculates Euclidean distance and other error measures.
"""

import numpy as np
try:
    from ..models.sigmoid import predict_probability
except ImportError:
    from models.sigmoid import predict_probability


def euclidean_distance(actual, predicted):
    """
    Calculate element-wise Euclidean distance between actual and predicted values.

    For vectors: |actual_i - predicted_i|

    Args:
        actual: Array of actual values
        predicted: Array of predicted values

    Returns:
        Array of distances, same shape as inputs
    """
    return np.abs(actual - predicted)


def average_error(actual, predicted):
    """
    Calculate mean Euclidean distance between actual and predicted values.

    Args:
        actual: Array of actual values
        predicted: Array of predicted values

    Returns:
        Scalar mean distance
    """
    distances = euclidean_distance(actual, predicted)
    return np.mean(distances)


def calculate_error_for_beta(X, y, beta):
    """
    Calculate average prediction error for a single beta vector.

    Uses sigmoid probabilities as predictions.

    Args:
        X: Feature matrix of shape (n, 3)
        y: Actual response vector of shape (n,)
        beta: Parameter vector of shape (3,)

    Returns:
        Average Euclidean distance between y and sigmoid(X @ beta)
    """
    # Get predicted probabilities
    predicted = predict_probability(X, beta)

    # Calculate average error
    error = average_error(y, predicted)

    return error


def calculate_errors_for_betas(X, y, beta_dict):
    """
    Calculate average error for multiple beta vectors.

    Args:
        X: Feature matrix of shape (n, 3)
        y: Actual response vector of shape (n,)
        beta_dict: Dictionary of {iteration: beta}

    Returns:
        Dictionary of {iteration: error}
    """
    errors = {}
    for iteration, beta in beta_dict.items():
        error = calculate_error_for_beta(X, y, beta)
        errors[iteration] = error
    return errors
