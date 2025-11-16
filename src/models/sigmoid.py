"""
Sigmoid function module.
Implements the logistic sigmoid function with numerical stability.
"""

import numpy as np
try:
    from ..config import parameters as params
except ImportError:
    from config import parameters as params


def sigmoid(z):
    """
    Compute the sigmoid function: 1 / (1 + exp(-z)).

    Implements numerical stability by clipping extreme values.

    Args:
        z: Scalar, array, or matrix of any shape

    Returns:
        Sigmoid output of the same shape as z, values in range (0, 1)
    """
    # Clip values to prevent overflow in exp
    z_clipped = np.clip(z, -params.CLIP_VALUE, params.CLIP_VALUE)

    # Compute sigmoid
    return 1.0 / (1.0 + np.exp(-z_clipped))


def predict_probability(X, beta):
    """
    Calculate predicted probabilities using sigmoid function.

    Computes: sigmoid(X @ beta)

    Args:
        X: Feature matrix of shape (n, 3) with [x0=1, x1, x2]
        beta: Parameter vector of shape (3,) with [β0, β1, β2]

    Returns:
        Probability vector of shape (n,) with values in range (0, 1)
    """
    # Compute linear combination: z = X @ beta
    z = np.dot(X, beta)

    # Apply sigmoid function
    probabilities = sigmoid(z)

    return probabilities


def sigmoid_derivative(z):
    """
    Compute derivative of sigmoid function.

    Derivative: sigmoid(z) * (1 - sigmoid(z))

    Args:
        z: Scalar, array, or matrix of any shape

    Returns:
        Derivative values of the same shape as z
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


def stable_log_sigmoid(z):
    """
    Compute log(sigmoid(z)) in a numerically stable way.

    Uses the identity: log(sigmoid(z)) = -log(1 + exp(-z))

    Args:
        z: Scalar, array, or matrix of any shape

    Returns:
        Log-sigmoid values of the same shape as z
    """
    # For numerical stability
    z_clipped = np.clip(z, -params.CLIP_VALUE, params.CLIP_VALUE)

    # Use different formulas depending on sign of z
    # If z >= 0: log(sigmoid(z)) = -log(1 + exp(-z))
    # If z < 0: log(sigmoid(z)) = z - log(1 + exp(z))

    result = np.where(
        z_clipped >= 0,
        -np.log1p(np.exp(-z_clipped)),
        z_clipped - np.log1p(np.exp(z_clipped))
    )

    return result


def stable_log_one_minus_sigmoid(z):
    """
    Compute log(1 - sigmoid(z)) in a numerically stable way.

    Uses the identity: log(1 - sigmoid(z)) = log(sigmoid(-z))

    Args:
        z: Scalar, array, or matrix of any shape

    Returns:
        Log(1 - sigmoid) values of the same shape as z
    """
    # log(1 - sigmoid(z)) = log(sigmoid(-z))
    return stable_log_sigmoid(-z)


def apply_sigmoid_to_dataset(X, beta):
    """
    Apply sigmoid function to entire dataset.

    Convenience function that combines feature matrix with beta parameters.

    Args:
        X: Feature matrix of shape (n, 3)
        beta: Parameter vector of shape (3,)

    Returns:
        Dictionary containing:
            - 'probabilities': Probability predictions
            - 'z_values': Linear combinations (X @ beta)
    """
    z = np.dot(X, beta)
    probabilities = sigmoid(z)

    return {
        'probabilities': probabilities,
        'z_values': z
    }
