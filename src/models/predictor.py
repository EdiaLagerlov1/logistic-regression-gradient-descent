"""
Prediction module for classification.
Converts sigmoid probabilities to binary classifications.
"""

import numpy as np
try:
    from .sigmoid import predict_probability
    from ..config import parameters as params
except ImportError:
    from models.sigmoid import predict_probability
    from config import parameters as params


def predict_class(probabilities, threshold=None):
    """
    Convert probabilities to binary class predictions.

    Args:
        probabilities: Array of probabilities in range [0, 1]
        threshold: Classification threshold (default from params)

    Returns:
        Binary array with 0 or 1 class predictions
    """
    if threshold is None:
        threshold = params.CLASSIFICATION_THRESHOLD

    # Convert probabilities to binary: 1 if prob >= threshold, else 0
    predictions = (probabilities >= threshold).astype(int)

    return predictions


def predict_response(X, beta, threshold=None):
    """
    Predict binary responses for feature matrix.

    Combines sigmoid probability calculation with binary classification.

    Args:
        X: Feature matrix of shape (n, 3) with [x0=1, x1, x2]
        beta: Parameter vector of shape (3,) with [β0, β1, β2]
        threshold: Classification threshold (default from params)

    Returns:
        Binary array of shape (n,) with 0 or 1 predictions
    """
    # Calculate probabilities
    probabilities = predict_probability(X, beta)

    # Convert to binary predictions
    predictions = predict_class(probabilities, threshold)

    return predictions


def predict_with_probabilities(X, beta, threshold=None):
    """
    Predict both probabilities and binary classes.

    Args:
        X: Feature matrix of shape (n, 3)
        beta: Parameter vector of shape (3,)
        threshold: Classification threshold (default from params)

    Returns:
        Dictionary containing:
            - 'probabilities': Continuous probabilities
            - 'predictions': Binary class predictions
    """
    probabilities = predict_probability(X, beta)
    predictions = predict_class(probabilities, threshold)

    return {
        'probabilities': probabilities,
        'predictions': predictions
    }


def split_predictions_by_class(X, beta, threshold=None):
    """
    Split predictions into two groups based on predicted class.

    Args:
        X: Feature matrix of shape (n, 3)
        beta: Parameter vector of shape (3,)
        threshold: Classification threshold (default from params)

    Returns:
        Tuple of (class0_indices, class1_indices)
    """
    predictions = predict_response(X, beta, threshold)

    class0_indices = np.where(predictions == 0)[0]
    class1_indices = np.where(predictions == 1)[0]

    return class0_indices, class1_indices


def calculate_decision_boundary(beta, x1_range):
    """
    Calculate decision boundary line for visualization.

    The decision boundary is where sigmoid(X @ beta) = 0.5,
    which occurs when X @ beta = 0.

    For x0=1: β0 + β1*x1 + β2*x2 = 0
    Solving for x2: x2 = -(β0 + β1*x1) / β2

    Args:
        beta: Parameter vector of shape (3,) with [β0, β1, β2]
        x1_range: Array of x1 values to calculate boundary for

    Returns:
        Array of x2 values for the decision boundary, or None if β2 == 0
    """
    if np.abs(beta[2]) < 1e-10:
        # Vertical or no decision boundary
        return None

    # Calculate x2 values: x2 = -(β0 + β1*x1) / β2
    x2_values = -(beta[0] + beta[1] * x1_range) / beta[2]

    return x2_values


def get_classification_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics.

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels

    Returns:
        Dictionary with accuracy and confusion matrix values
    """
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)

    # Calculate confusion matrix elements
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return {
        'accuracy': accuracy,
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative
    }
