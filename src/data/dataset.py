"""
Dataset module for creating and managing training data.
Structures data in the format (x0=1, x1, x2, response).
"""

import numpy as np


def create_dataset(group1_points, group2_points):
    """
    Create dataset from two groups of points.

    Group 1 is labeled as 0 (blue), Group 2 is labeled as 1 (red).

    Args:
        group1_points: Array of shape (n1, 2) with [x1, x2] for group 1
        group2_points: Array of shape (n2, 2) with [x1, x2] for group 2

    Returns:
        Dictionary containing:
            - 'X': Feature matrix of shape (n1+n2, 3) with [x0=1, x1, x2]
            - 'y': Response vector of shape (n1+n2,) with binary labels
            - 'group1_indices': Indices of group 1 points
            - 'group2_indices': Indices of group 2 points
    """
    n1 = len(group1_points)
    n2 = len(group2_points)
    total_points = n1 + n2

    # Add bias term x0 = 1
    x0_group1 = np.ones((n1, 1))
    x0_group2 = np.ones((n2, 1))

    # Combine x0 with x1, x2 for both groups
    group1_features = np.hstack([x0_group1, group1_points])
    group2_features = np.hstack([x0_group2, group2_points])

    # Combine both groups
    X = np.vstack([group1_features, group2_features])

    # Create response vector: group1 = 0 (blue), group2 = 1 (red)
    y = np.concatenate([
        np.zeros(n1),  # Group 1: blue (response = 0)
        np.ones(n2)    # Group 2: red (response = 1)
    ])

    # Store indices for each group
    group1_indices = np.arange(n1)
    group2_indices = np.arange(n1, total_points)

    return {
        'X': X,
        'y': y,
        'group1_indices': group1_indices,
        'group2_indices': group2_indices
    }


def get_features(dataset):
    """
    Extract feature matrix from dataset.

    Args:
        dataset: Dictionary returned by create_dataset()

    Returns:
        Feature matrix X of shape (n, 3) with [x0, x1, x2]
    """
    return dataset['X']


def get_responses(dataset):
    """
    Extract response vector from dataset.

    Args:
        dataset: Dictionary returned by create_dataset()

    Returns:
        Response vector y of shape (n,)
    """
    return dataset['y']


def get_group_indices(dataset):
    """
    Get indices for each group.

    Args:
        dataset: Dictionary returned by create_dataset()

    Returns:
        Tuple of (group1_indices, group2_indices)
    """
    return dataset['group1_indices'], dataset['group2_indices']


def get_dataset_size(dataset):
    """
    Get the size of the dataset.

    Args:
        dataset: Dictionary returned by create_dataset()

    Returns:
        Integer representing total number of points
    """
    return len(dataset['y'])


def split_dataset_by_response(dataset):
    """
    Split dataset back into original groups based on response.

    Args:
        dataset: Dictionary returned by create_dataset()

    Returns:
        Tuple of (group1_features, group2_features) without x0 column
    """
    X = dataset['X']
    y = dataset['y']

    # Get points without bias term (columns 1 and 2 are x1, x2)
    group1_features = X[y == 0, 1:]
    group2_features = X[y == 1, 1:]

    return group1_features, group2_features


def get_feature_statistics(dataset):
    """
    Calculate statistics for features in dataset.

    Args:
        dataset: Dictionary returned by create_dataset()

    Returns:
        Dictionary with statistics for each feature
    """
    X = dataset['X']

    return {
        'x1_mean': np.mean(X[:, 1]),
        'x1_std': np.std(X[:, 1]),
        'x2_mean': np.mean(X[:, 2]),
        'x2_std': np.std(X[:, 2]),
        'x1_range': (np.min(X[:, 1]), np.max(X[:, 1])),
        'x2_range': (np.min(X[:, 2]), np.max(X[:, 2]))
    }
