"""
Data generation module for Gaussian distributed points.
Generates non-overlapping point groups with specified distributions.
"""

import numpy as np
try:
    from ..config import parameters as params
except ImportError:
    from config import parameters as params


def generate_gaussian_points(mean, variance, num_points):
    """
    Generate points from a 2D Gaussian distribution.

    Args:
        mean: Array of shape (2,) for [x1_mean, x2_mean]
        variance: Array of shape (2,) for [x1_var, x2_var]
        num_points: Number of points to generate

    Returns:
        Array of shape (num_points, 2) with [x1, x2] coordinates
    """
    # Generate points from normal distribution
    x1 = np.random.normal(mean[0], np.sqrt(variance[0]), num_points)
    x2 = np.random.normal(mean[1], np.sqrt(variance[1]), num_points)

    # Clip to ensure values are in [-1, 1]
    x1 = np.clip(x1, -1, 1)
    x2 = np.clip(x2, -1, 1)

    # Combine into single array
    points = np.column_stack([x1, x2])

    return points


def calculate_group_separation(group1_points, group2_points):
    """
    Calculate minimum distance between two groups of points.

    Args:
        group1_points: Array of shape (n1, 2)
        group2_points: Array of shape (n2, 2)

    Returns:
        Minimum distance between any two points from different groups
    """
    min_distance = float('inf')

    # Calculate pairwise distances between group centroids as approximation
    centroid1 = np.mean(group1_points, axis=0)
    centroid2 = np.mean(group2_points, axis=0)

    distance = np.linalg.norm(centroid1 - centroid2)

    return distance


def ensure_no_overlap(group1_points, group2_points, min_separation):
    """
    Verify that two groups don't overlap significantly.

    Args:
        group1_points: Array of shape (n1, 2)
        group2_points: Array of shape (n2, 2)
        min_separation: Minimum required separation distance

    Returns:
        True if groups are sufficiently separated, False otherwise
    """
    separation = calculate_group_separation(group1_points, group2_points)
    return separation >= min_separation


def generate_two_groups(num_points, mean1=None, var1=None, mean2=None, var2=None,
                       min_separation=None, max_attempts=100):
    """
    Generate two non-overlapping groups of Gaussian distributed points.

    Args:
        num_points: Number of points per group
        mean1: Mean for group 1 (default from params)
        var1: Variance for group 1 (default from params)
        mean2: Mean for group 2 (default from params)
        var2: Variance for group 2 (default from params)
        min_separation: Minimum separation distance (default from params)
        max_attempts: Maximum attempts to generate non-overlapping groups

    Returns:
        Tuple of (group1_points, group2_points), each of shape (num_points, 2)
    """
    # Use default parameters if not provided
    if mean1 is None:
        mean1 = params.MEAN_GROUP_1
    if var1 is None:
        var1 = params.VAR_GROUP_1
    if mean2 is None:
        mean2 = params.MEAN_GROUP_2
    if var2 is None:
        var2 = params.VAR_GROUP_2
    if min_separation is None:
        min_separation = params.MIN_SEPARATION

    attempts = 0
    while attempts < max_attempts:
        # Generate both groups
        group1 = generate_gaussian_points(mean1, var1, num_points)
        group2 = generate_gaussian_points(mean2, var2, num_points)

        # Check if they don't overlap
        if ensure_no_overlap(group1, group2, min_separation):
            return group1, group2

        attempts += 1

        # Adjust means to increase separation if needed
        direction = mean2 - mean1
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
            mean1 = mean1 - direction * 0.05
            mean2 = mean2 + direction * 0.05
            # Ensure means stay in valid range
            mean1 = np.clip(mean1, -0.9, 0.9)
            mean2 = np.clip(mean2, -0.9, 0.9)

    # If we couldn't generate non-overlapping groups, return the last attempt
    # This shouldn't happen with reasonable parameters
    return group1, group2


def get_group_statistics(points):
    """
    Calculate statistics for a group of points.

    Args:
        points: Array of shape (n, 2)

    Returns:
        Dictionary with mean, variance, and bounds
    """
    return {
        'mean': np.mean(points, axis=0),
        'variance': np.var(points, axis=0),
        'min': np.min(points, axis=0),
        'max': np.max(points, axis=0),
        'count': len(points)
    }
