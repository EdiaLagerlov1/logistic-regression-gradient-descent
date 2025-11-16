"""
Distribution plotting module.
Visualizes original and predicted point classifications.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from ..models.predictor import split_predictions_by_class
    from ..config import parameters as params
except ImportError:
    from models.predictor import split_predictions_by_class
    from config import parameters as params


def plot_original_groups(ax, dataset):
    """Plot original groups with their true labels."""
    X, y = dataset['X'], dataset['y']
    x1, x2 = X[:, 1], X[:, 2]
    mask_group1, mask_group2 = (y == 0), (y == 1)
    ax.scatter(x1[mask_group1], x2[mask_group1], c=params.COLOR_GROUP_1_ORIGINAL,
              marker='o', s=params.MARKER_SIZE, label='Group 1 (Original)', alpha=0.7,
              edgecolors='black', linewidth=0.5)
    ax.scatter(x1[mask_group2], x2[mask_group2], c=params.COLOR_GROUP_2_ORIGINAL,
              marker='x', s=params.MARKER_SIZE*1.5, label='Group 2 (Original)', alpha=0.9,
              linewidth=2)


def plot_predicted_groups(ax, X, beta):
    """Plot predicted groups based on sigmoid classification."""
    class0_indices, class1_indices = split_predictions_by_class(X, beta)
    x1, x2 = X[:, 1], X[:, 2]
    ax.scatter(x1[class0_indices], x2[class0_indices],
              c=params.COLOR_GROUP_1_PREDICTED, marker='o', s=params.MARKER_SIZE,
              label='Group 1 (Predicted)', alpha=params.ALPHA_PREDICTED,
              edgecolors='blue', linewidth=1.5)
    ax.scatter(x1[class1_indices], x2[class1_indices],
              c=params.COLOR_GROUP_2_PREDICTED, marker='x', s=params.MARKER_SIZE*1.5,
              label='Group 2 (Predicted)', alpha=params.ALPHA_PREDICTED,
              linewidth=2)


def create_combined_distribution_plot(dataset, beta):
    """Create plot with original and predicted groups overlaid."""
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_original_groups(ax, dataset)
    plot_predicted_groups(ax, dataset['X'], beta)
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Original (Solid) vs Predicted (Transparent)', fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal')
    return ax


def add_decision_boundary(ax, beta, x_range=None):
    """Add decision boundary line to plot."""
    if x_range is None:
        x_range = np.linspace(0, 1, 100)
    if np.abs(beta[2]) > 1e-10:
        x2_boundary = -(beta[0] + beta[1] * x_range) / beta[2]
        valid_mask = (x2_boundary >= 0) & (x2_boundary <= 1)
        ax.plot(x_range[valid_mask], x2_boundary[valid_mask],
               'k--', linewidth=2, label='Decision Boundary')
