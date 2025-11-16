"""
Configuration module for all parameters.
Stores all configurable parameters with default values.
"""

import numpy as np

# Data generation parameters
NUM_POINTS = 100  # Number of points per group

# Gaussian distribution parameters for Group 1 (Blue, response=0)
# Mean for x1 and x2 coordinates
MEAN_GROUP_1 = np.array([-0.4, -0.4])
# Variance for x1 and x2
VAR_GROUP_1 = np.array([0.1, 0.1])

# Gaussian distribution parameters for Group 2 (Red, response=1)
MEAN_GROUP_2 = np.array([0.4, 0.4])
VAR_GROUP_2 = np.array([0.1, 0.1])

# Minimum separation distance between groups to avoid overlap
MIN_SEPARATION = 0.15

# Model parameters
INITIAL_BETA = np.array([1, 0.2, 2])  # [β0, β1, β2]

# Optimization parameters
NUM_ITERATIONS = 20  # Number of gradient descent iterations
LEARNING_RATE = 0.1  # Learning step size
BETA_HISTORY_STEP = 200  # Store beta every N iterations (5β, 4β, 3β, 2β, β)

# Prediction parameters
CLASSIFICATION_THRESHOLD = 0.5  # Threshold for binary classification

# Visualization parameters
FIGURE_SIZE = (14, 6)  # Size of the combined plot
DPI = 100  # Resolution

# Color scheme
COLOR_GROUP_1_ORIGINAL = 'blue'
COLOR_GROUP_2_ORIGINAL = 'red'
COLOR_GROUP_1_PREDICTED = 'lightblue'
COLOR_GROUP_2_PREDICTED = 'lightcoral'
ALPHA_PREDICTED = 0.5  # Transparency for predicted points
MARKER_SIZE = 50  # Size of scatter plot markers

# Numerical stability
EPSILON = 1e-15  # Small value to prevent log(0)
CLIP_VALUE = 500  # Clip values in sigmoid to prevent overflow

# Random seed for reproducibility (set to None for random behavior)
RANDOM_SEED = 42

# Set random seed if specified
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)
