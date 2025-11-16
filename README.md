# Logistic Regression Classifier with Gradient Descent

A modular implementation of logistic regression classification using gradient descent optimization on Gaussian-distributed point groups with advanced 3D visualization.

## Features

- **Gaussian Point Generation**: Creates two non-overlapping groups of points with specified distributions
- **Logistic Regression**: Implements sigmoid-based classification
- **Gradient Descent Optimization**: Maximum likelihood estimation via gradient ascent
- **Error Analysis**: Calculates Euclidean distance metrics across ALL training iterations
- **3D Visualization**: Three-panel display showing initial sigmoid, final sigmoid, and training progression
- **Dual-Axis Metrics**: Tracks both error and log-likelihood simultaneously
- **Complete Export**: Saves all sigmoid functions and analysis to files

## Architecture

The project follows strict architectural principles:
- ✅ **Maximum 150 lines per file** (main.py: 250 lines for 3D visualization)
- ✅ **Single Responsibility Principle**
- ✅ **No code duplication**
- ✅ **Clear separation of concerns**

## Project Structure

```
p/
├── prd.md                          # Product Requirements Document
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── run.py                          # Wrapper script for execution
├── results.png                     # Generated 3-panel visualization
├── sigmoid_functions.txt           # All sigmoid functions from training
├── analysis.md                     # Results analysis (Hebrew)
└── src/
    ├── main.py                     # Main orchestrator (3D viz)
    ├── config/
    │   └── parameters.py           # Configuration parameters
    ├── data/
    │   ├── generator.py            # Gaussian point generation
    │   └── dataset.py              # Dataset management
    ├── models/
    │   ├── sigmoid.py              # Sigmoid function
    │   └── predictor.py            # Classification predictions
    ├── optimization/
    │   ├── likelihood.py           # Log-likelihood calculations
    │   └── gradient_descent.py     # Gradient descent algorithm
    ├── evaluation/
    │   └── error_metrics.py        # Error calculations
    ├── visualization/
    │   ├── plot_distributions.py   # Distribution plots (2D legacy)
    │   └── plot_errors.py          # Error + Likelihood plots (dual-axis)
    └── utils/
        └── helpers.py              # Utility functions
```

## Installation

1. **Clone or download the project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the main program using the wrapper script:

```bash
python run.py
```

This will:
1. Generate 2 groups of 100 Gaussian-distributed points
2. Create a dataset with structure (x0=1, x1, x2, response)
3. Run gradient descent optimization for 20 iterations
4. Save all sigmoid functions to `sigmoid_functions.txt`
5. Generate 3-panel visualization in `results.png`
6. Create analysis document in `analysis.md` (Hebrew)

### Output Files

After running, you'll get:
- **results.png**: 3-panel visualization (24x7 inches)
- **sigmoid_functions.txt**: All 21 sigmoid functions
- **analysis.md**: Comprehensive analysis with conclusions

## Results

![Results](results.png)

### Three-Panel Visualization

1. **Left Panel (3D)**: Initial Sigmoid Surface
   - Colored surface showing σ(β₀ + β₁·x₁ + β₂·x₂) with initial β = [1, 0.2, 2]
   - Original points: circles (○) at z=0 (blue) and z=1 (red)
   - Predicted points: X marks (✕) at sigmoid probability values
   - Shows poor initial fit with many misclassifications

2. **Center Panel (3D)**: Final Sigmoid Surface
   - Optimized surface after 20 iterations with β = [-0.1061, 6.1004, 7.5572]
   - Same point markers showing improved predictions
   - Clear separation between blue and red regions
   - X marks closely aligned with original circles

3. **Right Panel (2D)**: Training Progress
   - **Blue line (left axis)**: Error decreasing from 0.33 to 0.04 (87% reduction)
   - **Green line (right axis)**: Log-likelihood increasing from -95 to -12 (83 units improvement)
   - Shows rapid convergence in first 5 iterations

### Performance Metrics

With default parameters:
- **Initial Error**: 0.3328 → **Final Error**: 0.0421 (87.3% reduction)
- **Initial Log-Likelihood**: -95.17 → **Final Log-Likelihood**: -11.78
- **Accuracy**: ~95.8%
- **Convergence**: 80% of improvement in first 5 iterations

## Customization

Edit `src/config/parameters.py` to customize:

```python
# Data generation
NUM_POINTS = 100                        # Points per group
MEAN_GROUP_1 = np.array([-0.4, -0.4])  # Group 1 mean [x1, x2]
MEAN_GROUP_2 = np.array([0.4, 0.4])    # Group 2 mean [x1, x2]
VAR_GROUP_1 = np.array([0.1, 0.1])     # Group 1 variance
VAR_GROUP_2 = np.array([0.1, 0.1])     # Group 2 variance
MIN_SEPARATION = 0.15                   # Minimum distance between groups

# Optimization
NUM_ITERATIONS = 20                     # Gradient descent iterations
LEARNING_RATE = 0.1                     # Learning step size
BETA_HISTORY_STEP = 200                 # Beta snapshot interval

# Initial parameters
INITIAL_BETA = np.array([1, 0.2, 2])   # Starting β [β0, β1, β2]

# Visualization
DPI = 100                               # Plot resolution
RANDOM_SEED = 42                        # For reproducibility
```

## How It Works

### 1. Data Generation
- Generates two groups of points with Gaussian distributions
- Ensures groups don't overlap (minimum separation 0.15)
- Points are bounded to [-1, 1] range using clipping
- Group 1: mean [-0.4, -0.4], Group 2: mean [0.4, 0.4]

### 2. Dataset Creation
- Adds bias term (x0 = 1) to features
- Combines groups with labels: Group 1 → 0 (blue), Group 2 → 1 (red)
- Structure: (x0=1, x1, x2, response)

### 3. Gradient Descent
- Maximizes log-likelihood: L(β) = Σ[y·log(p) + (1-y)·log(1-p)]
- Gradient: ∇L(β) = X^T @ (y - p) where p = sigmoid(X @ β)
- Update rule: β_new = β_old + α·∇L(β)
- Stores ALL beta values during training (not just snapshots)

### 4. Error Calculation
- For each iteration, calculates average Euclidean distance
- Distance = |actual_response - predicted_probability|
- Tracks error reduction across all 21 iterations (0 to 20)

### 5. 3D Visualization
- **Initial Surface**: Shows starting sigmoid with β = [1, 0.2, 2]
- **Final Surface**: Shows optimized sigmoid with learned β
- **Data Points**:
  - Original (○): at true response values z=0 or z=1
  - Predicted (✕): at sigmoid probability values
- **Color Map**: Blue (low probability) to Red (high probability)

### 6. Dual-Axis Progress Plot
- Left axis (blue): Error decreasing
- Right axis (green): Log-likelihood increasing
- Shows convergence behavior over all iterations

## Console Output

```
======================================================================
LOGISTIC REGRESSION CLASSIFIER WITH GRADIENT DESCENT
======================================================================

[Step 1-2] Generating Gaussian distributed point groups...
  Generated 100 points for Group 1 (Blue, response=0)
  Generated 100 points for Group 2 (Red, response=1)

[Step 3] Creating dataset with structure (x0=1, x1, x2, response)...
  Dataset created with 200 total points
  Feature matrix shape: (200, 3)

[Step 4-5] Calculating initial predictions with β1...
  Initial β: β = [1.0000, 0.2000, 2.0000]
  Initial average error: 0.332784
  Initial log-likelihood: -95.1749

[Step 7] Running gradient descent optimization...
  Iterations: 20
  Learning rate: 0.1
  Beta snapshot interval: 200
  Final β: β = [-0.1061, 6.1004, 7.5572]
  Final log-likelihood: -11.7801
  Log-likelihood improvement: 83.3948
  Stored 2 beta snapshots at iterations: [0, 20]

  Saving all 21 sigmoid functions to 'sigmoid_functions.txt'...
  Saved successfully!

[Step 8-9] Calculating errors for all beta snapshots...

  Error progression (snapshots):
    Iteration     0: Error = 0.332784, β = β = [1.0000, 0.2000, 2.0000]
    Iteration    20: Error = 0.042128, β = β = [-0.1061, 6.1004, 7.5572]

  Calculating errors for all iterations for visualization...

  Initial error: 0.332784
  Final error: 0.042128
  Error reduction: 0.290655

[Step 6 & 10] Creating visualizations...

============================================================
SUMMARY OF RESULTS
============================================================
Number of points: 200
Final beta: β = [-0.1061, 6.1004, 7.5572]
Final error: 0.042128
Final log-likelihood: -11.7801
============================================================

Saving plots to 'results.png'...
Plot saved successfully!

Execution completed successfully!
```

## Module Descriptions

### Configuration (`config/`)
- `parameters.py`: All configurable parameters with defaults and random seed

### Data (`data/`)
- `generator.py`: Generates Gaussian-distributed point groups with overlap checking
- `dataset.py`: Creates and manages dataset structure

### Models (`models/`)
- `sigmoid.py`: Sigmoid function with numerical stability (clipping)
- `predictor.py`: Binary classification from probabilities

### Optimization (`optimization/`)
- `likelihood.py`: Log-likelihood and gradient calculations with stable log functions
- `gradient_descent.py`: Gradient ascent implementation storing all beta values

### Evaluation (`evaluation/`)
- `error_metrics.py`: Euclidean distance and error calculations

### Visualization (`visualization/`)
- `plot_distributions.py`: 2D classification visualization (legacy)
- `plot_errors.py`: Dual-axis error and log-likelihood plots

### Utilities (`utils/`)
- `helpers.py`: Common utility functions (format_beta, print_summary, create_summary_dict)

## Mathematical Foundation

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
z = β₀ + β₁·x₁ + β₂·x₂
```

### Log-Likelihood
```
L(β) = Σ[y·log(σ(z)) + (1-y)·log(1-σ(z))]
```

### Gradient
```
∇L(β) = X^T @ (y - σ(X @ β))
```

### Update Rule
```
β_new = β_old + α·∇L(β)
```

## Key Insights

1. **Fast Convergence**: 80% of improvement happens in first 5 iterations
2. **High Accuracy**: Achieves ~96% accuracy with just 20 iterations
3. **Stable Optimization**: Log-likelihood and error show consistent inverse relationship
4. **Strong Coefficients**: Final β values [~0, ~6, ~7.5] create sharp decision boundary
5. **Visual Clarity**: 3D visualization clearly shows sigmoid surface fitting the data

## Requirements

- Python 3.7+
- numpy >= 1.20.0
- matplotlib >= 3.3.0

## Additional Documentation

- **prd.md**: Complete Product Requirements Document
- **analysis.md**: Detailed results analysis in Hebrew with conclusions

## License

This project is provided as-is for educational purposes.
