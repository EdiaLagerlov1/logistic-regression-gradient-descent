# Product Requirements Document
## Logistic Regression Classifier with Gradient Descent

---

## Overview
Build a logistic regression classifier that generates two Gaussian-distributed point groups, trains using gradient descent optimization, and visualizes both original and predicted classifications.

---

## Architecture Principles
- **Maximum file length**: 150 lines per file (strict)
- **Single Responsibility**: Each file/class/function does ONE thing
- **No duplicate code**: Extract common logic into reusable functions
- **Separation of Concerns**: Separate data, business logic, and presentation

---

## Project Structure

```
src/
├── config/
│   └── parameters.py          # All configurable parameters (< 150 lines)
├── data/
│   ├── generator.py           # Gaussian point generation (< 150 lines)
│   └── dataset.py             # Dataset structure and creation (< 150 lines)
├── models/
│   ├── sigmoid.py             # Sigmoid function logic (< 150 lines)
│   └── predictor.py           # Response prediction (< 150 lines)
├── optimization/
│   ├── gradient_descent.py    # Gradient descent algorithm (< 150 lines)
│   └── likelihood.py          # Log-likelihood calculation (< 150 lines)
├── evaluation/
│   └── error_metrics.py       # Error calculation functions (< 150 lines)
├── visualization/
│   ├── plot_distributions.py  # Distribution plotting (< 150 lines)
│   └── plot_errors.py         # Error + Likelihood plotting (< 150 lines)
├── utils/
│   └── helpers.py             # Common utility functions (< 150 lines)
└── main.py                    # Main orchestrator (< 250 lines - expanded for 3D viz)

Output Files:
├── results.png                # 3-panel visualization
├── sigmoid_functions.txt      # All sigmoid functions from training
└── analysis.md                # Results analysis in Hebrew
```

---

## Module Specifications

### 1. Configuration Module (`config/parameters.py`)
**Responsibility**: Store all configurable parameters
**Exports**:
- `NUM_POINTS`: Number of points per group (default: 100)
- `NUM_ITERATIONS`: Gradient descent iterations (default: 20)
- `LEARNING_RATE`: Learning step size (default: 0.1)
- `MEAN_GROUP_1`: Gaussian mean for group 1 (default: [-0.4, -0.4])
- `MEAN_GROUP_2`: Gaussian mean for group 2 (default: [0.4, 0.4])
- `VAR_GROUP_1`, `VAR_GROUP_2`: Gaussian variances (default: [0.1, 0.1])
- `INITIAL_BETA`: Initial beta vector [β0, β1, β2] (default: [1, 0.2, 2])
- `BETA_HISTORY_STEP`: Snapshot interval (default: 200)
- `MIN_SEPARATION`: Minimum distance between groups (default: 0.15)
- `RANDOM_SEED`: Reproducibility seed (default: 42)

**Constraints**: < 150 lines

---

### 2. Data Generation Module (`data/generator.py`)
**Responsibility**: Generate non-overlapping Gaussian distributed points

**Functions**:
- `generate_gaussian_points(mean, variance, num_points)` → Returns (x1, x2) arrays
  - Generate points from 2D Gaussian distribution
  - Ensure x1, x2 ∈ [-1, 1] using clipping

- `calculate_group_separation(group1_points, group2_points)` → Float
  - Calculate distance between group centroids

- `ensure_no_overlap(group1_points, group2_points, min_separation)` → Boolean
  - Verify groups don't overlap
  - Return True if separation >= min_separation

- `generate_two_groups(num_points, mean1, var1, mean2, var2, min_separation)` → Returns (group1, group2)
  - Generate both groups with non-overlapping constraint
  - Retry with adjusted means if overlap detected (max 100 attempts)

- `get_group_statistics(points)` → Dict
  - Calculate mean, variance, min, max for a group

**Constraints**: < 150 lines, pure data generation logic

---

### 3. Dataset Module (`data/dataset.py`)
**Responsibility**: Create and manage dataset structure

**Functions**:
- `create_dataset(group1_points, group2_points)` → DataFrame/Array
  - Add x0 = 1 column (bias term)
  - Combine groups: blue (response=0), red (response=1)
  - Structure: [x0, x1, x2, response]

- `get_features(dataset)` → Returns X matrix [x0, x1, x2]
- `get_responses(dataset)` → Returns y vector [response]

**Constraints**: < 150 lines, data structure only

---

### 4. Sigmoid Module (`models/sigmoid.py`)
**Responsibility**: Implement sigmoid function

**Functions**:
- `sigmoid(z)` → Returns 1 / (1 + e^(-z))
  - Vectorized implementation
  - Handle numerical stability (clip extreme values)

- `predict_probability(X, beta)` → Returns sigmoid(X @ beta)
  - Calculate probabilities for all points
  - X: feature matrix, beta: parameter vector

**Constraints**: < 150 lines, pure mathematical functions

---

### 5. Predictor Module (`models/predictor.py`)
**Responsibility**: Generate predictions from sigmoid

**Functions**:
- `predict_class(probabilities, threshold=0.5)` → Returns binary classifications
  - Convert probabilities to binary (0 or 1)

- `predict_response(X, beta)` → Returns predicted responses
  - Combines sigmoid and classification

**Constraints**: < 150 lines

---

### 6. Log-Likelihood Module (`optimization/likelihood.py`)
**Responsibility**: Calculate log-likelihood for optimization

**Functions**:
- `log_likelihood(X, y, beta)` → Returns scalar
  - Calculate: Σ[y * log(p) + (1-y) * log(1-p)]
  - Where p = sigmoid(X @ beta)

- `log_likelihood_gradient(X, y, beta)` → Returns gradient vector
  - Calculate: X^T @ (y - sigmoid(X @ beta))

**Constraints**: < 150 lines, pure calculation

---

### 7. Gradient Descent Module (`optimization/gradient_descent.py`)
**Responsibility**: Optimize beta using gradient descent

**Functions**:
- `gradient_descent(X, y, initial_beta, learning_rate, num_iterations)` → Returns beta_history
  - Iterate: beta_new = beta_old + learning_rate * gradient
  - Store intermediate betas every N iterations
  - Return: {iteration: beta} dictionary

- `store_beta_snapshots(beta_history, step)` → Returns selected betas
  - Keep 5β, 4β, 3β, 2β, 1β (every step iterations)

**Constraints**: < 150 lines, optimization logic only

---

### 8. Error Metrics Module (`evaluation/error_metrics.py`)
**Responsibility**: Calculate prediction errors

**Functions**:
- `euclidean_distance(actual, predicted)` → Returns distances
  - Calculate |actual - predicted| for each point

- `average_error(actual, predicted)` → Returns mean Euclidean distance
  - Mean of all distances

- `calculate_errors_for_betas(X, y, beta_dict)` → Returns {beta_id: error}
  - For each beta in history, calculate average error

**Constraints**: < 150 lines, metric calculations only

---

### 9. Distribution Plot Module (`visualization/plot_distributions.py`)
**Responsibility**: Visualize point classifications (legacy 2D support)

**Functions**:
- `plot_original_groups(ax, dataset)` → Plots on 2D axis
  - Blue circles (response=0)
  - Red X marks (response=1)

- `plot_predicted_groups(ax, X, beta)` → Plots on 2D axis
  - Light blue circles (predicted=0)
  - Light red X marks (predicted=1)
  - Use alpha transparency

- `create_combined_distribution_plot(dataset, beta)` → Returns figure
  - Combine original and predicted groups

- `add_decision_boundary(ax, beta, x_range)` → Adds line to plot
  - Draw decision boundary where sigmoid = 0.5

**Note**: Main visualization now in main.py using 3D plots

**Constraints**: < 150 lines, visualization only

---

### 10. Error Plot Module (`visualization/plot_errors.py`)
**Responsibility**: Visualize error and likelihood progression

**Functions**:
- `plot_error_vs_iterations(error_dict, ax, likelihood_history=None)` → Plots on axis
  - Left Y-axis (blue): average error
  - Right Y-axis (green): log-likelihood (if provided)
  - X-axis: iteration number
  - Dual-axis plot showing both metrics
  - Includes initial and final markers

**Constraints**: < 150 lines

---

### 11. Utility Module (`utils/helpers.py`)
**Responsibility**: Common reusable functions

**Functions**:
- `format_beta(beta)` → String
  - Pretty print beta vector as "β = [β₀, β₁, β₂]"

- `print_summary(summary_dict)` → None
  - Print formatted summary statistics

- `create_summary_dict(dataset, final_beta, final_error, final_likelihood)` → Dict
  - Create summary dictionary with key metrics

**Constraints**: < 150 lines

---

### 12. Main Orchestrator (`main.py`)
**Responsibility**: Coordinate entire workflow

**Flow**:
1. Load parameters from config
2. Generate two Gaussian groups (data/generator.py)
3. Create dataset with x0=1 (data/dataset.py)
4. Calculate initial predictions and metrics (models/)
5. Run gradient descent optimization (optimization/)
6. Store beta snapshots at intervals
7. **Save all sigmoid functions to sigmoid_functions.txt**
8. Calculate errors for snapshots and ALL iterations (evaluation/)
9. Create 3-panel visualization:
   - **Subplot 1 (3D)**: Initial sigmoid surface with data points
     - Orange/blue/red surface showing σ(β₀ + β₁·x₁ + β₂·x₂)
     - Original: circles at z=0 (blue) and z=1 (red)
     - Predicted: X marks at sigmoid probabilities
   - **Subplot 2 (3D)**: Final sigmoid surface with data points
     - Same structure but with optimized β
   - **Subplot 3 (2D)**: Dual-axis error and log-likelihood plot
     - Blue line: error decreasing
     - Green line: log-likelihood increasing
10. Save results.png
11. Print summary statistics

**Constraints**: < 250 lines (expanded for 3D visualization code)

---

## Data Flow

```
Config → Generator → Dataset → [Training Loop] → Evaluation → Visualization
                                     ↓                ↓            ↓
                              Sigmoid ← Beta      Error Calc   3D Plots
                                     ↓                ↓            ↓
                              Gradient Descent   Likelihood   Dual-Axis
                                     ↓
                              Beta History → sigmoid_functions.txt
                                     ↓
                              Final Results → analysis.md
```

---

## Additional Files

### run.py
**Purpose**: Wrapper script to handle Python import paths
- Adds `src/` to Python path
- Imports and runs main.py
- Ensures relative imports work correctly

### requirements.txt
**Dependencies**:
```
numpy>=1.20.0
matplotlib>=3.3.0
```

---

## Key Requirements

### Task 1-3: Data Generation
- Generate 2 groups of 100 points each
- Gaussian distribution with specified mean/variance
  - Group 1: mean=[-0.4, -0.4], var=[0.1, 0.1]
  - Group 2: mean=[0.4, 0.4], var=[0.1, 0.1]
- Ensure minimum separation (0.15) between groups
- Points in range [-1, 1]
- Create dataset: (x0=1, x1, x2, response)

### Task 4-5: Initial Prediction
- Calculate sigmoid with initial β = [1, 0.2, 2]
- Generate predictions for all points
- Calculate initial error and log-likelihood

### Task 6: Visualization (Enhanced to 3D)
- **3D Plot 1**: Initial sigmoid surface
  - Original: circles (○) at z=0/z=1
  - Predicted: X marks (✕) at sigmoid probabilities
  - Surface: σ(β₀ + β₁·x₁ + β₂·x₂)
- **3D Plot 2**: Final sigmoid surface
  - Same structure with optimized β

### Task 7: Optimization
- Run gradient descent for max log-likelihood
- 20 iterations, learning rate 0.1
- Store ALL beta values during training
- Save beta snapshots at specified intervals
- **Export all sigmoid functions to txt file**

### Task 8-9: Error Analysis
- Calculate errors for ALL iterations (not just snapshots)
- Compute average Euclidean distance for each β
- Plot error AND log-likelihood vs iteration on dual-axis

### Task 10: Combined Display
- **3 graphs on same page (1x3 layout)**
  - Left: Initial sigmoid (3D)
  - Center: Final sigmoid (3D)
  - Right: Error + Likelihood (2D dual-axis)
- Save as results.png
- **Generate analysis.md with Hebrew commentary**

---

## Actual Results

With default parameters (20 iterations, learning rate 0.1):
- **Initial Error**: 0.3328 (33.3% average distance)
- **Final Error**: 0.0421 (4.2% average distance)
- **Error Reduction**: 87.3%
- **Initial Log-Likelihood**: -95.17 (poor fit)
- **Final Log-Likelihood**: -11.78 (good fit)
- **Improvement**: 83.4 units
- **Final β**: [-0.1061, 6.1004, 7.5572]
- **Convergence**: Most improvement in first 5 iterations

**Key Insight**: The model achieves ~95.8% accuracy in just 20 iterations, demonstrating fast convergence.

---

## Testing Strategy

Each module should have:
1. Unit tests for individual functions
2. Integration tests for module interactions
3. Validation of line count constraints

---

## Dependencies
- `numpy>=1.20.0`: Numerical computations and array operations
- `matplotlib>=3.3.0`: 2D and 3D visualization
  - `mpl_toolkits.mplot3d`: 3D plotting support

---

## Output Files

### 1. results.png
- 3-panel visualization (24x7 inches, 100 DPI)
- Panel 1: Initial sigmoid surface (3D)
- Panel 2: Final sigmoid surface (3D)
- Panel 3: Error and log-likelihood progression (2D)

### 2. sigmoid_functions.txt
- All 21 sigmoid functions from training
- Format: `σ(β₀ + β₁·x₁ + β₂·x₂)` for each iteration
- Includes beta coefficients

### 3. analysis.md
- Comprehensive results analysis in Hebrew
- Embedded visualization
- Mathematical formulas
- Conclusions and recommendations

---

## Success Criteria
1. ✅ All modules < 150 lines (main.py < 250 lines)
2. ✅ No duplicate code
3. ✅ Clear separation of concerns
4. ✅ Single responsibility per module
5. ✅ Successful classification with 3D visualization
6. ✅ Error reduction through gradient descent
7. ✅ Three plots display on same page
8. ✅ Dual-axis plot showing error AND likelihood
9. ✅ Export all sigmoid functions to file
10. ✅ Generate Hebrew analysis document
11. ✅ Reproducible results (random seed)
