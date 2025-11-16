"""
Main orchestrator for logistic regression classifier.
Coordinates the entire workflow from data generation to visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import configuration
from config import parameters as params

# Import data modules
from data.generator import generate_two_groups
from data.dataset import create_dataset, get_features, get_responses

# Import model modules
from models.sigmoid import predict_probability
from models.predictor import predict_response

# Import optimization modules
from optimization.gradient_descent import gradient_descent, store_beta_snapshots
from optimization.likelihood import log_likelihood

# Import evaluation modules
from evaluation.error_metrics import calculate_error_for_beta, calculate_errors_for_betas

# Import visualization modules
from visualization.plot_distributions import add_decision_boundary
from visualization.plot_errors import plot_error_vs_iterations

# Import utilities
from utils.helpers import format_beta, print_summary, create_summary_dict


def main():
    """
    Main execution function.
    Implements the complete workflow for logistic regression classification.
    """
    print("="*70)
    print("LOGISTIC REGRESSION CLASSIFIER WITH GRADIENT DESCENT")
    print("="*70)

    # Step 1-2: Generate two groups of points
    print("\n[Step 1-2] Generating Gaussian distributed point groups...")
    group1, group2 = generate_two_groups(params.NUM_POINTS)
    print(f"  Generated {params.NUM_POINTS} points for Group 1 (Blue, response=0)")
    print(f"  Generated {params.NUM_POINTS} points for Group 2 (Red, response=1)")

    # Step 3: Create dataset
    print("\n[Step 3] Creating dataset with structure (x0=1, x1, x2, response)...")
    dataset = create_dataset(group1, group2)
    X = get_features(dataset)
    y = get_responses(dataset)
    print(f"  Dataset created with {len(y)} total points")
    print(f"  Feature matrix shape: {X.shape}")

    # Step 4-5: Calculate initial predictions with beta1
    print("\n[Step 4-5] Calculating initial predictions with β1...")
    beta1 = params.INITIAL_BETA.copy()
    print(f"  Initial β: {format_beta(beta1)}")
    initial_probabilities = predict_probability(X, beta1)
    initial_predictions = predict_response(X, beta1)
    initial_error = calculate_error_for_beta(X, y, beta1)
    initial_likelihood = log_likelihood(X, y, beta1)
    print(f"  Initial average error: {initial_error:.6f}")
    print(f"  Initial log-likelihood: {initial_likelihood:.4f}")

    # Step 7: Run gradient descent
    print("\n[Step 7] Running gradient descent optimization...")
    print(f"  Iterations: {params.NUM_ITERATIONS}")
    print(f"  Learning rate: {params.LEARNING_RATE}")
    print(f"  Beta snapshot interval: {params.BETA_HISTORY_STEP}")

    gd_results = gradient_descent(X, y, beta1, params.LEARNING_RATE, params.NUM_ITERATIONS)
    final_beta = gd_results['final_beta']
    beta_history = gd_results['beta_history']
    likelihood_history = gd_results['likelihood_history']

    print(f"  Final β: {format_beta(final_beta)}")
    print(f"  Final log-likelihood: {likelihood_history[-1]:.4f}")
    print(f"  Log-likelihood improvement: {likelihood_history[-1] - likelihood_history[0]:.4f}")

    # Store beta snapshots
    beta_snapshots = store_beta_snapshots(beta_history, params.BETA_HISTORY_STEP)
    print(f"  Stored {len(beta_snapshots)} beta snapshots at iterations: {sorted(beta_snapshots.keys())}")

    # Save all sigmoid functions to file
    print(f"\n  Saving all {len(beta_history)} sigmoid functions to 'sigmoid_functions.txt'...")
    with open('sigmoid_functions.txt', 'w') as f:
        f.write("ALL SIGMOID FUNCTIONS FOUND DURING GRADIENT DESCENT\n")
        f.write("="*70 + "\n\n")
        f.write("Sigmoid function form: σ(β₀ + β₁·x₁ + β₂·x₂)\n\n")
        for iteration in sorted(beta_history.keys()):
            beta = beta_history[iteration]
            f.write(f"Iteration {iteration:5d}: β = [{beta[0]:10.6f}, {beta[1]:10.6f}, {beta[2]:10.6f}]\n")
            f.write(f"                σ({beta[0]:.6f} + {beta[1]:.6f}·x₁ + {beta[2]:.6f}·x₂)\n\n")
    print(f"  Saved successfully!")

    # Step 8-9: Calculate errors for all beta snapshots
    print("\n[Step 8-9] Calculating errors for all beta snapshots...")
    error_dict_snapshots = calculate_errors_for_betas(X, y, beta_snapshots)

    print("\n  Error progression (snapshots):")
    for iteration in sorted(error_dict_snapshots.keys()):
        beta = beta_snapshots[iteration]
        error = error_dict_snapshots[iteration]
        print(f"    Iteration {iteration:5d}: Error = {error:.6f}, β = {format_beta(beta)}")

    # Calculate errors for ALL iterations for plotting
    print("\n  Calculating errors for all iterations for visualization...")
    error_dict_all = calculate_errors_for_betas(X, y, beta_history)

    # Calculate final metrics
    final_error = calculate_error_for_beta(X, y, final_beta)
    print(f"\n  Initial error: {initial_error:.6f}")
    print(f"  Final error: {final_error:.6f}")
    print(f"  Error reduction: {initial_error - final_error:.6f}")

    # Step 6 & 10: Create visualizations
    print("\n[Step 6 & 10] Creating visualizations...")

    # Create figure with three subplots: two 3D plots and one 2D plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(24, 7), dpi=params.DPI)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3)

    # Subplot 1: 3D Sigmoid Surface Visualization
    import numpy as np

    # Create meshgrid for sigmoid surface
    x1_range = np.linspace(-1, 1, 50)
    x2_range = np.linspace(-1, 1, 50)
    X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
    X_mesh = np.c_[np.ones(X1_mesh.ravel().shape), X1_mesh.ravel(), X2_mesh.ravel()]

    # Calculate initial and final sigmoid surfaces
    beta_initial = beta_history[0]
    Z_initial = predict_probability(X_mesh, beta_initial).reshape(X1_mesh.shape)
    Z_final = predict_probability(X_mesh, final_beta).reshape(X1_mesh.shape)

    # Plot initial sigmoid surface on ax1
    surf = ax1.plot_surface(X1_mesh, X2_mesh, Z_initial, cmap='RdBu_r',
                           alpha=0.6, edgecolor='none', vmin=0, vmax=1)

    # Plot data points in 3D
    X = dataset['X']
    y = dataset['y']

    # ORIGINAL DISTRIBUTION (circles at true response values z=0 or z=1)
    mask_group1 = (y == 0)
    mask_group2 = (y == 1)

    ax1.scatter(X[mask_group1, 1], X[mask_group1, 2], y[mask_group1],
               c='blue', marker='o', s=60, alpha=0.3, edgecolors='darkblue', linewidth=1.5,
               label='Original Group 1 (○)', depthshade=True)

    ax1.scatter(X[mask_group2, 1], X[mask_group2, 2], y[mask_group2],
               c='red', marker='o', s=60, alpha=0.3, edgecolors='darkred', linewidth=1.5,
               label='Original Group 2 (○)', depthshade=True)

    # PREDICTED DISTRIBUTION according to INITIAL sigmoid (X marks at predicted probabilities)
    # Calculate sigmoid predictions for all points using initial beta
    predictions_initial = predict_response(X, beta_initial)
    probabilities_initial = predict_probability(X, beta_initial)

    # Predicted as Group 1 (blue X)
    pred_mask_group1_initial = (predictions_initial == 0)
    ax1.scatter(X[pred_mask_group1_initial, 1], X[pred_mask_group1_initial, 2], probabilities_initial[pred_mask_group1_initial],
               c='blue', marker='x', s=80, alpha=0.9, linewidth=2.5,
               label='Predicted Group 1 (✕)', depthshade=True)

    # Predicted as Group 2 (red X)
    pred_mask_group2_initial = (predictions_initial == 1)
    ax1.scatter(X[pred_mask_group2_initial, 1], X[pred_mask_group2_initial, 2], probabilities_initial[pred_mask_group2_initial],
               c='red', marker='x', s=80, alpha=0.9, linewidth=2.5,
               label='Predicted Group 2 (✕)', depthshade=True)

    # Set labels and title
    ax1.set_xlabel('x1', fontsize=11, labelpad=10)
    ax1.set_ylabel('x2', fontsize=11, labelpad=10)
    ax1.set_zlabel('Probability (Sigmoid Output)', fontsize=11, labelpad=10)
    ax1.set_title('Initial Sigmoid Surface (Iteration 0)', fontsize=13, fontweight='bold', pad=20)

    # Set axis limits
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(0, 1)

    # Add legend
    ax1.legend(loc='upper left', fontsize=8)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Sigmoid Probability', fontsize=10)

    # Set viewing angle
    ax1.view_init(elev=20, azim=225)

    # Subplot 2: 3D Sigmoid with Final Only
    # Plot final sigmoid surface only
    surf2 = ax2.plot_surface(X1_mesh, X2_mesh, Z_final, cmap='RdBu_r',
                           alpha=0.6, edgecolor='none', vmin=0, vmax=1)

    # Plot data points in 3D
    ax2.scatter(X[mask_group1, 1], X[mask_group1, 2], y[mask_group1],
               c='blue', marker='o', s=60, alpha=0.3, edgecolors='darkblue', linewidth=1.5,
               label='Original Group 1 (○)', depthshade=True)

    ax2.scatter(X[mask_group2, 1], X[mask_group2, 2], y[mask_group2],
               c='red', marker='o', s=60, alpha=0.3, edgecolors='darkred', linewidth=1.5,
               label='Original Group 2 (○)', depthshade=True)

    # PREDICTED DISTRIBUTION according to FINAL sigmoid (X marks at predicted probabilities)
    # Calculate sigmoid predictions for all points using final beta
    predictions_final = predict_response(X, final_beta)
    probabilities_final = predict_probability(X, final_beta)

    # Predicted as Group 1 (blue X)
    pred_mask_group1_final = (predictions_final == 0)
    ax2.scatter(X[pred_mask_group1_final, 1], X[pred_mask_group1_final, 2], probabilities_final[pred_mask_group1_final],
               c='blue', marker='x', s=80, alpha=0.9, linewidth=2.5,
               label='Predicted Group 1 (✕)', depthshade=True)

    # Predicted as Group 2 (red X)
    pred_mask_group2_final = (predictions_final == 1)
    ax2.scatter(X[pred_mask_group2_final, 1], X[pred_mask_group2_final, 2], probabilities_final[pred_mask_group2_final],
               c='red', marker='x', s=80, alpha=0.9, linewidth=2.5,
               label='Predicted Group 2 (✕)', depthshade=True)

    # Set labels and title
    ax2.set_xlabel('x1', fontsize=11, labelpad=10)
    ax2.set_ylabel('x2', fontsize=11, labelpad=10)
    ax2.set_zlabel('Probability (Sigmoid Output)', fontsize=11, labelpad=10)
    ax2.set_title(f'Final Sigmoid Surface (Iteration {params.NUM_ITERATIONS})', fontsize=13, fontweight='bold', pad=20)

    # Set axis limits
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(0, 1)

    # Add legend
    ax2.legend(loc='upper left', fontsize=8)

    # Add colorbar
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
    cbar2.set_label('Sigmoid Probability', fontsize=10)

    # Set viewing angle
    ax2.view_init(elev=20, azim=225)

    # Subplot 3: Error and Likelihood plot
    # Convert likelihood_history list to dict for ALL iterations
    likelihood_dict = {i: likelihood_history[i] for i in error_dict_all.keys() if i < len(likelihood_history)}
    plot_error_vs_iterations(error_dict_all, ax3, likelihood_dict)

    # Adjust layout
    plt.tight_layout()

    # Print summary
    summary = create_summary_dict(dataset, final_beta, final_error, likelihood_history[-1])
    print_summary(summary)

    # Save and show plots
    print("Saving plots to 'results.png'...")
    plt.savefig('results.png', dpi=150, bbox_inches='tight')
    print("Plot saved successfully!")

    # Uncomment the line below to display plots interactively
    # plt.show()

    print("\nExecution completed successfully!")


if __name__ == "__main__":
    main()
