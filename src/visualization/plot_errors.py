"""
Error plotting module.
Visualizes error metrics over iterations.
"""

import matplotlib.pyplot as plt


def plot_error_vs_iterations(error_dict, ax, likelihood_history=None):
    """Plot average error and log-likelihood vs iteration number."""
    iterations = sorted(error_dict.keys())
    errors = [error_dict[i] for i in iterations]

    # Plot error on left y-axis
    ax.plot(iterations, errors, 'b-o', linewidth=2, markersize=8, label='Average Error')
    if len(iterations) > 0:
        ax.plot(iterations[0], errors[0], 'go', markersize=12,
               label=f'Initial Error: {errors[0]:.4f}')
        ax.plot(iterations[-1], errors[-1], 'ro', markersize=12,
               label=f'Final Error: {errors[-1]:.4f}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Average Euclidean Distance', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('Error and Log-Likelihood vs Iteration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if len(iterations) > 0:
        ax.set_xlim(left=-max(iterations)*0.05)

    # Plot likelihood on right y-axis if provided
    if likelihood_history is not None:
        ax2 = ax.twinx()
        likelihood_iterations = sorted(likelihood_history.keys())
        likelihoods = [likelihood_history[i] for i in likelihood_iterations]
        ax2.plot(likelihood_iterations, likelihoods, 'g-s', linewidth=2, markersize=6,
                label='Log-Likelihood', alpha=0.7)
        ax2.set_ylabel('Log-Likelihood', fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    else:
        ax.legend(loc='best', fontsize=10)
