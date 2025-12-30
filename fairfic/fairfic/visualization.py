import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def set_publication_style():
    """
    Set publication-quality plotting style.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
    })


def plot_fic_heatmap(fic_results, alphaF, metric='accuracy', ax=None, title=None):
    """
    Plot FIC heatmap for a specific alphaF value.
    
    Parameters
    ----------
    fic_results : dict
        Results from FairnessInformationCriterion.analyze_fairness()
    alphaF : float
        AlphaF value to plot.
    metric : str
        Metric name (for title).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Custom title.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if alphaF not in fic_results or not fic_results[alphaF]:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Extract groups and create matrix
    pairs = list(fic_results[alphaF].keys())
    all_groups = sorted(set(g for p in pairs for g in p.split(' - ')))
    n = len(all_groups)
    
    # Create matrix
    mat = np.full((n, n), np.nan)
    group_idx = {g: i for i, g in enumerate(all_groups)}
    
    for pair, data in fic_results[alphaF].items():
        g1, g2 = pair.split(' - ')
        i, j = group_idx[g1], group_idx[g2]
        mat[i, j] = mat[j, i] = data['fic_score']
    
    # Plot heatmap
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='equal')
    
    # Add value labels
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:.2f}',
                       ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if abs(mat[i,j]) > 0.5 else 'black')
    
    # Customize axes
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_groups, rotation=45, ha='right', fontsize=13, fontweight='bold')
    ax.set_yticklabels(all_groups, fontsize=13, fontweight='bold')
    
    if title is None:
        title = f'FIC Heatmap (αF = {alphaF}, Metric: {metric})'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    return ax, im


def plot_fic_heatmaps_grid(fic_results, metric='accuracy', figsize=(20, 16)):
    """
    Plot a grid of FIC heatmaps for all alphaF values.
    
    Parameters
    ----------
    fic_results : dict
        Results from FairnessInformationCriterion.analyze_fairness()
    metric : str
        Metric name.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    tuple
        (figure, axes, colorbar)
    """
    set_publication_style()
    
    alphaF_values = sorted(fic_results.keys())
    if not alphaF_values:
        raise ValueError("No FIC results provided")
    
    n_plots = len(alphaF_values)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each heatmap
    images = []
    for idx, alphaF in enumerate(alphaF_values):
        if idx < len(axes):
            ax = axes[idx]
            _, im = plot_fic_heatmap(fic_results, alphaF, metric, ax=ax)
            images.append(im)
    
    # Hide unused axes
    for idx in range(len(alphaF_values), len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    if images:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(images[0], cax=cbar_ax)
        cbar.set_label('FIC Score', fontsize=14, fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=12)
        
        # Add tier annotations
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
    
    fig.suptitle(f'FIC Heatmaps for Different αF Values ({metric})',
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    return fig, axes, cbar if images else None


def plot_benchmarking_tiers(fic_results, alphaF, metric='accuracy', figsize=(16, 8)):
    """
    Plot benchmarking tiers for a specific alphaF value.
    
    Parameters
    ----------
    fic_results : dict
        Results from FairnessInformationCriterion.analyze_fairness()
    alphaF : float
        AlphaF value to plot.
    metric : str
        Metric name.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    set_publication_style()
    
    if alphaF not in fic_results or not fic_results[alphaF]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for tiers
    colors = {
        'Optimum': '#2E8B57',      # Sea green
        'Acceptable': '#FFD700',   # Gold
        'Questionable': '#FF8C00', # Dark orange
        'Unacceptable': '#DC143C'  # Crimson
    }
    
    data = fic_results[alphaF]
    pairs = list(data.keys())
    fic_scores = [data[p]['fic_score'] for p in pairs]
    tiers = [data[p]['tier'] for p in pairs]
    
    # Create bars with tier colors
    bar_colors = [colors[t] for t in tiers]
    bars = ax.bar(range(len(pairs)), fic_scores, color=bar_colors, 
                  edgecolor='black', linewidth=1.2, width=0.6)
    
    # Add tier threshold lines
    ax.axhline(0.75, color='darkgreen', linestyle='--', linewidth=2.0, alpha=0.7)
    ax.axhline(0.50, color='goldenrod', linestyle='--', linewidth=2.0, alpha=0.7)
    ax.axhline(0.00, color='darkred', linestyle='--', linewidth=2.0, alpha=0.7)
    
    # Customize axes
    ax.set_xlabel('Group Pairs', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('FIC Score', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'FIC Benchmarking Tiers ({metric}, αF = {alphaF})',
                 fontsize=16, fontweight='bold', pad=15)
    
    # Set x-ticks
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=11, fontweight='bold')
    
    # Set y-axis limits
    y_min = min(fic_scores + [-0.1])
    y_max = max(fic_scores + [1.0])
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    
    # Format y-tick labels
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks], fontsize=11, fontweight='bold')
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create legends
    legend_elements = [
        Patch(facecolor=colors['Optimum'], edgecolor='black', label='Optimum (FIC > 0.75)'),
        Patch(facecolor=colors['Acceptable'], edgecolor='black', label='Acceptable (0.50 < FIC ≤ 0.75)'),
        Patch(facecolor=colors['Questionable'], edgecolor='black', label='Questionable (0 < FIC ≤ 0.50)'),
        Patch(facecolor=colors['Unacceptable'], edgecolor='black', label='Unacceptable (FIC ≤ 0)')
    ]
    
    line_legend_elements = [
        Line2D([0], [0], color='darkgreen', linestyle='--', linewidth=2, label='Optimum Threshold (0.75)'),
        Line2D([0], [0], color='goldenrod', linestyle='--', linewidth=2, label='Acceptable Threshold (0.50)'),
        Line2D([0], [0], color='darkred', linestyle='--', linewidth=2, label='Unacceptable Threshold (0.00)')
    ]
    
    # Add tier legend
    tier_legend = ax.legend(handles=legend_elements, fontsize=10, 
                            loc='upper left', bbox_to_anchor=(1.02, 1.0),
                            frameon=True, framealpha=0.9, edgecolor='black',
                            title='FIC Tiers', title_fontsize=11)
    tier_legend.get_title().set_fontweight('bold')
    
    # Add threshold legend
    ax.legend(handles=line_legend_elements, fontsize=9, 
              loc='upper left', bbox_to_anchor=(1.02, 0.65),
              frameon=True, framealpha=0.9, edgecolor='black',
              title='Thresholds', title_fontsize=10)
    
    # Add tier legend back (it was removed by second legend)
    ax.add_artist(tier_legend)
    
    # Add formula annotation
    annotation_text = f'αF = {alphaF}\nFIC = 1 - (ω/αF)\nω = |M₁ - M₂|'
    ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig