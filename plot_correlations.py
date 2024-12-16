#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_parameter_correlations(cells_df, output_dir, sensor_type):
    """Plot correlations between cell parameters"""
    # Create correlation matrix
    params = ['area', 'perimeter', 'eccentricity', 'green', 'blue', 'ratio']
    corr_matrix = cells_df[params].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'{sensor_type} Parameter Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sensor_type.lower()}_correlations.png')
    plt.close()
    
    # Plot individual correlations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{sensor_type} Parameter Relationships')
    
    # Plot each parameter vs ratio
    for i, param in enumerate(['area', 'perimeter', 'eccentricity', 'green', 'blue']):
        row = i // 3
        col = i % 3
        sns.scatterplot(data=cells_df, x=param, y='ratio', ax=axes[row, col], alpha=0.5)
        axes[row, col].set_title(f'{param.capitalize()} vs Ratio')
        
        # Add correlation coefficient
        corr = cells_df[param].corr(cells_df['ratio'])
        axes[row, col].text(0.05, 0.95, f'r = {corr:.2f}',
                          transform=axes[row, col].transAxes,
                          verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{sensor_type.lower()}_parameter_relationships.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot NAD sensor correlations')
    parser.add_argument('results_dir', help='Directory containing results')
    
    args = parser.parse_args()
    output_dir = Path(args.results_dir)
    
    # Load data
    nad_cells = pd.read_csv(output_dir / "nad_cells.csv")
    venus_cells = pd.read_csv(output_dir / "venus_cells.csv")
    
    # Plot correlations for both sensors
    plot_parameter_correlations(nad_cells, output_dir, 'NAD')
    plot_parameter_correlations(venus_cells, output_dir, 'Venus')
    
    # Print correlation summaries
    print("\nNAD Sensor correlations with ratio:")
    for param in ['area', 'perimeter', 'eccentricity', 'green', 'blue']:
        corr = nad_cells[param].corr(nad_cells['ratio'])
        print(f"{param}: {corr:.3f}")
    
    print("\nVenus correlations with ratio:")
    for param in ['area', 'perimeter', 'eccentricity', 'green', 'blue']:
        corr = venus_cells[param].corr(venus_cells['ratio'])
        print(f"{param}: {corr:.3f}")

if __name__ == '__main__':
    main()
