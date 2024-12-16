#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit, root_scalar
import argparse

def variable_slope_model(x, bottom, top, logEC50, slope):
    """Four parameter logistic curve (variable slope model)"""
    return bottom + (top - bottom)/(1 + 10**((logEC50 - x)*slope))

def plot_parameter_correlations(cells_df, output_dir, sensor_type, ylim=(0, 4)):
    """Plot correlations between cell parameters and by position"""
    # Overall correlations
    params = ['area', 'perimeter', 'eccentricity', 'green', 'blue', 'ratio']
    corr_matrix = cells_df[params].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'{sensor_type} Overall Parameter Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / f'{sensor_type.lower()}_correlations.png')
    plt.close()
    
    # Plot correlations by position within each concentration
    concentrations = sorted(cells_df['treatment_group'].unique())
    for param in ['green', 'blue', 'area']:
        for conc in concentrations:
            conc_data = cells_df[cells_df['treatment_group'] == conc]
            positions = sorted(conc_data['position'].unique())
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 15))
            fig.suptitle(f'{sensor_type}: {param.capitalize()} vs Ratio\n{conc}')
            
            for i, pos in enumerate(positions):
                row = i // 3
                col = i % 3
                pos_data = conc_data[conc_data['position'] == pos]
                
                sns.scatterplot(data=pos_data, x=param, y='ratio', ax=axes[row, col], alpha=0.5)
                axes[row, col].set_title(f'Position {pos} (n={len(pos_data)})')
                axes[row, col].set_ylim(0, 4)
                axes[row, col].set_aspect('auto')
                
                # Add correlation coefficient
                corr = pos_data[param].corr(pos_data['ratio'])
                axes[row, col].text(0.05, 3.8, f'r = {corr:.2f}',
                                  transform=axes[row, col].transAxes,
                                  verticalalignment='top')
                
                # Add linear regression line
                x = pos_data[param]
                y = pos_data['ratio']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axes[row, col].plot(x, p(x), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{sensor_type.lower()}_{param}_vs_ratio_{conc}.png')
            plt.close()

def filter_intensity(cells_df, green_min=None, green_max=None, output_dir=None, label=''):
    """Filter cells based on green intensity thresholds
    
    Args:
        cells_df (pd.DataFrame): Cell data
        green_min (float): Minimum allowed green intensity
        green_max (float): Maximum allowed green intensity
        output_dir (Path): Directory to save plots
        label (str): Label for plot titles (e.g., 'NAD' or 'Venus')
    """
    filtered_df = cells_df.copy()
    
    print(f"\nIntensity filtering summary:")
    print(f"Original cells: {len(cells_df)}")
    
    if green_min is not None:
        filtered_df = filtered_df[filtered_df['green'] >= green_min]
        print(f"Removed {len(cells_df) - len(filtered_df)} cells with green < {green_min:.1f}")
    
    if green_max is not None:
        cells_before = len(filtered_df)
        filtered_df = filtered_df[filtered_df['green'] <= green_max]
        print(f"Removed {cells_before - len(filtered_df)} cells with green > {green_max:.1f}")
    
    print(f"Final filtered cells: {len(filtered_df)}")
    
    # Plot intensity distribution if output_dir is provided
    if output_dir is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(cells_df['green'], bins=50, alpha=0.5, label='Original')
        plt.hist(filtered_df['green'], bins=50, alpha=0.5, label='Filtered')
        if green_min is not None:
            plt.axvline(x=green_min, color='r', linestyle='--', label='Min threshold')
        if green_max is not None:
            plt.axvline(x=green_max, color='r', linestyle='--', label='Max threshold')
        plt.xlabel('Green Intensity')
        plt.ylabel('Count')
        plt.title(f'{label} Green Intensity Distribution')
        plt.legend()
        plt.savefig(output_dir / f'{label.lower()}_intensity_distribution.png')
        plt.close()
    
    return filtered_df

def plot_dose_response(nad_cells, venus_cells, output_dir):
    """Plot both raw and normalized dose-response curves"""
    #Define concentrations and their mapping
    conc_map = {
        'concentration_1': 0.1,
        'concentration_2': 1,
        'concentration_3': 10,
        'concentration_4': 100,
        'concentration_5': 500,
        'concentration_6': 1000,
        'concentration_7': 10000
    }
    
    # Calculate ratios and errors
    norm_ratios = []
    raw_ratios = []
    ratio_errors = []
    concentrations = []
    
    for conc_name, conc_val in conc_map.items():
        nad_group = nad_cells[nad_cells['treatment_group'] == conc_name]
        venus_group = venus_cells[venus_cells['treatment_group'] == conc_name]
        
        # Skip if no data for this concentration
        if len(nad_group) == 0 or len(venus_group) == 0:
            continue
        
        # Calculate position-wise means
        nad_pos_means = nad_group.groupby('position')['ratio'].mean()
        venus_pos_means = venus_group.groupby('position')['ratio'].mean()
        
        # Skip if no valid positions
        if len(nad_pos_means) == 0 or len(venus_pos_means) == 0:
            continue
        
        # Calculate all possible ratios
        all_ratios = []
        for nad_ratio in nad_pos_means:
            for venus_ratio in venus_pos_means:
                all_ratios.append(nad_ratio / venus_ratio)
        
        if len(all_ratios) > 0:
            raw_ratios.append(np.mean(all_ratios))
            ratio_errors.append(np.std(all_ratios))
            concentrations.append(conc_val)
    
    # Convert to arrays
    raw_ratios = np.array(raw_ratios)
    ratio_errors = np.array(ratio_errors)
    log_concs = np.log10(concentrations)
    
    # Also store normalized version
    norm_ratios = raw_ratios / raw_ratios[0]
    norm_errors = ratio_errors / raw_ratios[0]
    
    # Plot raw data
    plt.figure(figsize=(10, 6))
    plt.errorbar(log_concs, raw_ratios, yerr=ratio_errors, fmt='o', label='Data')
    plt.xlabel('log[NAD] (μM)')
    plt.ylabel('NAD/Venus Ratio')
    plt.title('NAD Sensor Response (Raw Data)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'standard_curve_raw.png')
    plt.close()
    
    # Plot normalized data
    plt.figure(figsize=(10, 6))
    plt.errorbar(log_concs, norm_ratios, yerr=norm_errors, fmt='o', label='Data')
    plt.xlabel('log[NAD] (μM)')
    plt.ylabel('Normalized NAD/Venus Ratio')
    plt.title('NAD Sensor Response (Normalized to First Point)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'standard_curve_normalized.png')
    plt.close()
    
    # Print statistics
    print("\nDose-response statistics:")
    for conc, raw, norm, err in zip(concentrations, raw_ratios, norm_ratios, ratio_errors):
        print(f"[NAD] = {conc:6.1f} μM: Raw = {raw:.3f} ± {err:.3f}, Normalized = {norm:.3f}")
    
    return None  # For now, removing concentration calculation until curve fitting is fixed

def analyze_regions(nad_cells, venus_cells, output_dir, exclude_positions=None):
    """Analyze data by individual regions within each concentration"""
    # Convert exclude_positions to list of integers if provided
    excluded = []
    if exclude_positions:
        excluded = [int(p.strip()) for p in exclude_positions.split(',')]
        print(f"\nExcluding positions: {excluded}")
        nad_cells = nad_cells[~nad_cells['position'].isin(excluded)]
        venus_cells = venus_cells[~venus_cells['position'].isin(excluded)]
    
    concentrations = sorted(nad_cells['treatment_group'].unique())
    
    # Plot regional analysis
    for conc in concentrations:
        fig, axes = plt.subplots(2, 3, figsize=(15, 15))
        fig.suptitle(f'Regional Analysis for {conc}')
        
        conc_data = nad_cells[nad_cells['treatment_group'] == conc]
        positions = sorted(conc_data['position'].unique())
        
        for i, pos in enumerate(positions):
            row = i // 3
            col = i % 3
            
            pos_data = conc_data[conc_data['position'] == pos]
            
            # Plot with fixed y-range
            axes[row, col].scatter(pos_data['green'], pos_data['ratio'], alpha=0.5)
            axes[row, col].set_title(f'Position {pos} (n={len(pos_data)})')
            axes[row, col].set_xlabel('Green Intensity')
            axes[row, col].set_ylabel('Ratio')
            axes[row, col].set_ylim(0, 4)
            
            # Make subplot square
            axes[row, col].set_aspect('auto')
            
            # Add correlation coefficient
            corr = np.corrcoef(pos_data['green'], pos_data['ratio'])[0,1]
            axes[row, col].text(0.05, 3.8, f'r = {corr:.2f}',
                              transform=axes[row, col].transAxes,
                              verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'regional_analysis_{conc}.png')
        plt.close()

def plot_intensity_ratios(cells_df, output_dir, sensor_type):
    """Plot green vs blue intensity ratios for individual cells"""
    concentrations = sorted(cells_df['treatment_group'].unique())
    
    # Create plots for each concentration
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{sensor_type}: Green vs Blue Intensity by Concentration', fontsize=16)
    
    # Define color palette for positions
    colors = plt.cm.tab10(np.linspace(0, 1, 6))  # 6 positions per concentration
    
    for i, conc in enumerate(concentrations):
        row = i // 4
        col = i % 4
        
        # Get data for this concentration
        conc_data = cells_df[cells_df['treatment_group'] == conc]
        
        # Plot each position with different color and its own regression line
        for j, pos in enumerate(sorted(conc_data['position'].unique())):
            pos_data = conc_data[conc_data['position'] == pos]
            
            # Scatter plot with smaller points
            axes[row, col].scatter(pos_data['green'], pos_data['blue'], 
                                 alpha=0.3, s=10,  # Even smaller points
                                 color=colors[j], 
                                 label=f'Pos {pos} (r={pos_data["ratio"].mean():.2f})')
            
            # Add position-specific regression line
            x = pos_data['green']
            y = pos_data['blue']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[row, col].plot(x, p(x), "--", color=colors[j], alpha=0.8)
        
        axes[row, col].set_title(f'{conc}\n(n={len(conc_data)})')
        axes[row, col].set_xlabel('Green Intensity')
        axes[row, col].set_ylabel('Blue Intensity')
        
        # Add overall correlation coefficient
        corr = np.corrcoef(conc_data['green'], conc_data['blue'])[0,1]
        axes[row, col].text(0.05, 0.95, f'Overall r = {corr:.2f}',
                          transform=axes[row, col].transAxes,
                          verticalalignment='top')
        
        # Add legend with mean ratios
        axes[row, col].legend(fontsize='x-small', 
                            bbox_to_anchor=(1.05, 1), 
                            loc='upper left')
        
        # Set equal aspect ratio
        axes[row, col].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{sensor_type.lower()}_intensity_ratios.png',
                bbox_inches='tight', dpi=300)  # Higher DPI for better quality
    plt.close()

def setup_output_folders(output_dir):
    """Create organized output folders"""
    folders = {
        'correlations': output_dir / 'correlations',
        'intensity': output_dir / 'intensity',
        'dose_response': output_dir / 'dose_response',
        'regional': output_dir / 'regional'
    }
    
    for folder in folders.values():
        folder.mkdir(exist_ok=True, parents=True)
    
    return folders

def plot_all_cells_overview(nad_cells, venus_cells, output_dir):
    """Create overview plots of all cells' intensity relationships"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # NAD sensor green vs ratio
    sns.scatterplot(data=nad_cells, x='green', y='ratio', 
                   alpha=0.2, s=5, color='green',  # Smaller points
                   ax=axes[0,0])
    axes[0,0].set_title('NAD Sensor: Green vs Ratio')
    corr = np.corrcoef(nad_cells['green'], nad_cells['ratio'])[0,1]
    axes[0,0].text(0.05, 0.95, f'r = {corr:.2f}', transform=axes[0,0].transAxes,
                  verticalalignment='top')
    
    # NAD sensor blue vs ratio
    sns.scatterplot(data=nad_cells, x='blue', y='ratio',
                   alpha=0.2, s=5, color='blue',
                   ax=axes[0,1])
    axes[0,1].set_title('NAD Sensor: Blue vs Ratio')
    corr = np.corrcoef(nad_cells['blue'], nad_cells['ratio'])[0,1]
    axes[0,1].text(0.05, 0.95, f'r = {corr:.2f}', transform=axes[0,1].transAxes,
                  verticalalignment='top')
    
    # Venus green vs ratio
    sns.scatterplot(data=venus_cells, x='green', y='ratio',
                   alpha=0.2, s=5, color='green',
                   ax=axes[1,0])
    axes[1,0].set_title('Venus: Green vs Ratio')
    corr = np.corrcoef(venus_cells['green'], venus_cells['ratio'])[0,1]
    axes[1,0].text(0.05, 0.95, f'r = {corr:.2f}', transform=axes[1,0].transAxes,
                  verticalalignment='top')
    
    # Venus blue vs ratio
    sns.scatterplot(data=venus_cells, x='blue', y='ratio',
                   alpha=0.2, s=5, color='blue',
                   ax=axes[1,1])
    axes[1,1].set_title('Venus: Blue vs Ratio')
    corr = np.corrcoef(venus_cells['blue'], venus_cells['ratio'])[0,1]
    axes[1,1].text(0.05, 0.95, f'r = {corr:.2f}', transform=axes[1,1].transAxes,
                  verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_cells_overview.png', dpi=300)
    plt.close()

def plot_green_blue_correlations(nad_cells, venus_cells, output_dir):
    """Plot green vs blue intensity correlations for all cells"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # NAD sensor
    sns.scatterplot(data=nad_cells, x='green', y='blue',
                   alpha=0.2, s=5,  # Small points for better visibility
                   ax=ax1)
    ax1.set_title('NAD Sensor: Green vs Blue')
    corr = np.corrcoef(nad_cells['green'], nad_cells['blue'])[0,1]
    ax1.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax1.transAxes,
             verticalalignment='top')
    ax1.set_aspect('equal')
    
    # Venus
    sns.scatterplot(data=venus_cells, x='green', y='blue',
                   alpha=0.2, s=5,
                   ax=ax2)
    ax2.set_title('Venus: Green vs Blue')
    corr = np.corrcoef(venus_cells['green'], venus_cells['blue'])[0,1]
    ax2.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax2.transAxes,
             verticalalignment='top')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'green_blue_correlations.png', dpi=300)
    plt.close()

def plot_intensity_by_concentration(cells_df, output_dir, sensor_type):
    """Plot green vs blue intensity for each concentration with position-specific colors"""
    concentrations = sorted(cells_df['treatment_group'].unique())
    
    # Create plots for each concentration
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'{sensor_type}: Green vs Blue Intensity by Concentration', fontsize=16)
    
    # Define color palette for positions (6 positions per concentration)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
    for i, conc in enumerate(concentrations):
        row = i // 4
        col = i % 4
        
        # Get data for this concentration
        conc_data = cells_df[cells_df['treatment_group'] == conc]
        
        # Plot each position with different color
        for j, pos in enumerate(sorted(conc_data['position'].unique())):
            pos_data = conc_data[conc_data['position'] == pos]
            
            # Scatter plot with small points
            axes[row, col].scatter(pos_data['green'], pos_data['blue'], 
                                 alpha=0.3, s=5,  # Small points
                                 color=colors[j], 
                                 label=f'Pos {pos} (r={pos_data["ratio"].mean():.2f})')
            
            # Add regression line
            x = pos_data['green']
            y = pos_data['blue']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[row, col].plot(x, p(x), "--", color=colors[j], alpha=0.8)
        
        axes[row, col].set_title(f'{conc}\n(n={len(conc_data)})')
        axes[row, col].set_xlabel('Green Intensity')
        axes[row, col].set_ylabel('Blue Intensity')
        
        # Add overall correlation
        corr = np.corrcoef(conc_data['green'], conc_data['blue'])[0,1]
        axes[row, col].text(0.05, 0.95, f'Overall r = {corr:.2f}',
                          transform=axes[row, col].transAxes,
                          verticalalignment='top')
        
        # Add legend
        axes[row, col].legend(fontsize='x-small', 
                            bbox_to_anchor=(1.05, 1), 
                            loc='upper left')
        
        # Set equal aspect ratio
        axes[row, col].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{sensor_type.lower()}_intensity_ratios.png',
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_ratio_stability(cells_df, output_dir, sensor_type):
    """Analyze where ratio stabilizes with respect to green and blue intensities"""
    
    # Create figure for both channels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Analyze green channel stability
    df_green = cells_df.sort_values('green')
    window = len(cells_df) // 50  # Use 2% of data points for window
    rolling_mean_green = df_green['ratio'].rolling(window=window, center=True).mean()
    rolling_std_green = df_green['ratio'].rolling(window=window, center=True).std()
    
    # Plot green channel analysis
    ax1.scatter(df_green['green'], df_green['ratio'], alpha=0.1, s=1, color='green')
    ax1.plot(df_green['green'], rolling_mean_green, 'r-', label='Rolling mean')
    ax1.fill_between(df_green['green'], 
                    rolling_mean_green - rolling_std_green,
                    rolling_mean_green + rolling_std_green,
                    alpha=0.2, color='red')
    
    # Find thresholds for green
    green_lower = np.percentile(cells_df['green'], 40)  # 40th percentile as lower threshold
    green_upper = df_green['green'].iloc[np.where(rolling_mean_green == rolling_mean_green.max())[0][0]]
    
    # Add threshold lines for green
    ax1.axvline(x=green_lower, color='black', linestyle='--',
               label=f'Lower threshold {green_lower:.0f}')
    ax1.axvline(x=green_upper, color='red', linestyle='--',
               label=f'Upper threshold {green_upper:.0f}')
    
    ax1.set_xlabel('Green Intensity')
    ax1.set_ylabel('Ratio')
    ax1.set_title(f'{sensor_type}: Green Channel Stability')
    ax1.legend()
    
    # Analyze blue channel stability
    df_blue = cells_df.sort_values('blue')
    rolling_mean_blue = df_blue['ratio'].rolling(window=window, center=True).mean()
    rolling_std_blue = df_blue['ratio'].rolling(window=window, center=True).std()
    
    # Plot blue channel analysis
    ax2.scatter(df_blue['blue'], df_blue['ratio'], alpha=0.1, s=1, color='blue')
    ax2.plot(df_blue['blue'], rolling_mean_blue, 'r-', label='Rolling mean')
    ax2.fill_between(df_blue['blue'], 
                    rolling_mean_blue - rolling_std_blue,
                    rolling_mean_blue + rolling_std_blue,
                    alpha=0.2, color='red')
    
    # Find thresholds for blue
    blue_lower = np.percentile(cells_df['blue'], 40)  # 40th percentile as lower threshold
    blue_upper = df_blue['blue'].iloc[np.where(rolling_mean_blue == rolling_mean_blue.max())[0][0]]
    
    # Add threshold lines for blue
    ax2.axvline(x=blue_lower, color='black', linestyle='--',
               label=f'Lower threshold {blue_lower:.0f}')
    ax2.axvline(x=blue_upper, color='red', linestyle='--',
               label=f'Upper threshold {blue_upper:.0f}')
    
    ax2.set_xlabel('Blue Intensity')
    ax2.set_ylabel('Ratio')
    ax2.set_title(f'{sensor_type}: Blue Channel Stability')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{sensor_type.lower()}_ratio_stability.png', dpi=300)
    plt.close()
    
    # Print summary
    print(f"\n{sensor_type} Stability Analysis:")
    print(f"Green channel thresholds: {green_lower:.0f} - {green_upper:.0f}")
    print(f"Blue channel thresholds: {blue_lower:.0f} - {blue_upper:.0f}")
    
    return (green_lower, green_upper), (blue_lower, blue_upper)

def main():
    parser = argparse.ArgumentParser(description='Analyze NAD sensor results')
    parser.add_argument('results_dir', help='Directory containing results')
    parser.add_argument('--correlations-only', action='store_true',
                      help='Only generate correlation plots')
    # NAD sensor options
    parser.add_argument('--nad-green-min', type=float, default=None,
                      help='Minimum allowed green intensity for NAD sensor')
    parser.add_argument('--nad-green-max', type=float, default=None,
                      help='Maximum allowed green intensity for NAD sensor')
    parser.add_argument('--exclude-nad-positions', type=str, default=None,
                      help='Comma-separated list of positions to exclude from NAD (e.g., "0,5,23")')
    # Venus options
    parser.add_argument('--venus-green-min', type=float, default=None,
                      help='Minimum allowed green intensity for Venus')
    parser.add_argument('--venus-green-max', type=float, default=None,
                      help='Maximum allowed green intensity for Venus')
    parser.add_argument('--exclude-venus-positions', type=str, default=None,
                      help='Comma-separated list of positions to exclude from Venus (e.g., "0,5,23")')
    # Add the missing argument
    parser.add_argument('--exclude-positions', type=str, default=None,
                      help='Comma-separated list of positions to exclude from analysis (e.g., "0,5,23")')
    # Add threshold mode arguments
    parser.add_argument('--auto-threshold', action='store_true',
                      help='Automatically determine intensity thresholds')
    parser.add_argument('--nad-green-range', type=str, default=None,
                      help='Manual NAD green intensity range (min,max), e.g., "800,2000"')
    parser.add_argument('--venus-green-range', type=str, default=None,
                      help='Manual Venus green intensity range (min,max), e.g., "1000,2500"')
    
    args = parser.parse_args()
    output_dir = Path(args.results_dir)
    
    # Load and process data
    nad_cells = pd.read_csv(output_dir / "nad_cells.csv")
    venus_cells = pd.read_csv(output_dir / "venus_cells.csv")
    
    # Handle position exclusions and filtering
    if args.exclude_nad_positions:
        excluded_nad = [int(p.strip()) for p in args.exclude_nad_positions.split(',')]
        print(f"\nExcluding NAD positions: {excluded_nad}")
        nad_cells = nad_cells[~nad_cells['position'].isin(excluded_nad)]
    
    if args.exclude_venus_positions:
        excluded_venus = [int(p.strip()) for p in args.exclude_venus_positions.split(',')]
        print(f"\nExcluding Venus positions: {excluded_venus}")
        venus_cells = venus_cells[~venus_cells['position'].isin(excluded_venus)]
    
    # Setup output folders
    folders = setup_output_folders(output_dir)
    
    # Always generate initial plots
    plot_intensity_by_concentration(nad_cells, folders['correlations'], 'NAD')
    plot_intensity_by_concentration(venus_cells, folders['correlations'], 'Venus')
    plot_all_cells_overview(nad_cells, venus_cells, folders['correlations'])
    plot_green_blue_correlations(nad_cells, venus_cells, folders['correlations'])
    plot_parameter_correlations(nad_cells, folders['correlations'], 'NAD')
    plot_parameter_correlations(venus_cells, folders['correlations'], 'Venus')
    
    # Analyze stability and set thresholds
    print("\nAnalyzing ratio stability...")
    (nad_auto_min, nad_auto_max), _ = analyze_ratio_stability(nad_cells, folders['intensity'], 'NAD')
    (venus_auto_min, venus_auto_max), _ = analyze_ratio_stability(venus_cells, folders['intensity'], 'Venus')
    
    # Set thresholds based on mode
    if args.auto_threshold:
        print("\nUsing auto-detected thresholds:")
        args.nad_green_min = nad_auto_min
        args.nad_green_max = nad_auto_max
        args.venus_green_min = venus_auto_min
        args.venus_green_max = venus_auto_max
    elif args.nad_green_range or args.venus_green_range:
        print("\nUsing manual thresholds:")
        if args.nad_green_range:
            nad_min, nad_max = map(float, args.nad_green_range.split(','))
            args.nad_green_min = nad_min
            args.nad_green_max = nad_max
            print(f"NAD green range: {nad_min:.0f} - {nad_max:.0f}")
        if args.venus_green_range:
            venus_min, venus_max = map(float, args.venus_green_range.split(','))
            args.venus_green_min = venus_min
            args.venus_green_max = venus_max
            print(f"Venus green range: {venus_min:.0f} - {venus_max:.0f}")
    else:
        print("\nNo thresholds specified. Using auto-detected thresholds:")
        args.nad_green_min = nad_auto_min
        args.nad_green_max = nad_auto_max
        args.venus_green_min = venus_auto_min
        args.venus_green_max = venus_auto_max
    
    # Print final thresholds
    print(f"\nFinal thresholds:")
    print(f"NAD green: {args.nad_green_min:.0f} - {args.nad_green_max:.0f}")
    print(f"Venus green: {args.venus_green_min:.0f} - {args.venus_green_max:.0f}")
    
    # Filter intensities
    if args.nad_green_min is not None or args.nad_green_max is not None:
        nad_cells = filter_intensity(nad_cells, args.nad_green_min, args.nad_green_max,
                                   folders['intensity'], 'NAD')
    
    if args.venus_green_min is not None or args.venus_green_max is not None:
        venus_cells = filter_intensity(venus_cells, args.venus_green_min, args.venus_green_max,
                                     folders['intensity'], 'Venus')
    
    # Update dose response plotting
    cell_conc = plot_dose_response(nad_cells, venus_cells, folders['dose_response'])
    if cell_conc is not None:
        print(f"\nEstimated cell sample concentration: {cell_conc:.1f} μM")

if __name__ == '__main__':
    main() 