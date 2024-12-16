#!/usr/bin/env python3

# Core imports
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
import argparse

# Image processing
from nd2reader import ND2Reader
from cellpose import models
from skimage import (
    segmentation,
    measure,
    morphology,
    filters
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Scientific computing
import scipy.stats as stats
from scipy.optimize import curve_fit

class NADAnalyzer:
    def __init__(self, model_type='cyto', gpu=False, log_file='analysis.log'):
        """Initialize NAD Analyzer with specialized parameters
        
        Args:
            model_type (str): Cellpose model type ('cyto' or 'nuclei')
            gpu (bool): Whether to use GPU acceleration
            log_file (str): Path to log file
        """
        self.setup_logging(log_file)
        self.logger = logging.getLogger('NADAnalyzer')
        self.logger.info("Initializing NAD Analyzer...")
        
        # Always use cyto model with specialized parameters
        self.model = models.Cellpose(model_type='cyto', gpu=gpu)
        self.logger.info(f"Loaded Cellpose model: cyto_specialized (GPU: {gpu})")
        
        # Store specialized parameters
        self.segmentation_params = {
            'diameter': 70,
            'flow_threshold': 0.8,
            'cellprob_threshold': -1.0,
            'min_size': 30
        }
        
        # Define treatment groups (6 positions per group)
        self.treatment_groups = {
            range(0, 6): 'concentration_1',
            range(6, 12): 'concentration_2',
            range(12, 18): 'concentration_3',
            range(18, 24): 'concentration_4',
            range(24, 30): 'concentration_5',
            range(30, 36): 'concentration_6',
            range(36, 42): 'concentration_7',
            range(42, 48): 'concentration_8'
        }
    
    def setup_logging(self, log_file):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def find_channel_by_wavelength(self, channels, wavelength):
        """Find channel containing wavelength string, case-insensitive
        
        Args:
            channels (list): List of Channel objects from ND2 metadata
            wavelength (str): Wavelength to search for (e.g., '488', '405')
            
        Returns:
            str: Matching channel name or None
        """
        wavelength = wavelength.lower()
        for channel in channels:
            # Extract channel name from Channel object
            channel_name = channel.channel.name
            if wavelength in channel_name.lower():
                return channel_name
        return None
    
    def find_nd2_files(self, data_folder):
        """Find NAD and Venus ND2 files in data folder based on file names"""
        self.logger.info(f"Searching for ND2 files in: {data_folder}")
        data_path = Path(data_folder)
        
        # Search for files containing 'sensor' or 'venus' in their names
        sensor_files = list(data_path.glob("*sensor*.nd2"))
        venus_files = list(data_path.glob("*venus*.nd2"))
        
        if not sensor_files:
            raise ValueError("Could not find NAD sensor file (should contain 'sensor' in name)")
        if not venus_files:
            raise ValueError("Could not find Venus file (should contain 'venus' in name)")
        
        # Use the first matching file if multiple found
        nad_file = str(sensor_files[0])
        venus_file = str(venus_files[0])
        
        # Verify channels in both files
        for file_path, file_type in [(nad_file, "NAD sensor"), (venus_file, "Venus")]:
            with ND2Reader(file_path) as images:
                # Get channel information from metadata
                metadata = images.metadata
                channels = metadata['channels']
                self.logger.info(f"{file_type} file channels: {channels}")
                
                has_488 = any('488' in str(ch).lower() for ch in channels)
                has_405 = any('405' in str(ch).lower() for ch in channels)
                
                if not has_488:
                    raise ValueError(f"{file_type} file {file_path} does not contain 488nm channel")
                if not has_405:
                    raise ValueError(f"{file_type} file {file_path} does not contain 405nm channel")
        
        self.logger.info(f"Found NAD sensor file: {nad_file}")
        self.logger.info(f"Found Venus file: {venus_file}")
        
        if len(sensor_files) > 1 or len(venus_files) > 1:
            self.logger.warning(f"Multiple matching files found. Using: {nad_file} and {venus_file}")
        
        return nad_file, venus_file
    
    def get_treatment_group(self, position):
        """Get treatment group for a given position"""
        for pos_range, group in self.treatment_groups.items():
            if position in pos_range:
                return group
        return None
        
    def load_image(self, file_path, position, channel):
        """Load specific position and channel from ND2 file"""
        with ND2Reader(file_path) as images:
            # Get channel information from metadata
            metadata = images.metadata
            channels = metadata['channels']
            
            # Green channel is always 488nm
            if channel == 'green':
                channel_idx = next((i for i, ch in enumerate(channels) 
                                  if '488' in str(ch).lower()), None)
                if channel_idx is None:
                    raise ValueError(f"Could not find 488nm channel in {file_path}")
            # Blue channel is always 405nm
            elif channel == 'blue':
                channel_idx = next((i for i, ch in enumerate(channels) 
                                  if '405' in str(ch).lower()), None)
                if channel_idx is None:
                    raise ValueError(f"Could not find 405nm channel in {file_path}")
            else:
                raise ValueError(f"Invalid channel: {channel}")
            
            # Set position and get frame
            images.default_coords['v'] = position  # Set position
            frame = images[channel_idx]  # Get frame for channel
            
            return frame.astype(np.float32)
    
    def segment_cells(self, image, diameter=None):
        """Segment cells using specialized Cellpose parameters"""
        # Normalize image to 0-1 range
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Convert to uint8 (0-255)
        image_8bit = (image_norm * 255).astype('uint8')
        
        # Use specialized parameters
        try:
            # Run initial segmentation with adjusted parameters
            masks, _, _, _ = self.model.eval(
                image_8bit,
                diameter=self.segmentation_params['diameter'],
                channels=[0,0],
                flow_threshold=0.6,  # Adjusted for better edge detection
                cellprob_threshold=-0.5,  # More lenient detection
                min_size=30,
                do_3D=False
            )
            
            # Post-process masks to exclude nuclear holes
            from skimage.morphology import remove_small_holes, binary_dilation, binary_erosion
            from scipy import ndimage
            
            processed_masks = np.zeros_like(masks)
            for label in range(1, masks.max() + 1):
                cell_mask = masks == label
                
                # Get cell boundary
                boundary = binary_dilation(cell_mask) & ~cell_mask
                
                # Check intensity along boundary
                boundary_intensity = np.mean(image_norm[boundary])
                
                # Get potential holes
                holes = ndimage.binary_fill_holes(cell_mask) & ~cell_mask
                
                # Check each hole
                labeled_holes, num_holes = ndimage.label(holes)
                for hole_idx in range(1, num_holes + 1):
                    hole = labeled_holes == hole_idx
                    hole_intensity = np.mean(image_norm[hole])
                    
                    # If hole is significantly darker than boundary, keep it
                    if hole_intensity < 0.6 * boundary_intensity:
                        cell_mask = cell_mask | hole
                
                processed_masks[cell_mask] = label
            
            print(f"Segmentation found {len(np.unique(processed_masks))-1} cells")
            return processed_masks
            
        except Exception as e:
            print(f"Segmentation failed: {str(e)}")
            return None
    
    def filter_border_objects(self, masks):
        """Remove objects touching the image border"""
        border_labels = np.unique(np.concatenate([
            masks[0,:], masks[-1,:],
            masks[:,0], masks[:,-1]
        ]))
        for label in border_labels:
            if label != 0:  # don't remove background
                masks[masks == label] = 0
        return masks
    
    def measure_intensity(self, image, masks, min_intensity=0.01, max_intensity=0.99):
        """Measure mean intensity for each cell and filter out dim/saturated cells"""
        # Normalize image to 0-1 range for consistent thresholds
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        results = []
        filtered_dim = 0
        filtered_bright = 0
        filtered_saturated = 0
        
        for cell_id in range(1, masks.max() + 1):
            cell_mask = masks == cell_id
            
            # Skip if mask is empty
            if not np.any(cell_mask):
                continue
            
            # Calculate intensity metrics
            mean_intensity = np.mean(image_norm[cell_mask])
            max_pixel = np.max(image_norm[cell_mask])
            
            # Skip if NaN values
            if np.isnan(mean_intensity) or np.isnan(max_pixel):
                continue
            
            # Debug intensity values
            self.logger.debug(f"Cell {cell_id}: mean={mean_intensity:.3f}, max={max_pixel:.3f}")
            
            # Filter with detailed logging
            if mean_intensity < min_intensity:
                filtered_dim += 1
                continue
            if mean_intensity > max_intensity:
                filtered_bright += 1
                continue
            if max_pixel > 0.98:  # Relaxed saturation threshold
                filtered_saturated += 1
                continue
            
            results.append({
                'cell_id': cell_id,
                'mean_intensity': mean_intensity,
                'max_intensity': max_pixel,
                'area': np.sum(cell_mask)
            })
        
        # Log filtering results
        total_cells = masks.max()
        passed_cells = len(results)
        self.logger.info(f"Intensity filtering results for {total_cells} cells:")
        self.logger.info(f"  Filtered dim: {filtered_dim}")
        self.logger.info(f"  Filtered bright: {filtered_bright}")
        self.logger.info(f"  Filtered saturated: {filtered_saturated}")
        self.logger.info(f"  Passed: {passed_cells}")
        
        return pd.DataFrame(results)
    
    def process_position(self, nad_file, venus_file, position):
        """Process a single position"""
        self.logger.info(f"Processing position {position}")
        
        # Load images
        nad_green = self.load_image(nad_file, position, 'green')
        nad_blue = self.load_image(nad_file, position, 'blue')
        venus_green = self.load_image(venus_file, position, 'green')
        venus_blue = self.load_image(venus_file, position, 'blue')
        
        # Process NAD sensor cells
        nad_masks = self.segment_cells(nad_green)
        nad_masks = self.filter_border_objects(nad_masks)
        
        nad_cells = []
        for region in measure.regionprops(nad_masks):
            cell_mask = nad_masks == region.label
            nad_g = np.mean(nad_green[cell_mask])
            nad_b = np.mean(nad_blue[cell_mask])
            if nad_b > 0:  # Avoid division by zero
                nad_cells.append({
                    'position': position,
                    'treatment_group': self.get_treatment_group(position),
                    'cell_id': region.label,
                    'area': region.area,  # Keep only area as it's useful for filtering
                    'green': nad_g,
                    'blue': nad_b,
                    'ratio': nad_g / nad_b
                })
        
        # Process Venus cells
        venus_masks = self.segment_cells(venus_green)
        venus_masks = self.filter_border_objects(venus_masks)
        
        venus_cells = []
        for region in measure.regionprops(venus_masks):
            cell_mask = venus_masks == region.label
            venus_g = np.mean(venus_green[cell_mask])
            venus_b = np.mean(venus_blue[cell_mask])
            if venus_b > 0:  # Avoid division by zero
                venus_cells.append({
                    'position': position,
                    'treatment_group': self.get_treatment_group(position),
                    'cell_id': region.label,
                    'area': region.area,  # Keep only area as it's useful for filtering
                    'green': venus_g,
                    'blue': venus_b,
                    'ratio': venus_g / venus_b
                })
        
        # Calculate region averages
        if len(nad_cells) > 0 and len(venus_cells) > 0:
            nad_df = pd.DataFrame(nad_cells)
            venus_df = pd.DataFrame(venus_cells)
            
            # Calculate region statistics
            nad_mean_ratio = nad_df['ratio'].mean()
            venus_mean_ratio = venus_df['ratio'].mean()
            normalized_ratio = nad_mean_ratio / venus_mean_ratio
            
            return {
                'summary': pd.DataFrame({
                    'position': [position],
                    'treatment_group': [self.get_treatment_group(position)],
                    'nad_cells': [len(nad_cells)],
                    'venus_cells': [len(venus_cells)],
                    'nad_mean_ratio': [nad_mean_ratio],
                    'nad_std_ratio': [nad_df['ratio'].std()],
                    'venus_mean_ratio': [venus_mean_ratio],
                    'venus_std_ratio': [venus_df['ratio'].std()],
                    'normalized_ratio': [normalized_ratio]
                }),
                'nad_cells': nad_df,
                'venus_cells': venus_df
            }
        
        return {
            'summary': pd.DataFrame(),
            'nad_cells': pd.DataFrame(),
            'venus_cells': pd.DataFrame()
        }
    
    def process_all_positions(self, nad_file, venus_file, output_dir):
        """Process all positions and save detailed cell data"""
        self.logger.info("Starting batch processing of all positions")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_summaries = []
        all_nad_cells = []
        all_venus_cells = []
        failed_positions = []
        
        with tqdm(total=48, desc="Processing positions") as pbar:
            for position in range(48):
                try:
                    results = self.process_position(nad_file, venus_file, position)
                    if not results['summary'].empty:
                        all_summaries.append(results['summary'])
                        all_nad_cells.append(results['nad_cells'])
                        all_venus_cells.append(results['venus_cells'])
                    pbar.set_postfix(cells=len(results['summary']))
                except Exception as e:
                    self.logger.error(f"Error processing position {position}: {str(e)}")
                    failed_positions.append(position)
                finally:
                    pbar.update(1)
        
        if failed_positions:
            self.logger.warning(f"Failed positions: {failed_positions}")
        
        # Combine and save all results
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_nad_cells = pd.concat(all_nad_cells, ignore_index=True)
        combined_venus_cells = pd.concat(all_venus_cells, ignore_index=True)
        
        # Save all data
        combined_summary.to_csv(output_dir / "results.csv", index=False)
        combined_nad_cells.to_csv(output_dir / "nad_cells.csv", index=False)
        combined_venus_cells.to_csv(output_dir / "venus_cells.csv", index=False)
        
        # Generate visualizations
        self.plot_dose_response(combined_summary, output_dir)
        
        self.logger.info("Completed processing all positions")
        return combined_summary
    
    def plot_dose_response(self, results, output_dir):
        """Generate dose-response plots
        
        Args:
            results (pd.DataFrame): Combined results from all positions
            output_dir (Path): Directory to save plots
        """
        # Box plot of normalized ratios by treatment group
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results, x='treatment_group', y='normalized_ratio')
        plt.xticks(rotation=45)
        plt.title('Normalized Ratio by Treatment Group')
        plt.tight_layout()
        plt.savefig(output_dir / 'dose_response_box.png')
        plt.close()
        
        # Violin plot for distribution visualization
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=results, x='treatment_group', y='normalized_ratio')
        plt.xticks(rotation=45)
        plt.title('Distribution of Normalized Ratios by Treatment Group')
        plt.tight_layout()
        plt.savefig(output_dir / 'dose_response_violin.png')
        plt.close()
    
    def test_image_loading(self, nad_file, venus_file, position):
        """Test image loading for a single position with visualization"""
        import matplotlib.pyplot as plt
        
        # Print file information
        with ND2Reader(nad_file) as images:
            print("\nNAD sensor file info:")
            print(f"Metadata: {images.metadata}")
            print(f"Channels: {images.metadata['channels']}")
            print(f"Sizes: {images.sizes}")
        
        with ND2Reader(venus_file) as images:
            print("\nVenus file info:")
            print(f"Metadata: {images.metadata}")
            print(f"Channels: {images.metadata['channels']}")
            print(f"Sizes: {images.sizes}")
        
        # Load all channels
        nad_green = self.load_image(nad_file, position, 'green')
        nad_blue = self.load_image(nad_file, position, 'blue')
        venus_green = self.load_image(venus_file, position, 'green')
        venus_blue = self.load_image(venus_file, position, 'blue')
        
        # Print image information
        print("\nImage shapes:")
        print(f"NAD green: {nad_green.shape}")
        print(f"NAD blue: {nad_blue.shape}")
        print(f"Venus green: {venus_green.shape}")
        print(f"Venus blue: {venus_blue.shape}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Plot images with proper scaling
        for ax, img, title in zip(axes.flat, 
                                [nad_green, nad_blue, venus_green, venus_blue],
                                ['NAD Green (Ex488)', 'NAD Blue (Ex405)', 
                                 'Venus Green (Ex488)', 'Venus Blue (Ex405)']):
            im = ax.imshow(img, cmap='gray')
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        return fig
    
    def test_segmentation(self, nad_file, venus_file, position):
        """Test segmentation for a single position with visualization"""
        # Load NAD green channel for segmentation
        nad_green = self.load_image(nad_file, position, 'green')
        
        # Print image information
        print("\nImage information:")
        print(f"Shape: {nad_green.shape}")
        print(f"Data type: {nad_green.dtype}")
        print(f"Value range: {nad_green.min()} to {nad_green.max()}")
        
        # Create visualization of input image
        fig = plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(nad_green, cmap='gray')
        plt.title('Original Image')
        plt.colorbar()
        
        # Try segmentation
        masks = self.segment_cells(nad_green)
        
        # Show binary mask
        plt.subplot(132)
        plt.imshow(masks > 0, cmap='gray')
        plt.title(f'Segmentation Mask ({len(np.unique(masks))-1} cells)')
        
        # Show overlay
        plt.subplot(133)
        from skimage.segmentation import find_boundaries
        boundaries = find_boundaries(masks)
        
        # Normalize image for overlay
        img_norm = (nad_green - nad_green.min()) / (nad_green.max() - nad_green.min())
        overlay = np.stack([img_norm]*3, axis=-1)
        overlay[boundaries] = [1, 0, 0]  # Red boundaries
        
        plt.imshow(overlay)
        plt.title('Overlay')
        
        plt.tight_layout()
        
        # Return the figure instead of masks
        return fig
    
    def test_ratio_calculation(self, nad_file, venus_file, position):
        """Test ratio calculation for a single position with visualization"""
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
        from skimage.measure import regionprops
        
        # Load all channels
        nad_green = self.load_image(nad_file, position, 'green')
        nad_blue = self.load_image(nad_file, position, 'blue')
        venus_green = self.load_image(venus_file, position, 'green')
        venus_blue = self.load_image(venus_file, position, 'blue')
        
        # Segment cells using NAD green channel
        masks = self.segment_cells(nad_green, diameter=100)
        masks = self.filter_border_objects(masks.copy())
        
        # Calculate intensities and ratios for NAD sensor
        nad_data = []
        for region in regionprops(masks):
            cell_id = region.label
            y, x = region.centroid
            
            # Get intensities for this cell
            cell_mask = masks == cell_id
            nad_g = np.mean(nad_green[cell_mask])
            nad_b = np.mean(nad_blue[cell_mask])
            ratio = nad_g / nad_b
            
            nad_data.append({
                'cell_id': cell_id,
                'x': x, 'y': y,
                'green': nad_g,
                'blue': nad_b,
                'ratio': ratio
            })
        
        # Segment cells using Venus green channel
        venus_masks = self.segment_cells(venus_green, diameter=100)
        venus_masks = self.filter_border_objects(venus_masks.copy())
        
        # Calculate intensities and ratios for Venus
        venus_data = []
        for region in regionprops(venus_masks):
            cell_id = region.label
            y, x = region.centroid
            
            # Get intensities for this cell
            cell_mask = venus_masks == cell_id
            venus_g = np.mean(venus_green[cell_mask])
            venus_b = np.mean(venus_blue[cell_mask])
            ratio = venus_g / venus_b
            
            venus_data.append({
                'cell_id': cell_id,
                'x': x, 'y': y,
                'green': venus_g,
                'blue': venus_b,
                'ratio': ratio
            })
        
        # Calculate region averages
        nad_region_ratio = np.mean([d['ratio'] for d in nad_data])
        venus_region_ratio = np.mean([d['ratio'] for d in venus_data])
        normalized_ratio = nad_region_ratio / venus_region_ratio
        
        # Create visualization
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 4, figure=fig)
        
        # Plot NAD channels
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(nad_green, cmap='gray')
        ax1.set_title('NAD Green (Ex488)')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(nad_blue, cmap='gray')
        ax2.set_title('NAD Blue (Ex405)')
        
        # Plot Venus channels
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(venus_green, cmap='gray')
        ax3.set_title('Venus Green (Ex488)')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(venus_blue, cmap='gray')
        ax4.set_title('Venus Blue (Ex405)')
        
        # Add intensity and ratio labels for NAD
        for cell in nad_data:
            # Label on green channel
            ax1.text(cell['x'], cell['y'], 
                    f"G:{cell['green']:.1f}", 
                    color='yellow', ha='center', va='bottom')
            # Label on blue channel
            ax2.text(cell['x'], cell['y'], 
                    f"B:{cell['blue']:.1f}\nR:{cell['ratio']:.2f}", 
                    color='yellow', ha='center', va='top')
        
        # Add intensity and ratio labels for Venus
        for cell in venus_data:
            # Label on green channel
            ax3.text(cell['x'], cell['y'], 
                    f"G:{cell['green']:.1f}", 
                    color='yellow', ha='center', va='bottom')
            # Label on blue channel
            ax4.text(cell['x'], cell['y'], 
                    f"B:{cell['blue']:.1f}\nR:{cell['ratio']:.2f}", 
                    color='yellow', ha='center', va='top')
        
        # Plot ratio distributions
        ax5 = fig.add_subplot(gs[1, 0:2])
        ratios = [d['ratio'] for d in nad_data]
        ax5.hist(ratios, bins=30, color='blue', alpha=0.7, label='NAD ratios')
        ax5.axvline(nad_region_ratio, color='blue', linestyle='--', 
                    label=f'NAD mean: {nad_region_ratio:.2f}')
        ax5.set_title('NAD Ratio Distribution')
        ax5.set_xlabel('Green/Blue Ratio')
        ax5.legend()
        
        ax6 = fig.add_subplot(gs[1, 2:4])
        ratios = [d['ratio'] for d in venus_data]
        ax6.hist(ratios, bins=30, color='green', alpha=0.7, label='Venus ratios')
        ax6.axvline(venus_region_ratio, color='green', linestyle='--', 
                    label=f'Venus mean: {venus_region_ratio:.2f}')
        ax6.set_title(f'Venus Ratio Distribution\nNormalized Ratio: {normalized_ratio:.2f}')
        ax6.set_xlabel('Green/Blue Ratio')
        ax6.legend()
        
        plt.tight_layout()
        return fig
    
    def test_file_structure(self, file_path):
        """Test ND2 file structure and print detailed information"""
        with ND2Reader(file_path) as images:
            metadata = images.metadata
            print(f"\nFile structure for: {file_path}")
            print(f"Total frames: {len(images)}")
            print(f"Channels: {metadata['channels']}")
            print(f"Images per channel: {metadata['total_images_per_channel']}")
            print(f"Metadata: {metadata}")
            
            # Try reading first few frames
            print("\nTesting frame access:")
            for i in range(min(5, len(images))):
                frame = images[i]
                print(f"Frame {i}: shape={frame.shape}")
    
    def plot_dose_response_curve(self, results, output_dir):
        """Generate dose-response curve
        
        Args:
            results (pd.DataFrame): Combined results from all positions
            output_dir (Path): Directory to save plots
        """
        # Define drug concentrations (you'll need to adjust these values)
        concentrations = {
            'concentration_1': 0,    # Control
            'concentration_2': 0.1,  # Adjust these values
            'concentration_3': 0.3,  # to match your actual
            'concentration_4': 1.0,  # drug concentrations
            'concentration_5': 3.0,  # in your experiment
            'concentration_6': 10.0,
            'concentration_7': 30.0,
            'concentration_8': 100.0
        }
        
        # Calculate mean and std for each concentration
        summary = results.groupby('treatment_group').agg({
            'normalized_ratio': ['mean', 'std', 'count']
        }).reset_index()
        summary.columns = ['treatment_group', 'mean_ratio', 'std_ratio', 'n']
        
        # Add concentration values
        summary['concentration'] = summary['treatment_group'].map(concentrations)
        
        # Create dose-response plot
        plt.figure(figsize=(10, 6))
        
        # Plot individual points
        for group in summary['treatment_group']:
            group_data = results[results['treatment_group'] == group]
            plt.scatter([concentrations[group]] * len(group_data), 
                       group_data['normalized_ratio'], 
                       alpha=0.3, color='gray')
        
        # Plot mean ± std
        plt.errorbar(summary['concentration'], 
                    summary['mean_ratio'],
                    yerr=summary['std_ratio'],
                    fmt='o-', color='blue', 
                    capsize=5, capthick=2, 
                    label='Mean ± STD')
        
        plt.xscale('log')
        plt.xlabel('Drug Concentration (µM)')
        plt.ylabel('Normalized Ratio (NAD/Venus)')
        plt.title('Dose-Response Curve')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        
        # Add sample sizes
        for i, row in summary.iterrows():
            plt.annotate(f'n={row["n"]}', 
                        (row['concentration'], row['mean_ratio']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dose_response_curve.png')
        plt.close()
        
        # Save summary statistics
        summary.to_csv(output_dir / 'dose_response_summary.csv', index=False)
    
    def filter_outliers(self, data, group_col, value_col, n_std=2):
        """Filter outliers within each group using IQR method
        
        Args:
            data (pd.DataFrame): Input data
            group_col (str): Column name for grouping
            value_col (str): Column name for values to filter
            n_std (float): Number of standard deviations for filtering
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = pd.DataFrame()
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group].copy()
            
            # Calculate mean and std for the group
            mean = group_data[value_col].mean()
            std = group_data[value_col].std()
            
            # Filter outliers
            mask = (group_data[value_col] > mean - n_std * std) & \
                   (group_data[value_col] < mean + n_std * std)
            
            # Print outlier information
            outliers = group_data[~mask]
            if len(outliers) > 0:
                print(f"\nOutliers in {group}:")
                print(f"Group mean: {mean:.3f}")
                print(f"Group std: {std:.3f}")
                print(f"Outlier values: {outliers[value_col].values}")
            
            filtered_data = pd.concat([filtered_data, group_data[mask]])
        
        print(f"\nRemoved {len(data) - len(filtered_data)} outliers from {len(data)} total points")
        return filtered_data
    
    def analyze_dose_response(self, results_file, output_dir):
        """Analyze dose-response data with different transformations and filtering"""
        results = pd.read_csv(results_file)
        output_dir = Path(output_dir)
        
        # Group by treatment and remove highest/lowest ratios
        filtered_results = []
        for group in results['treatment_group'].unique():
            group_data = results[results['treatment_group'] == group]
            if len(group_data) > 2:  # Only filter if we have more than 2 positions
                # Sort by normalized ratio
                sorted_data = group_data.sort_values('normalized_ratio')
                # Remove highest and lowest
                filtered_group = sorted_data.iloc[1:-1]
                filtered_results.append(filtered_group)
        
        filtered_results = pd.concat(filtered_results, ignore_index=True)
        
        # Save filtered results
        filtered_results.to_csv(output_dir / "filtered_results.csv", index=False)
        
        # Create comparison plots for raw and filtered data
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Define order of treatment groups
        treatment_order = [f'concentration_{i}' for i in range(1, 9)]
        
        # Plot raw data
        sns.boxplot(data=results, x='treatment_group', y='normalized_ratio', 
                    order=treatment_order, ax=axes[0])
        axes[0].set_title('Raw Data Distribution')
        axes[0].set_xticklabels(treatment_order, rotation=45)
        axes[0].set_xlabel('Treatment Group')
        axes[0].set_ylabel('Normalized Ratio')
        
        # Plot filtered data
        sns.boxplot(data=filtered_results, x='treatment_group', y='normalized_ratio', 
                    order=treatment_order, ax=axes[1])
        axes[1].set_title('Filtered Data Distribution\n(Removed highest and lowest from each group)')
        axes[1].set_xticklabels(treatment_order, rotation=45)
        axes[1].set_xlabel('Treatment Group')
        axes[1].set_ylabel('Normalized Ratio')
        
        # Add statistics
        for ax in axes:
            # Add number of points per group
            for i, group in enumerate(treatment_order):
                n_raw = len(results[results['treatment_group'] == group])
                n_filtered = len(filtered_results[filtered_results['treatment_group'] == group])
                ax.text(i, ax.get_ylim()[1], f'n={n_raw if ax == axes[0] else n_filtered}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'data_comparison.png')
        plt.close()
        
        # Generate dose-response curves for both raw and filtered data
        self.plot_dose_response_comparison(results, filtered_results, output_dir)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("\nRaw Data:")
        print(results.groupby('treatment_group')['normalized_ratio'].describe())
        print("\nFiltered Data:")
        print(filtered_results.groupby('treatment_group')['normalized_ratio'].describe())
    
    def plot_dose_response_comparison(self, raw_results, filtered_results, output_dir):
        """Plot dose-response curves for both raw and filtered data"""
        concentrations = {
            'concentration_1': 0.001,
            'concentration_2': 0.1,
            'concentration_3': 0.3,
            'concentration_4': 1.0,
            'concentration_5': 3.0,
            'concentration_6': 10.0,
            'concentration_7': 30.0,
            'concentration_8': 100.0
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Process and plot raw data
        raw_summary = raw_results.groupby('treatment_group').agg({
            'normalized_ratio': ['mean', 'std', 'count']
        }).reset_index()
        raw_summary.columns = ['treatment_group', 'mean_ratio', 'std_ratio', 'n']
        raw_summary['concentration'] = raw_summary['treatment_group'].map(concentrations)
        
        ax.errorbar(raw_summary['concentration'], 
                    raw_summary['mean_ratio'],
                    yerr=raw_summary['std_ratio'],
                    fmt='o--', color='gray', alpha=0.5,
                    capsize=5, capthick=2, 
                    label='Raw Data')
        
        # Add individual points
        for group in raw_summary['treatment_group']:
            conc = concentrations[group]
            group_data = raw_results[raw_results['treatment_group'] == group]
            ax.scatter([conc] * len(group_data), 
                      group_data['normalized_ratio'],
                      color='gray', alpha=0.2, s=30)
        
        # Plot filtered data
        filtered_summary = filtered_results.groupby('treatment_group').agg({
            'normalized_ratio': ['mean', 'std', 'count']
        }).reset_index()
        filtered_summary.columns = ['treatment_group', 'mean_ratio', 'std_ratio', 'n']
        filtered_summary['concentration'] = filtered_summary['treatment_group'].map(concentrations)
        
        ax.errorbar(filtered_summary['concentration'], 
                    filtered_summary['mean_ratio'],
                    yerr=filtered_summary['std_ratio'],
                    fmt='o-', color='blue',
                    capsize=5, capthick=2, 
                    label='Filtered Data')
        
        ax.set_xscale('log')
        ax.set_xlabel('Drug Concentration (µM)')
        ax.set_ylabel('Normalized Ratio (NAD/Venus)')
        ax.set_title('Dose-Response Curve Comparison')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dose_response_comparison.png')
        plt.close()
        
        # Save summary statistics
        summary = pd.DataFrame({
            'treatment_group': raw_summary['treatment_group'],
            'concentration': raw_summary['concentration'],
            'raw_mean': raw_summary['mean_ratio'],
            'raw_std': raw_summary['std_ratio'],
            'raw_n': raw_summary['n'],
            'filtered_mean': filtered_summary['mean_ratio'],
            'filtered_std': filtered_summary['std_ratio'],
            'filtered_n': filtered_summary['n']
        })
        summary.to_csv(output_dir / 'dose_response_comparison.csv', index=False)
    
    def visualize_segmentation(self, image, masks, output_path):
        """Visualize segmentation results with nuclear holes"""
        plt.figure(figsize=(20, 5))
        
        # Original image
        plt.subplot(141)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        
        # Segmentation masks
        plt.subplot(142)
        plt.imshow(masks > 0, cmap='gray')
        plt.title(f'Segmentation ({len(np.unique(masks))-1} cells)')
        
        # Overlay
        plt.subplot(143)
        boundaries = segmentation.find_boundaries(masks)
        img_norm = (image - image.min()) / (image.max() - image.min())
        overlay = np.stack([img_norm]*3, axis=-1)
        overlay[boundaries] = [1, 0, 0]  # Red boundaries
        plt.imshow(overlay)
        plt.title('Overlay')
        
        # Intensity profile
        plt.subplot(144)
        for label in range(1, masks.max() + 1):
            cell_mask = masks == label
            cell_intensities = image[cell_mask]
            plt.hist(cell_intensities, bins=50, alpha=0.1, color='blue')
        plt.title('Cell Intensity Distributions')
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def test_segmentation_models(self, nad_file, venus_file, position):
        """Compare different Cellpose models and parameters
        
        Tests:
        1. cyto: basic cytoplasm model
        2. cyto2: newer cytoplasm model
        3. cyto_specialized: cyto model with custom parameters
        4. nuclei: nuclei model (for comparison)
        """
        # Load NAD green channel
        nad_green = self.load_image(nad_file, position, 'green')
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 4, figure=fig)
        
        # Original image
        ax_orig = fig.add_subplot(gs[0, :])
        ax_orig.imshow(nad_green, cmap='gray')
        ax_orig.set_title('Original Image')
        
        # Test different models and parameters
        models_to_test = [
            {
                'name': 'cyto',
                'params': {
                    'model_type': 'cyto',
                    'diameter': 100,
                    'flow_threshold': 0.4,
                    'cellprob_threshold': 0.0
                }
            },
            {
                'name': 'cyto2',
                'params': {
                    'model_type': 'cyto2',
                    'diameter': 100,
                    'flow_threshold': 0.4,
                    'cellprob_threshold': 0.0
                }
            },
            {
                'name': 'cyto_specialized',
                'params': {
                    'model_type': 'cyto',
                    'diameter': 70,
                    'flow_threshold': 0.8,
                    'cellprob_threshold': -1.0,
                    'min_size': 30
                }
            },
            {
                'name': 'nuclei',
                'params': {
                    'model_type': 'nuclei',
                    'diameter': 50,
                    'flow_threshold': 0.4,
                    'cellprob_threshold': 0.0
                }
            }
        ]
        
        # Process with each model
        for idx, model_config in enumerate(models_to_test):
            print(f"\nTesting {model_config['name']} model...")
            
            # Create new model instance
            test_model = models.Cellpose(
                model_type=model_config['params']['model_type'],
                gpu=self.model.gpu)
            
            # Normalize image
            image_norm = (nad_green - nad_green.min()) / (nad_green.max() - nad_green.min())
            image_8bit = (image_norm * 255).astype('uint8')
            
            # Run segmentation
            try:
                masks, _, _, _ = test_model.eval(
                    image_8bit,
                    diameter=model_config['params']['diameter'],
                    channels=[0,0],
                    flow_threshold=model_config['params']['flow_threshold'],
                    cellprob_threshold=model_config['params']['cellprob_threshold'],
                    min_size=model_config['params'].get('min_size', 0),
                    do_3D=False
                )
                
                # Plot results
                ax = fig.add_subplot(gs[1:, idx])
                
                # Create overlay
                boundaries = segmentation.find_boundaries(masks)
                overlay = np.stack([image_norm]*3, axis=-1)
                overlay[boundaries] = [1, 0, 0]  # Red boundaries
                
                ax.imshow(overlay)
                ax.set_title(f"{model_config['name']}\n({len(np.unique(masks))-1} cells)")
                
                # Add cell statistics
                cell_sizes = []
                cell_intensities = []
                for region in measure.regionprops(masks, intensity_image=image_norm):
                    cell_sizes.append(region.area)
                    cell_intensities.append(region.mean_intensity)
                
                if len(cell_sizes) > 0:
                    stats_text = (
                        f"Cell stats:\n"
                        f"Size: {np.mean(cell_sizes):.0f}±{np.std(cell_sizes):.0f} px\n"
                        f"Int: {np.mean(cell_intensities):.2f}±{np.std(cell_intensities):.2f}"
                    )
                    ax.text(0.02, 0.98, stats_text,
                           transform=ax.transAxes,
                           verticalalignment='top',
                           color='white',
                           bbox=dict(facecolor='black', alpha=0.7))
                
            except Exception as e:
                print(f"Error with {model_config['name']}: {str(e)}")
                continue
        
        plt.tight_layout()
        return fig
    
    def analyze_results(self, output_dir):
        """Comprehensive analysis of results with exploration"""
        output_dir = Path(output_dir)
        results_file = output_dir / "results.csv"
        
        if not results_file.exists():
            raise ValueError(f"Results file not found: {results_file}")
        
        # Load both summary and cell-level data
        results = pd.read_csv(results_file)  # Summary data
        nad_cells = pd.read_csv(output_dir / "nad_cells.csv")  # NAD cell data
        venus_cells = pd.read_csv(output_dir / "venus_cells.csv")  # Venus cell data
        
        # 1. Summary statistics
        print("\n=== Summary Statistics ===")
        summary_stats = pd.DataFrame({
            'Total Positions': len(results),
            'Average NAD cells per position': f"{results['nad_cells'].mean():.1f} ± {results['nad_cells'].std():.1f}",
            'Average Venus cells per position': f"{results['venus_cells'].mean():.1f} ± {results['venus_cells'].std():.1f}",
            'NAD Ratio (mean ± std)': f"{results['nad_mean_ratio'].mean():.3f} ± {results['nad_mean_ratio'].std():.3f}",
            'Venus Ratio (mean ± std)': f"{results['venus_mean_ratio'].mean():.3f} ± {results['venus_mean_ratio'].std():.3f}"
        }, index=[0])
        print(summary_stats)
        
        # 2. Cell-level analysis
        print("\n=== Cell-level Analysis ===")
        print("\nNAD Sensor Cells:")
        print(f"Total cells: {len(nad_cells)}")
        print(f"Area: {nad_cells['area'].mean():.1f} ± {nad_cells['area'].std():.1f}")
        print(f"Ratio: {nad_cells['ratio'].mean():.3f} �� {nad_cells['ratio'].std():.3f}")
        
        print("\nVenus Cells:")
        print(f"Total cells: {len(venus_cells)}")
        print(f"Area: {venus_cells['area'].mean():.1f} ± {venus_cells['area'].std():.1f}")
        print(f"Ratio: {venus_cells['ratio'].mean():.3f} ± {venus_cells['ratio'].std():.3f}")
        
        # 3. Correlation analysis
        plt.figure(figsize=(15, 5))
        
        # NAD cell size vs ratio
        plt.subplot(131)
        plt.scatter(nad_cells['area'], nad_cells['ratio'], alpha=0.5, label='NAD')
        plt.xlabel('Cell Area')
        plt.ylabel('Ratio')
        plt.title('NAD: Cell Size vs Ratio')
        
        # Venus cell size vs ratio
        plt.subplot(132)
        plt.scatter(venus_cells['area'], venus_cells['ratio'], alpha=0.5, label='Venus')
        plt.xlabel('Cell Area')
        plt.ylabel('Ratio')
        plt.title('Venus: Cell Size vs Ratio')
        
        # Region-level normalized ratios
        plt.subplot(133)
        sns.boxplot(data=results, x='treatment_group', y='normalized_ratio')
        plt.xticks(rotation=45)
        plt.title('Normalized Ratio by Group')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'exploratory_analysis.png')
        plt.close()
        
        # 4. Region-level analysis - keep position information
        region_data = results.copy()  # Keep all columns including position
        region_stats = region_data.groupby(['treatment_group']).agg({
            'normalized_ratio': ['mean', 'std', 'count'],
            'nad_cells': 'mean',
            'venus_cells': 'mean',
            'nad_mean_ratio': ['mean', 'std'],
            'venus_mean_ratio': ['mean', 'std']
        }).reset_index()
        
        print("\n=== Region-level Variation ===")
        for group in region_stats['treatment_group'].unique():
            group_data = region_stats[region_stats['treatment_group'] == group]
            print(f"\n{group}:")
            print(f"Positions: {group_data[('normalized_ratio', 'count')].iloc[0]}")
            print(f"CV of normalized ratio: {group_data[('normalized_ratio', 'std')].iloc[0] / group_data[('normalized_ratio', 'mean')].iloc[0]:.3f}")
            print(f"CV of NAD ratio: {group_data[('nad_mean_ratio', 'std')].iloc[0] / group_data[('nad_mean_ratio', 'mean')].iloc[0]:.3f}")
            print(f"CV of Venus ratio: {group_data[('venus_mean_ratio', 'std')].iloc[0] / group_data[('venus_mean_ratio', 'mean')].iloc[0]:.3f}")
        
        # 5. Filter outliers and generate dose-response
        filtered_results = self.filter_region_outliers(region_data)
        self.plot_dose_response_comparison(region_data, filtered_results, output_dir)
        
        # Add cell parameter analysis
        self.analyze_cell_parameters(output_dir)
        
        # Add intensity relationship analysis
        self.analyze_intensity_relationships(output_dir)
    
    def analyze_cell_parameters(self, output_dir):
        """Analyze relationships between cell parameters and ratios"""
        output_dir = Path(output_dir)
        nad_cells = pd.read_csv(output_dir / "nad_cells.csv")
        venus_cells = pd.read_csv(output_dir / "venus_cells.csv")
        
        # Create figure for NAD sensor cells
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NAD Sensor Cell Parameters vs Ratio', fontsize=16)
        
        # Area vs Ratio
        sns.scatterplot(data=nad_cells, x='area', y='ratio', ax=axes[0,0], alpha=0.5)
        axes[0,0].set_title('Area vs Ratio')
        
        # Perimeter vs Ratio
        sns.scatterplot(data=nad_cells, x='perimeter', y='ratio', ax=axes[0,1], alpha=0.5)
        axes[0,1].set_title('Perimeter vs Ratio')
        
        # Eccentricity vs Ratio
        sns.scatterplot(data=nad_cells, x='eccentricity', y='ratio', ax=axes[0,2], alpha=0.5)
        axes[0,2].set_title('Eccentricity vs Ratio')
        
        # Green intensity vs Ratio
        sns.scatterplot(data=nad_cells, x='green', y='ratio', ax=axes[1,0], alpha=0.5)
        axes[1,0].set_title('Green Intensity vs Ratio')
        
        # Blue intensity vs Ratio
        sns.scatterplot(data=nad_cells, x='blue', y='ratio', ax=axes[1,1], alpha=0.5)
        axes[1,1].set_title('Blue Intensity vs Ratio')
        
        # Distribution of ratios
        sns.boxplot(data=nad_cells, x='treatment_group', y='ratio', ax=axes[1,2])
        axes[1,2].set_title('Ratio Distribution by Group')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'nad_cell_parameters.png')
        plt.close()
        
        # Create correlation matrix
        corr_matrix = nad_cells[['area', 'perimeter', 'eccentricity', 'green', 'blue', 'ratio']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('NAD Sensor Cell Parameter Correlations')
        plt.tight_layout()
        plt.savefig(output_dir / 'nad_correlations.png')
        plt.close()
        
        # Print summary statistics
        print("\nNAD Sensor Cell Parameter Statistics:")
        print("\nCorrelations with Ratio:")
        for param in ['area', 'perimeter', 'eccentricity', 'green', 'blue']:
            corr = nad_cells[param].corr(nad_cells['ratio'])
            print(f"{param}: {corr:.3f}")
        
        # Do the same for Venus cells
        # ... (similar code for Venus cells)
    
    def filter_region_outliers(self, region_data, n_std=2):
        """Filter outliers at the region level
        
        Args:
            region_data (pd.DataFrame): Region-level data
            n_std (float): Number of standard deviations for filtering
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = []
        
        for group in region_data['treatment_group'].unique():
            group_data = region_data[region_data['treatment_group'] == group].copy()
            
            # Calculate mean and std for normalized ratio
            mean_ratio = group_data['normalized_ratio'].mean()
            std_ratio = group_data['normalized_ratio'].std()
            
            # Filter based on normalized ratio
            mask = (
                (group_data['normalized_ratio'] > mean_ratio - n_std * std_ratio) &
                (group_data['normalized_ratio'] < mean_ratio + n_std * std_ratio)
            )
            
            # Print filtering info
            outliers = group_data[~mask]
            if len(outliers) > 0:
                print(f"\nOutliers in {group}:")
                print(f"Mean ratio: {mean_ratio:.3f}")
                print(f"Std ratio: {std_ratio:.3f}")
                print("Outlier positions and values:")
                for _, row in outliers.iterrows():
                    print(f"Position {row['position']}: {row['normalized_ratio']:.3f}")
            
            filtered_data.append(group_data[mask])
        
        filtered_data = pd.concat(filtered_data)
        
        print(f"\nFiltering summary:")
        print(f"Original positions: {len(region_data)}")
        print(f"Filtered positions: {len(filtered_data)}")
        print(f"Removed {len(region_data) - len(filtered_data)} outlier positions")
        
        # Calculate CV improvement
        def calculate_cv(data, group):
            values = data[data['treatment_group'] == group]['normalized_ratio']
            return values.std() / values.mean()
        
        print("\nCV comparison (before → after):")
        for group in region_data['treatment_group'].unique():
            cv_before = calculate_cv(region_data, group)
            cv_after = calculate_cv(filtered_data, group)
            print(f"{group}: {cv_before:.3f} → {cv_after:.3f}")
        
        return filtered_data
    
    def analyze_intensity_relationships(self, output_dir):
        """Analyze relationships between intensities and ratios by concentration"""
        output_dir = Path(output_dir)
        nad_cells = pd.read_csv(output_dir / "nad_cells.csv")
        venus_cells = pd.read_csv(output_dir / "venus_cells.csv")
        
        # Create plots for NAD sensor cells
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('NAD Sensor Intensity Relationships by Concentration', fontsize=16)
        
        # Get all concentrations
        concentrations = sorted(nad_cells['treatment_group'].unique())
        
        # Create plots for first 8 concentrations
        for i, conc in enumerate(concentrations):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            # Get data for this concentration
            conc_data = nad_cells[nad_cells['treatment_group'] == conc]
            
            # Create scatter plot with multiple relationships
            scatter1 = ax.scatter(conc_data['green'], conc_data['ratio'], 
                                alpha=0.5, label='Green vs Ratio', color='green')
            
            # Add second y-axis for blue channel
            ax2 = ax.twinx()
            scatter2 = ax2.scatter(conc_data['green'], conc_data['blue'], 
                                 alpha=0.5, label='Green vs Blue', color='blue')
            
            # Add correlations
            corr_gr = np.corrcoef(conc_data['green'], conc_data['ratio'])[0,1]
            corr_gb = np.corrcoef(conc_data['green'], conc_data['blue'])[0,1]
            corr_br = np.corrcoef(conc_data['blue'], conc_data['ratio'])[0,1]
            
            ax.text(0.05, 0.95, 
                    f'Correlations:\nGreen-Ratio: {corr_gr:.2f}\n'
                    f'Green-Blue: {corr_gb:.2f}\nBlue-Ratio: {corr_br:.2f}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Green Intensity')
            ax.set_ylabel('Ratio', color='green')
            ax2.set_ylabel('Blue Intensity', color='blue')
            ax.set_title(f'{conc}\n(n={len(conc_data)})')
            
            # Adjust tick colors
            ax.tick_params(axis='y', labelcolor='green')
            ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'nad_intensity_relationships.png')
        plt.close()
        
        # Create separate plots for specific relationships
        relationships = [
            ('green', 'ratio', 'Green Intensity vs Ratio'),
            ('blue', 'ratio', 'Blue Intensity vs Ratio'),
            ('green', 'blue', 'Green vs Blue Intensity')
        ]
        
        for x_var, y_var, title in relationships:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'NAD Sensor: {title} by Concentration', fontsize=16)
            
            for i, conc in enumerate(concentrations):
                row = i // 4
                col = i % 4
                ax = axes[row, col]
                
                conc_data = nad_cells[nad_cells['treatment_group'] == conc]
                
                # Create scatter plot
                sns.scatterplot(data=conc_data, x=x_var, y=y_var, ax=ax, alpha=0.5)
                
                # Add correlation
                corr = np.corrcoef(conc_data[x_var], conc_data[y_var])[0,1]
                ax.text(0.05, 0.95, f'Correlation: {corr:.2f}',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.8))
                
                ax.set_title(f'{conc}\n(n={len(conc_data)})')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'nad_{x_var}_{y_var}_by_conc.png')
            plt.close()
        
        # Do the same for Venus cells
        # ... (similar code for Venus cells)

def main():
    parser = argparse.ArgumentParser(description='NAD Sensor Analysis')
    parser.add_argument('--data-folder', 
                      help='Path to folder containing ND2 files')
    parser.add_argument('--output-dir', default='results', 
                      help='Directory to save results (default: results)')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU acceleration for cell segmentation')
    parser.add_argument('--model-type', default='cyto',
                      help='Cellpose model type (default: cyto)')
    parser.add_argument('--log-file', default='analysis.log',
                      help='Path to log file (default: analysis.log)')
    parser.add_argument('--test-position', type=int, default=None,
                      help='Process only this position for testing')
    parser.add_argument('--test-loading', action='store_true',
                      help='Test image loading with visualization')
    parser.add_argument('--test-segmentation', action='store_true',
                      help='Test cell segmentation with visualization')
    parser.add_argument('--test-ratio', action='store_true',
                      help='Test ratio calculation with visualization')
    parser.add_argument('--test-structure', action='store_true',
                      help='Test ND2 file structure')
    parser.add_argument('--analyze-results', type=str,
                      help='Analyze existing results.csv file')
    parser.add_argument('--compare-models', action='store_true',
                      help='Compare different Cellpose models')
    parser.add_argument('--analyze', type=str,
                      help='Analyze results in specified output directory')
    
    args = parser.parse_args()
    
    analyzer = NADAnalyzer(model_type=args.model_type, gpu=args.gpu, 
                          log_file=args.log_file)
    
    try:
        # Analysis modes that don't need data folder
        if args.analyze:
            analyzer.analyze_results(args.analyze)
            return
            
        if args.analyze_results:
            analyzer.analyze_dose_response(args.analyze_results, args.output_dir)
            return
        
        # Modes that need data folder
        if not args.data_folder:
            parser.error("--data-folder is required for processing or testing")
            
        nad_file, venus_file = analyzer.find_nd2_files(args.data_folder)
        
        if args.compare_models:
            # Compare different segmentation models
            pos = args.test_position if args.test_position is not None else 0
            fig = analyzer.test_segmentation_models(nad_file, venus_file, pos)
            fig.savefig('model_comparison.png')
            plt.close(fig)
            print(f"Saved model comparison to model_comparison.png")
            return
            
        if args.test_loading:
            # Test image loading
            pos = args.test_position if args.test_position is not None else 0
            fig = analyzer.test_image_loading(nad_file, venus_file, pos)
            fig.savefig('test_loading.png')
            print(f"Saved test images to test_loading.png")
            return
            
        if args.test_position is not None:
            # Test mode - process only one position
            results = analyzer.process_position(nad_file, venus_file, args.test_position)
            print(f"Processed position {args.test_position}:")
            
            # Print summary of results
            if results['summary'].empty:
                print("No cells found")
            else:
                print(f"\nSummary:")
                print(results['summary'])
                
                print(f"\nNAD cells found: {len(results['nad_cells'])}")
                if len(results['nad_cells']) > 0:
                    print("\nFirst few NAD cells:")
                    print(results['nad_cells'].head())
                
                print(f"\nVenus cells found: {len(results['venus_cells'])}")
                if len(results['venus_cells']) > 0:
                    print("\nFirst few Venus cells:")
                    print(results['venus_cells'].head())
            
            if args.test_segmentation:
                # Test segmentation
                fig = analyzer.test_segmentation(nad_file, venus_file, args.test_position)
                fig.savefig('test_segmentation.png')
                plt.close(fig)
                print(f"\nSaved segmentation test to test_segmentation.png")
                return
            
        else:
            # Process all positions
            analyzer.process_all_positions(nad_file, venus_file, args.output_dir)
            
        if args.test_ratio:
            # Test ratio calculation
            pos = args.test_position if args.test_position is not None else 0
            fig = analyzer.test_ratio_calculation(nad_file, venus_file, pos)
            fig.savefig('test_ratio.png')
            print(f"Saved ratio test to test_ratio.png")
            return
            
        if args.test_structure:
            print("Testing NAD sensor file structure...")
            analyzer.test_file_structure(nad_file)
            print("\nTesting Venus file structure...")
            analyzer.test_file_structure(venus_file)
            return
            
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()