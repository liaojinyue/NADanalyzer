# NAD Sensor Quantification Project

## Project Overview
This project analyzes microscopy data to measure NAD sensor activity in relation to different drug treatments, using venus protein as a normalization control.

## Data Structure
For each experiment, we have two ND2 microscopy files:
1. NAD sensor measurements
2. Venus protein measurements

### Image Specifications
- **Resolution**: 2048 x 2048 pixels
- **Channels per File**: 
  - Blue channel (Ex405)
  - Green channel (Ex488)
  - TD (transmission) - will not be used
- **Fields of View**: 48 positions per file
- **Pixel Size**: 0.429606 microns

### Experimental Design
The 48 fields of view are organized into 8 treatment groups (6 images per group):
- Positions 1-6: Drug concentration 1
- Positions 7-12: Drug concentration 2 (10x of concentration 1)
- Positions 13-18: Drug concentration 3 (10x of concentration 2)
- And so on...

## Analysis Goals
1. **Signal Quantification**
   - Measure NAD sensor signal intensity (green))
   - Measure Venus protein signal intensity (blue)
   - Calculate the ratio (NAD sensor/Venus) for normalization

2. **Data Analysis**
   - Analyze the relationship between drug concentrations and normalized NAD sensor signals
   - Generate visualization of dose-response relationships

## Technical Requirements
- Python environment (version 3.10)
- Previous analysis pipeline available in `pipeline.cppipe`
- Script to be run in cursor

## Expected Outputs
1. Quantified signal measurements for each channel
2. Normalized ratios for each field of view
3. Statistical analysis of dose-response relationships
4. Visualization of results