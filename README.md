# Deep Learning for Phase Unwrapping in Quantitative Phase Imaging

This repository contains the complete implementation of deep learning methods for phase unwrapping in quantitative phase imaging, developed as part of a bachelor's thesis. The project implements and compares multiple approaches for solving the challenging problem of phase unwrapping in microscopy and holographic imaging systems.

## Project Overview

Phase unwrapping is a fundamental challenge in quantitative phase imaging where wrapped phase maps (limited to [-π, π]) need to be reconstructed to their true unwrapped values. This project implements two main deep learning approaches:

- **Direct Regression (DRG)**: Direct prediction of unwrapped phase values using U-Net architecture
- **Deep Wrap Count (DWC)**: Classification-based approach that predicts integer multiples of 2π to reconstruct unwrapped phase

## Key Features

- **Dynamic Dataset Generation**: Real-time simulation of phase data with configurable parameters
- **Multiple Neural Network Architectures**: U-Net with various encoders (ResNet, EfficientNet, etc.)
- **Comprehensive Evaluation**: PSNR, SSIM, MAE, and custom gradient difference loss (GDL)
- **Data Augmentation**: Geometric and pixel-level augmentations for improved generalization
- **Comparison with Traditional Methods**: L2 norm and Goldstein phase unwrapping algorithms
- **QDF File Support**: Integration with Telight Q-PHASE microscopy data format

## Repository Structure

### Core Training and Evaluation Scripts

#### Direct Regression (DRG) Methods
- `drg_best_model.py` - Training script for the best-performing DRG model
- `drg_optimalization.py` - Hyperparameter optimization for DRG models  
- `drg_evaluate.py` - Evaluation script for trained DRG models
- `drg_test.py` - Testing and visualization for DRG models

#### Deep Wrap Count (DWC) Methods
- `dwc_best_model.py` - Training script for the best-performing DWC model
- `dwc_optimalization.py` - Hyperparameter optimization for DWC models
- `dwc_evaluate.py` - Evaluation script for trained DWC models
- `dwc_test.py` - Testing and visualization for DWC models

### Dataset Management
- `split_dataset_dynamic_create.py` - Advanced dataset creation with stratified splitting
- `dataset_names_analyzer.py` - Analysis and filtering of dataset files by naming patterns
- `low_qual_deleter.py` - Quality-based dataset filtering and train/val/test splitting
- `experimental_simulation_32bit.py` - 32-bit float simulation data generation

### Data Processing and Utilities
- `cropper_resizer.py` - Image preprocessing (cropping and resizing)
- `channel_spliter.py` - QDF channel extraction and processing
- `cchm.py` & `cchm_unwrapped_substracted_create.py` - CCHM (Compensated phase) processing

### Traditional Methods and Comparison
- `PUDIP.py` - PUDIP (Phase Unwrapping via Deep Image Prior) implementation
- `final_comparison.py` - Comprehensive evaluation comparing all methods
- `PhaseUnwrap2D.m`, `Wrap.m`, `LocateResidues.m`, `UnwrapAroundCuts.m` - MATLAB implementations of traditional algorithms

### QDF File Processing
- `QDF.py` - Reader for Telight Q-PHASE QDF microscopy files

### Network Architecture
- `u_net_tiff_final.py` - U-Net implementation optimized for phase data

## Quick Start

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch
pip install scikit-image tifffile
pip install matplotlib numpy tqdm
```

### Training a Model

#### Direct Regression (DRG)
```python
python drg_best_model.py
```

#### Deep Wrap Count (DWC)
```python
python dwc_best_model.py
```

### Dataset Creation

Create a stratified dataset with dynamic simulation:
```python
python split_dataset_dynamic_create.py
```

### Evaluation

Run comprehensive evaluation across all methods:
```python
python final_comparison.py
```

## Model Architectures

### Direct Regression (DRG)
- **Architecture**: U-Net with configurable encoders
- **Input**: Wrapped phase images (normalized)
- **Output**: Unwrapped phase values
- **Loss Functions**: MAE, MSE, Gradient Difference Loss (GDL)

### Deep Wrap Count (DWC)
- **Architecture**: U-Net for semantic segmentation
- **Input**: Wrapped phase images
- **Output**: K-value classification (integer multiples of 2π)
- **Loss**: Weighted cross-entropy with edge enhancement
- **Reconstruction**: `unwrapped = wrapped + 2π × k_predicted`

## Key Innovations

1. **Dynamic Training Data Generation**: Real-time synthesis of training pairs with configurable simulation parameters
2. **Stratified Dataset Splitting**: Ensures balanced representation across different imaging conditions
3. **Edge-Weighted Loss**: Enhanced learning at phase discontinuities
4. **Multi-Scale Evaluation**: Assessment across different image scales and quality levels
5. **Comprehensive Benchmarking**: Direct comparison with traditional and deep learning methods

## Dataset Configuration

The project supports flexible dataset configuration through parameter ranges:

```python
SIMULATION_PARAM_RANGES = {
    "n_strips_param": (7, 8),                      # Fringe density
    "original_image_influence_param": (0.3, 0.5),  # Object influence
    "phase_noise_std_param": (0.024, 0.039),       # Noise level
    "smooth_original_image_sigma_param": (0.2, 0.5), # Smoothing
    "poly_scale_param": (0.02, 0.1),               # Background complexity
    "CURVATURE_AMPLITUDE_param": (1.4, 2.0),       # Phase curvature
    "background_offset_d_param": (-24.8, -6.8),    # Background offset
    "tilt_angle_deg_param": (-5.0, 17.0)           # Tilt angle
}
```

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Primary metric for reconstruction accuracy
- **PSNR (Peak Signal-to-Noise Ratio)**: Image quality assessment
- **SSIM (Structural Similarity Index)**: Perceptual quality measure
- **GDL (Gradient Difference Loss)**: Custom metric for edge preservation
- **K-Label Accuracy**: Classification accuracy for DWC methods

## Data Formats

### Supported Input Formats
- **TIFF**: 32-bit float phase images
- **QDF**: Telight Q-PHASE microscopy format
- **MAT**: MATLAB data files

### Dataset Structure
```
dataset/
├── images/           # Wrapped phase images
├── labels/           # Ground truth unwrapped phase
├── train/           # Training split
├── val/             # Validation split  
└── test/            # Test split
```

## Configuration

Training parameters can be configured in the respective `*_best_model.py` files:

```python
# Model Configuration
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
NUM_EPOCHS = 120
TARGET_IMG_SIZE = (512, 512)

# Augmentation
AUGMENTATION_STRENGTH_TRAIN = "medium"
```

## Results and Performance

The project demonstrates state-of-the-art performance in phase unwrapping with:
- Superior accuracy compared to traditional methods
- Real-time inference capability
- Robust performance across different imaging conditions
- Effective handling of noise and artifacts

## Applications

This implementation is suitable for:
- **Digital Holographic Microscopy (DHM)**
- **Quantitative Phase Imaging (QPI)**
- **Interferometric Microscopy**
- **Synthetic Aperture Radar (SAR) interferometry**
- **Medical imaging applications**

## Citation

If you use this code in your research, please cite:

```bibtex
@thesis{phase_unwrapping_dl,
  title={Deep Learning for Phase Unwrapping in Quantitative Phase Imaging},
  author={[Juraj Bendík]},
  year={2025},
  school={[Brno University of Technology]},
  type={Bachelor's Thesis}
}
```

## Acknowledgments

- PUDIP (Phase Unwrapping via Deep Image Prior) for baseline comparison
- Segmentation Models PyTorch for neural network architectures
- Telight for Q-PHASE microscopy system integration
- Scientific Python ecosystem (NumPy, SciPy, scikit-image)
