# SPECTROGRAM BASED CNN APPLIED TO PARKINSON DISEASE DETECTION 

The detection of Parkinson's Disease (PD) through speech analysis represents a significant advancement in non-invasive diagnostic techniques. Speech impairment, specifically dysarthria, often manifests as one of the earliest indicators of PD, frequently preceding the onset of motor symptoms. This early manifestation makes speech analysis particularly valuable as a potential early screening tool.

The developed system leverages deep learning and signal processing techniques to analyze subtle speech patterns characteristic of PD. At its core, the system utilizes Mel-frequency analysis, which transforms raw audio signals into spectral representations that closely align with human auditory perception. This transformation is coupled with advanced Convolutional Neural Networks (CNNs) that excel at pattern recognition in spectral data. The architecture is further enhanced by attention mechanisms, which enable the model to focus on the most relevant acoustic features while suppressing less informative ones.

## Project Structure

- `train.py`: Script to train the model.
- `evaluate.py`: Script to evaluate the trained model and generate visualizations.
- `setup.sh`: Shell script to set up the environment and dependencies.
- `model.py`: Contains the model architecture.
- `test_torch.py`: Script to verify PyTorch installation.
- `requirements.txt`: List of Python dependencies.

## Setup Instructions

### Prerequisites

- Ensure you have [Homebrew](https://brew.sh/) installed on your system.

### Step 1: Set Up the Environment

1. **Run the setup script**: This will install Python 3.11, create a virtual environment, and install all necessary dependencies.
   ```bash
   ./setup.sh
   ```

2. **Activate the virtual environment**: The setup script will automatically activate the virtual environment. If you need to activate it manually, use:
   ```bash
   source venv/bin/activate
   ```

### Step 2: Prepare Your Data

- Place your audio files in the following directories:
  - `sick/` for Parkinson's patients' audio files for the first dataset.
  - `healthy/` for control group audio files for the first dataset.
  - `sick2/` for Parkinson's patients' audio files for the second dataset.
  - `healthy2/` for control group audio files for the second dataset.

### Step 3: Train the Model

1. **Run the training script**: This will train the model using the audio data.
   ```bash
   python train.py
   ```

2. **Monitor the training**: The script will output training and validation metrics, and save the best model as `best_model.pth`.

### Step 4: Evaluate the Model

1. **Run the evaluation script**: This will evaluate the trained model and generate various visualizations.
   ```bash
   python evaluate.py
   ```

2. **Check the results**: The script will generate and save the following visualizations:
   - `confusion_matrix.png`
   - `roc_curve.png`
   - `pr_curve.png`
   - `confidence_dist.png`

## Additional Information

- **Evaluation Metrics**: The evaluation script provides detailed metrics including accuracy, ROC AUC, and class-specific metrics.

## Troubleshooting

- If you encounter issues with PyTorch or torchaudio, ensure that your system supports the required backend (e.g., MPS for macOS).
- Verify that all dependencies are correctly installed by checking the output of `test_torch.py`.

## Acknowledgments

- This project uses the [timm](https://github.com/rwightman/pytorch-image-models) library for model architecture.
