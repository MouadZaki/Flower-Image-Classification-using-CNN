
## Features
- Custom CNN architecture with 4 convolutional blocks
- Comprehensive exploratory data analysis and visualization
- Data augmentation with rotation, flips, and color jitter
- Batch normalization and dropout for regularization
- Learning rate scheduling and early stopping
- Detailed model evaluation with confusion matrix
- GPU acceleration support (automatic detection)

## Project Structure
- `Flower_Classification_CNN_Project.ipynb`: Complete Jupyter notebook with full pipeline
- `Flower_Classification_CNN_Project_Clean.ipynb`: Cleaned version without emojis
- `data/flower_photos/`: Dataset (daisy, dandelion, roses, sunflowers, tulips)
- `checkpoints/`: Saved model checkpoints
- `outputs/`: Training visualizations and evaluation results

Example dataset layout:
```
data/flower_photos/
	daisy/
		100080576_f52e8ee070_n.jpg
		...
	dandelion/
		...
	roses/
		...
	sunflowers/
		...
	tulips/
		...
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (recommended) or CPU-only

### Install Dependencies
```bash
# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Additional dependencies
pip install numpy pandas matplotlib seaborn scikit-learn pillow opencv-python jupyter
```

### Dataset Setup
Download the flower dataset and organize as follows:
```
data/flower_photos/
    daisy/
    dandelion/
    roses/
    sunflowers/
    tulips/
```

## Usage

### 1. Run the Complete Pipeline
Open and run the Jupyter notebook:
```bash
jupyter notebook Flower_Classification_CNN_Project_Clean.ipynb
```

The notebook includes:
- **Data Exploration**: Class distribution, image dimensions analysis
- **Visualization**: Sample images, statistical plots
- **Model Training**: CNN architecture with 27M parameters
- **Evaluation**: Accuracy metrics, confusion matrix, classification report

### 2. Key Configuration
- **Image Size**: 224x224 pixels
- **Batch Size**: 32 (adjust based on GPU memory)
- **Classes**: 5 flower types
- **Training**: Up to 50 epochs with early stopping

### 3. Model Architecture
- 4 Convolutional blocks (32→64→128→256 filters)
- Batch normalization after each conv layer
- Max pooling and dropout for regularization
- 2 Fully connected layers (512→256→5)
- Total parameters: ~27 million

## Model Performance
- **Validation Accuracy**: ~85-90% (varies by training)
- **Training Time**: ~30-60 minutes on GPU, ~2-4 hours on CPU
- **Memory Usage**: ~4GB GPU memory for batch size 32

## Troubleshooting

### GPU Issues
```python
# Check CUDA availability
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')
```

### Common Problems
- **Out of Memory**: Reduce `BATCH_SIZE` in notebook
- **Slow Training**: Ensure CUDA PyTorch installation
- **Dataset Not Found**: Verify `data/flower_photos/` structure

## Technical Details
- **Framework**: PyTorch
- **Normalization**: ImageNet statistics
- **Data Augmentation**: Rotation, flips, color jitter
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (0.25-0.5) and early stopping

## License
This project is intended for educational and research purposes. Feel free to use and modify for learning purposes.
