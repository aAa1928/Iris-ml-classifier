# Iris ML Classifier

A machine learning classifier built with PyTorch to identify Iris flower species.

## Overview

This project implements a neural network using PyTorch to classify Iris flowers into their respective species based on their measurements. The classic Iris dataset is used for training and testing the model. The neural network achieves high accuracy in distinguishing between three Iris species using four input features.

## Features

- PyTorch neural network implementation with 3 layers
- Training visualization with loss plot
- Automatic model saving after training
- Real-time training progress monitoring
- Training/testing split with 80/20 ratio
- Classification of three Iris species:
  - Setosa
  - Versicolor
  - Virginica

## Model Architecture

The neural network consists of:

- Input layer: 4 features (sepal length, sepal width, petal length, petal width)
- Hidden layer 1: 8 neurons with ReLU activation
- Hidden layer 2: 9 neurons with ReLU activation
- Output layer: 3 neurons (one for each species)

## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib

## Installation

1. Create and activate virtual environment:

```bash
py -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the classifier:

```bash
python main.py
```

2. The program will:
   - Load the Iris dataset
   - Train the model for 200 epochs
   - Display training progress and loss
   - Show a loss plot
   - Evaluate the model on test data
   - Save the trained model as 'iris_model.pth'

## Project Structure

```
Iris-ml-classifier/
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
├── main.py           # Main training script
├── iris_model.pth    # Saved model (generated after training)
└── loss_plot.png     # Training visualization (generated after training)
```

## Model Performance

The model typically achieves a 96% accuracy on the test set, with the exact performance varying due to the random nature of the training process.

## License

MIT License

## Acknowledgments

- The Iris dataset is from the UCI Machine Learning Repository
