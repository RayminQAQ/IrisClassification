# Iris Dataset Classification with PyTorch

This repository contains two projects focused on classifying the Iris dataset using neural networks built with PyTorch. The classification models include:

1. **Binary Classification** - Classifies whether the species is Setosa or Non-Setosa.
2. **Multi-class Classification** - Classifies among the three species of the Iris dataset.

## Project Overview

- **Binary Classification** (`IrisBinaryClassification.py`): This project transforms the Iris dataset into a binary classification problem, classifying whether a sample is of the Setosa species or not. It utilizes a simple feedforward neural network and evaluates model performance using accuracy, precision, recall, and AUC metrics.
  
- **Multi-class Classification** (`IrisMultiClassification.py`): This project addresses the multi-class nature of the Iris dataset, classifying samples into one of the three species (Setosa, Versicolor, or Virginica). The model's performance is measured by accuracy, precision, recall, F1 score, and AUC across multiple epochs. The results are plotted using custom visualization functions provided in the `utils.py` script.

## Requirements

To run the projects, you need the following libraries installed:

- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install the required libraries by running:
```bash
pip install torch numpy scikit-learn matplotlib
```

## How to Run
Binary Classification
To run the binary classification model, execute the following command:

```bash
python IrisBinaryClassification.py
```
The model will output accuracy, precision, recall, and AUC after training for a defined number of epochs.

Multi-class Classification
To run the multi-class classification model and plot the performance metrics:

```bash
python IrisMultiClassification.py
```
The training will run for multiple epochs, and plots will be saved to the Result directory.

## Results
After running the multi-class classification model, you will get saved plots for various performance metrics like Accuracy, Precision, Recall, F1 Score, and AUC over epochs.

## Acknowledgements
The Iris dataset is a well-known dataset in machine learning and is included as part of the sklearn datasets library.
PyTorch is used as the primary deep learning framework for this project.
