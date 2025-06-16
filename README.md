# Handwritten Tifinagh Character Recognition using a Neural Network from Scratch
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange)

This repository contains a Python implementation of a Multilayer Perceptron (MLP) neural network, built from scratch using only NumPy, to classify handwritten Tifinagh characters from the AMHCD dataset.

The final model achieves **95% accuracy** on the unseen test set.

***

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Key Features](#key-features)
3.  [Dataset](#dataset)
4.  [Methodology](#methodology)
5.  [Results](#results)
6.  [Installation and Usage](#installation-and-usage)
7.  [Future Work](#future-work)
8.  [License](#license)
9.  [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to develop a robust solution for the automatic classification of handwritten characters for the Tifinagh script, an under-resourced language used by Amazigh communities. The project involves building, training, and evaluating a neural network from the ground up, implementing advanced techniques to ensure high performance and good generalization. The entire data processing pipeline, model architecture, and training loop are custom-built.

---

## Key Features

- **Neural Network from Scratch:** The MLP is implemented using only NumPy, providing a deep understanding of the underlying mechanics.
- **Advanced Optimization:** Uses the **Adam optimizer** for faster and more stable convergence.
- **Regularization:** Implements **L2 Regularization** (Weight Decay) to prevent overfitting and improve model generalization.
- **Modern Activation Function:** Employs **Leaky ReLU** in the hidden layers to mitigate the "dying ReLU" problem.
- **Data Augmentation:** Increases the training dataset's diversity through random rotations and translations, making the model more robust.
- **Hyperparameter Tuning:** Includes a systematic search to find the optimal learning rate, batch size, and regularization strength.
- **Comprehensive Evaluation:** The model's performance is thoroughly assessed using a classification report, loss/accuracy curves, and a confusion matrix.

---

## Dataset

The model is trained on the **Amazigh Handwritten Character Database (AMHCD)**.
- **Content:** 28,182 grayscale images of 33 Tifinagh characters.
- **Preprocessing:** The original 64x64 images are resized to 32x32 pixels, normalized to a [0, 1] scale, and flattened into 1024-dimensional vectors.
- **Source:** [AMHCD on Kaggle](https://www.kaggle.com/datasets/benaddym/amazigh-handwritten-character-database-amhcd)

---

## Methodology

The core of the project is a `MultiClassNeuralNetwork` class with the following architecture:

| Layer          | Number of Neurons | Activation Function |
| :------------- | :---------------- | :------------------ |
| Input Layer    | 1024              | -                   |
| Hidden Layer 1 | 128               | Leaky ReLU          |
| Hidden Layer 2 | 64                | Leaky ReLU          |
| Output Layer   | 33                | Softmax             |

The training process minimizes the categorical cross-entropy loss function using backpropagation and the Adam optimization algorithm.

---

## Results

The final model, trained with the best-found hyperparameters, demonstrates strong performance and generalization.

- **Overall Accuracy:** **95%** on the held-out test set.

### Learning Curves
The training and validation curves show smooth convergence with a minimal gap, indicating that the model learns effectively without overfitting.

![Loss and Accuracy Curves](assets/loss_and_accuracy_curves.png)

### Confusion Matrix
The confusion matrix for the test set shows a strong diagonal, confirming high accuracy across most classes. Minor confusions occur between visually similar characters.

![Confusion Matrix](assets-confusion_matrix.png)
