# Chaos Game Representation Classifier

This repository contains code for a classifier based on Chaos Game Representation (CGR) for DNA sequences. The classifier utilizes a supervised machine learning algorithm to classify SARS-CoV-2 sequences into different variants (alpha, beta, delta, and gamma) using CGRs as feature vectors.

## Requirements

Ensure you have the following libraries installed:

- numpy
- pandas
- matplotlib
- scikit-learn

## Getting Started

1. Clone this repository.
2. Install the required dependencies.
3. Ensure you have the SARS-CoV-2 dataset containing fasta files with sequences of alpha, beta, delta, and gamma variants.
4. Run the `classifier.py` script.

## Usage

The `classifier.py` script performs the following steps:

### Step 1: Reading and Pre-processing Data

- Reads the fasta files containing SARS-CoV-2 sequences.
- Ensures sequences contain only uppercase A, C, G, and T.

### Step 2: Computing Feature Vectors

- Computes Chaos Game Representation (CGR) plots with k=7 for all sequences.
- Generates CGR plots representing frequencies of sub-words of length 7 over the alphabet set {A,C,G,T}.

### Step 3: Training ML Model

- Uses CGRs as feature vectors for supervised machine learning classification.
- Flattens the 2D CGRs into 1D vectors and normalizes them between 0 and 1.
- Utilizes a chosen machine learning algorithm (Random Forest) for classification.
- Performs 10-fold cross-validation to compute classification accuracy score.

### Step 4: Testing Using Trained ML Model

- Reads testing sequences from provided fasta files.
- Generates CGRs for testing sequences using k=7.
- Predicts labels of testing sequences using the trained model.
- Prints the predicted labels of the testing sequences.

## Running the Code

You can run the code using the following command:

```bash
python3 classifier.py
