# Wine Quality Prediction using MLP

This repository contains a simple implementation of a Multi-Layer Perceptron (MLP) for predicting wine quality. The dataset used is `WineQT.csv` and the script is `mlp.py`.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Script Explanation](#script-explanation)
- [Results](#results)
- [License](#license)

## Introduction

This project aims to classify wine quality using a neural network implemented from scratch. The model is trained and evaluated using the `WineQT.csv` dataset.

## Dataset

The dataset used in this project is `WineQT.csv`, which contains the following features:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality
- Id

The target variable is `quality`, which is converted into categorical labels (`bad`, `average`, `good`) for the purpose of classification.

## Requirements

To run the script, you need the following Python packages:
- numpy
- pandas
- scikit-learn

You can install these packages using pip:

```sh
pip install numpy pandas scikit-learn
```

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/wine-quality-prediction.git
   cd wine-quality-prediction
   ```

2. Ensure `WineQT.csv` is in the same directory as `mlp.py`.

3. Run the script:
   ```sh
   python mlp.py
   ```

## Script Explanation

The `mlp.py` script performs the following steps:

1. **Load the Dataset**: Reads the `WineQT.csv` file into a pandas DataFrame.
2. **Preprocess the Data**:
   - Converts the `quality` column into categorical data.
   - Encodes the categorical labels into numerical values.
3. **Split the Data**: Divides the data into training and testing sets.
4. **Define the MLP**:
   - Initializes weights and biases.
   - Implements forward and backward passes with sigmoid activation functions.
5. **Train the MLP**: Runs for a specified number of epochs to update weights and biases.
6. **Evaluate the Model**: Predicts the test set results and calculates the accuracy.

## Results

After running the script, the accuracy of the model on the test set will be printed to the console.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
