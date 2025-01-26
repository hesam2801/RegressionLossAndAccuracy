# Regression Loss and Accuracy

This repository contains an implementation of a simple Linear Regression model in Python, including calculations for accuracy metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) score.

## Overview

The Linear Regression model is built from scratch using NumPy. It fits a linear model to a dataset and provides predictions based on the learned weights and intercept. The implementation also includes methods to evaluate the model's performance using various loss functions.

## Features

- **Linear Regression Implementation**: A basic model that calculates weights and intercept based on input data.
- **Performance Metrics**: Calculates:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²) score

## Getting Started

### Prerequisites

- Python 3.x
- NumPy library

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hesam2801/RegressionLossAndAccuracy.git
2. Navigate to the project directory:
    ```bash
    cd RegressionLossAndAccuracy
3. Install the required packages (if not already installed):
    ```bash
    pip install numpy
## Usage
You can run the script to see the model in action. The code provided will:
1. Fit the Linear Regression model to the predefined dataset.
2. Calculate and print the MAE, MSE, and R² score.
To execute the script, run:
    ```bash
    python main.py
## Example Output
After running the script, you will see output similar to the following:

    MAE: <value> 
    MSE: <value> 
    R² Score: <value>

## Code Explanation
**Data Preparation:** The input data consists of x_data and y_data, which are used to train the model.\
**Linear Regression Class:** The LinearRegression class contains methods for fitting the model and making predictions. \
**Performance Metrics Calculation:** After predictions, the script calculates MAE, MSE, and R² score to evaluate the model's performance.
## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.