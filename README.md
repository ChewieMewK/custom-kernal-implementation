# Custom Kernel Implementation

## Overview

This repository includes a custom kernel implementation for a perceptron model in Python. The model supports two types of kernels: dot product and radial basis function (RBF). The implementation focuses on training the perceptron and making predictions based on the input data.

## File Descriptions

- **kernel.py**: Main file containing the perceptron implementation with support for different kernel functions.
- **util.py**: Utility functions for data loading, plotting, and handling datasets.
- **main.ipynb**: Jupyter Notebook demonstrating how to use the kernel implementation.

## Key Functions in `util.py`

- `add_intercept_fn(x)`: Adds an intercept to the input matrix.
- `load_csv(csv_path, label_col='y', add_intercept=False)`: Loads a dataset from a CSV file, optionally adding an intercept.
- `load_spam_dataset(tsv_path)`: Loads a spam dataset from a TSV file.
- `plot(x, y, theta, save_path, correction=1.0)`: Plots dataset and fitted logistic regression parameters.
- `plot_contour(predict_fn)`: Plots a contour based on a prediction function.
- `plot_points(x, y)`: Visualizes points based on their coordinates and labels.
- `write_json(filename, value)`: Saves provided data as JSON to a specified file.

## Key Functions in `kernel.py`

- `initial_state()`: Initializes the perceptron's state.
- `predict(state, kernel, x_i)`: Predicts the label for a given input using the current state and specified kernel.
- `update_state(state, kernel, learning_rate, x_i, y_i)`: Updates the perceptron state based on the input instance and its label.
- `dot_kernel(a, b)`: Computes the dot product kernel.
- `rbf_kernel(a, b, sigma=1)`: Computes the radial basis function kernel.
- `train_perceptron(kernel_name, kernel, learning_rate)`: Trains the perceptron using the specified kernel and saves predictions and plots.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/ChewieMewK/custom-kernal-implementation.git
   ```
2. Run the Jupyter Notebook to explore the implementation and visualize results.
