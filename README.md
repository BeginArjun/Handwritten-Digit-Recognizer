# Handwritten Digit Recognizer

This Git repository contains a simple handwritten digit recognition application built using Python and TensorFlow. It allows you to draw a digit on the canvas, and it will predict the digit using a neural network model trained on a dataset of handwritten digits.

## Requirements

Before running the application, make sure you have the following dependencies installed:

- `image` (version 1.5.20)
- `numpy` (version 1.24.1)
- `tensorflow`
- `tensorflow-datasets`
- `matplotlib`

You can install these dependencies using pip with the following command:

```bash
pip install -r requirements.txt
```

## How to Run

To run the Handwritten Digit Recognizer application, execute the following command:

```bash
python app.py
```

## Usage

1. Launch the application using `python app.py`.
2. A canvas will appear where you can draw a digit using your mouse.
3. Click the "Predict" button to make a prediction of the drawn digit.
4. The predicted digit will be displayed in the console.

You can also click the "Erase All" button to clear the canvas and start over.

## Code Structure

- `app.py`: This file contains the main application code, including the GUI using Tkinter, drawing on the canvas, and making predictions using the neural network model.

- `model.py`: This file defines functions for loading data, preprocessing images, training the neural network model, making predictions, and displaying errors.

- `data` : This folder contains `X.npy` and `y.npy` which are the datasets used by the Model.

## Neural Network Model

The neural network model used for digit recognition consists of three layers:

1. Input Layer: 400 units
2. Hidden Layer 1: 25 units with ReLU activation
3. Hidden Layer 2: 15 units with ReLU activation
4. Output Layer: 10 units with linear activation

The model is trained on a dataset of handwritten digits and compiled with the Adam optimizer and sparse categorical cross-entropy loss.

## Displaying Errors

The application can also display errors by comparing the predicted digits to the actual labels for a subset of the dataset. The errors are displayed in a grid, showing the misclassified digits and their predicted labels.

## Acknowledgments

This application was created as a learning project and uses the MNIST dataset for training the neural network model.