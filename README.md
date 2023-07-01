# Handwritten-Number-Recognizer

This is a project that uses a convolutional neural network (CNN) to recognize handwritten numbers from the MNIST dataset. The project is written in Python and utilizes the TensorFlow and Keras libraries.

## Requirements

To run this project, you need to have the following installed:

- Python 3.6 or higher: The programming language used for the project.
- TensorFlow 2.0 or higher: An open-source machine learning framework used for building and training the CNN model.
- Keras 2.3 or higher: A high-level neural networks API that runs on top of TensorFlow. It provides a simplified interface for building and training neural networks.
- Matplotlib 3.1 or higher: A plotting library used for visualizing the model predictions and performance.
- Numpy 1.17 or higher: A fundamental package for scientific computing with Python. It is used for numerical operations and data manipulation.

## Usage

To run this project, you need to open the `Handwritten Number Recognizer.ipynb` file in a Jupyter notebook or Google Colab. The notebook contains the code and explanations for each step of the project. Here are the main steps covered in the notebook:

1. Loading and preprocessing the data: The MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits, is loaded and preprocessed. The data is divided into training and testing sets, and preprocessing steps such as normalization and reshaping are performed.

2. Building and training the CNN model: A convolutional neural network (CNN) model is constructed using the Keras API. The model architecture typically consists of convolutional layers, pooling layers, and fully connected layers. The model is trained on the training set using the Adam optimizer and categorical cross-entropy loss function.

3. Evaluating the model performance: The trained model is evaluated on the testing set to measure its accuracy and performance. The evaluation metrics, such as accuracy, precision, recall, and F1 score, are computed to assess the model's ability to correctly classify handwritten digits.

4. Visualizing the model predictions: The model's predictions on a subset of the testing data are visualized using the Matplotlib library. The original images, along with their predicted labels, are displayed to provide a visual representation of the model's performance.

You can also modify the code and experiment with different parameters and architectures for the CNN model to improve its performance or adapt it to other similar tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
