# Mammogram Image Classifier
This project comprises two convolutional neural network classifiers developed using TensorFlow and Keras:

- One classifier detects cancerous vs. non-cancerous mammograms.
- The second classifier distinguishes real mammograms from synthetically generated ones.

These models were trained using a custom data pipeline that included grayscale preprocessing, image resizing, and augmentation from both image (.png) and synthetic data (.npz) sources.

The classifiers utilize multiple convolutional layers with dropout for regularization and are optimized using Adadelta and binary crossentropy for performance. Early stopping is employed to avoid overfitting.

Evaluation results indicate robust accuracy and effective separation of classes, with real-time inference and visual prediction visualization integrated into the workflow.
