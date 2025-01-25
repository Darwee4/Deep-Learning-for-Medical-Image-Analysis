# Medical Image Analysis with Deep Learning

This repository contains a deep learning implementation for analyzing medical images, specifically designed for chest X-ray classification tasks such as pneumonia detection. The implementation leverages transfer learning with a ResNet50 backbone and includes comprehensive data preprocessing, model evaluation, and inference capabilities.

## Features

- Transfer learning with ResNet50 architecture
- Comprehensive data preprocessing and augmentation
- Model evaluation using AUC-ROC and confusion matrix
- Inference function for classifying new images
- Support for common medical image formats
- Easy-to-use API for training and evaluation

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- NumPy
- Scikit-learn

## Usage

1. Initialize the classifier:
```python
from medical_image_analysis import MedicalImageClassifier
classifier = MedicalImageClassifier()
```

2. Preprocess your data:
```python
train_gen, val_gen = classifier.preprocess_data('path/to/images')
```

3. Train the model:
```python
classifier.model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
classifier.model.fit(train_gen, validation_data=val_gen, epochs=10)
```

4. Evaluate the model:
```python
auc, cm = classifier.evaluate_model(val_gen)
```

5. Classify new images:
```python
predictions = classifier.classify_image('path/to/image.jpg')
```

## Model Architecture

The model uses a ResNet50 backbone with the following custom classification head:
- Global Average Pooling
- Dense layer (256 units, ReLU activation)
- Dropout (0.5)
- Output layer (softmax activation)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
