# 🌎 Garbage Classification CNN 🗑️

![Garbage Classification](https://img.shields.io/badge/Computer%20Vision-Waste%20Classification-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-50%2B%25-blue)

> AI-powered image classification system to identify and sort various types of waste materials

## 📝 Overview

This project implements a Convolutional Neural Network (CNN) to classify waste into six categories: cardboard, glass, metal, paper, plastic, and trash. With growing environmental concerns about waste management, automated classification systems can significantly improve recycling efficiency and reduce the environmental impact of waste.

## ✨ Features

- **High Accuracy Classification**: Achieves 90%+ accuracy in waste material identification
- **6 Waste Categories**: Distinguishes between cardboard, glass, metal, paper, plastic, and general trash
- **Data Augmentation**: Implements rotation, zoom, shift, shear, and flip augmentations to improve model robustness
- **Optimized CNN Architecture**: Uses a deep neural network with batch normalization for efficient learning
- **Performance Analysis**: Includes confusion matrix and class-wise accuracy reporting

## 🛠️ Technical Architecture

The model uses a sequential CNN architecture with:
- Multiple convolutional layers (32 → 64 → 128 → 256 filters)
- Batch normalization for training stability
- MaxPooling layers to reduce dimensionality
- Dropout layers (0.2-0.4) to prevent overfitting
- Dense output layer with softmax activation for 6-class classification

## 📊 Model Performance

The model is trained to automatically stop when reaching 90% accuracy, but often exceeds this threshold.

```
Class-wise Accuracy:
cardboard: 95.23%
glass: 92.15%
metal: 94.87%
paper: 91.04%
plastic: 88.72%
trash: 89.91%
```

## 🔧 Installation & Setup

1. Clone this repository:
   ```
   git clone https://github.com/SanyaShresta25/Garbage-Classification-Using-CNN-AlexNet.git
   cd Garbage_Classification_Using_CNN&AlexNet
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Organize images in folders named after each category
   - Default folder structure:
     ```
     Garbage classification/
     ├── cardboard/
     ├── glass/
     ├── metal/
     ├── paper/
     ├── plastic/
     └── trash/
     ```

4. Configure settings in the script if needed:
   - Adjust `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, etc.

## 🚀 Usage

The script will:
1. Load and preprocess the dataset
2. Create data generators with augmentation
3. Define and compile the CNN model
4. Train until reaching 90% accuracy or the maximum epochs
5. Save the trained model to `garbage_classifier_model.h5`
6. Generate performance metrics and visualizations

### Inference

For classifying new images:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('garbage_classifier_model.h5')

# Load and preprocess an image
img = image.load_img('path_to_image.jpg', target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the class
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

# Map class index to label
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
result = labels[predicted_class]
print(f"Predicted class: {result}")
```

## 📋 Dataset

This project uses the [Garbage Classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) from Kaggle, containing labeled images of different waste materials.

Dataset composition:
- 2527 images across 6 categories
- Resolution varies, resized to 384×512 during training
- Split into 80% training and 20% validation

  ![image](https://github.com/user-attachments/assets/4c9b1ee6-8a9a-4917-a81b-08f30087a18a)


## 🔍 Model Evaluation

The model evaluation includes:
- Confusion matrix visualization
- Detailed classification report with precision, recall, and F1-score
- Class-wise accuracy metrics
- Hyperparameter tuning
  ![image](https://github.com/user-attachments/assets/6ef57ab6-52b3-4cde-bdc5-6517ddf842be)


## 🛣️ Future Improvements

- Implement real-time classification using webcam feed
- Create a web or mobile application interface
- Explore transfer learning with pre-trained models (ResNet, EfficientNet)
- Add object detection to identify multiple waste items in a single image
- Deploy the model to edge devices for on-site waste sorting


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or feedback, please open an issue in the repository or contact [sanyashresta@gmail.com](mailto:your-email@example.com).

---

Made with ❤️ for a cleaner planet
