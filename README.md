# Landmark Recognition Using Deep Learning

## Overview
This project implements **landmark recognition** using **deep learning techniques**. It is based on **InceptionV3 with transfer learning** and uses **GAN-based data augmentation** to handle an **imbalanced dataset**.

## Dataset
- **Google-Landmarks Dataset v2 (GLDv2)**: A large-scale dataset containing labeled landmark images.
- Data preprocessing includes **resizing, normalization, and augmentation** to improve model performance.

### Dataset Links
1. **Google Landmarks Dataset v2 (GLDv2)**: [Download](https://www.kaggle.com/c/landmark-recognition-2020/data)
2. **Google Research Dataset**: [GitHub Repository](https://github.com/cvdfoundation/google-landmark)

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **InceptionV3 (Pre-trained CNN model)**
- **Generative Adversarial Networks (GANs) for augmentation**
- **Matplotlib & Seaborn for visualization**

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib & Seaborn

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/landmark-recognition.git
   cd landmark-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
4. Test the model on an image:
   ```bash
   python test_model.py --image sample.jpg
   ```

## How to Load Model
```python
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('landmark_recognition_model.h5')

# Predict on an image
def preprocess_image(image_path):
    import cv2, numpy as np
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image / 255.0, axis=0)
    return image

image = preprocess_image('sample.jpg')
prediction = model.predict(image)
print("Predicted Landmark ID:", np.argmax(prediction))
```

## Key Code Components
- **Transfer Learning**: Uses **InceptionV3** for feature extraction.
- **Data Augmentation**: GANs are used to generate additional samples.
- **Classification Model**: Predicts landmarks from images.
- **Evaluation Metrics**: Accuracy, confusion matrix, and classification report.

## Results
- Achieved **high accuracy** on the Google-Landmarks dataset.
- Improved recognition performance through **GAN-based data augmentation**.

## Summary
✔ **Step 1:** Install dependencies  
✔ **Step 2:** Train the CNN model  
✔ **Step 3:** Save & Load the trained model  
✔ **Step 4:** Run predictions on test images  
✔ **Step 5:** Evaluate performance using accuracy & confusion matrix  

## External Description
The **Landmark Recognition System** is an advanced **deep learning application** that identifies **famous landmarks** from images. It combines **transfer learning, CNNs, and GAN-based data augmentation** for **improved classification accuracy**.

## Contributing
Feel free to contribute by submitting issues or pull requests!

## License
This project is licensed under the **MIT License**.

## Acknowledgments
- **Google-Landmarks dataset** for high-quality labeled images.
- **TensorFlow & Keras** for deep learning frameworks.
- **InceptionV3 developers** for the pre-trained model.
- **GAN researchers** for impro
