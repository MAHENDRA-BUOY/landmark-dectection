# Landmark Recognition Using Deep Learning

## Overview
This project focuses on **landmark recognition** using **deep learning techniques**. The **Google-Landmarks dataset** is used to train a model capable of identifying various landmarks from images.

## Dataset
- The project uses the **Google-Landmarks dataset**, which contains labeled images of thousands of landmarks.
- Preprocessed images were resized to **128x128 pixels** to optimize memory and computational efficiency.

### **Google-Landmarks Dataset Links:**
1. **Google Landmarks Dataset v2 (GLDv2)**
   - [Download from Kaggle](https://www.kaggle.com/c/landmark-recognition-2020/data)
   - [Google Research Dataset](https://github.com/cvdfoundation/google-landmark)  

2. **Google Landmarks Dataset v1 (GLDv1)**
   - [Google Research Dataset](https://github.com/cvdfoundation/google-landmark)  
   - [Original Google Drive Link](https://research.google/tools/datasets/google-landmarks-dataset/)  

### **Steps to Download the Dataset:**
1. **From Kaggle (Recommended)**
   - Sign in to **Kaggle** and accept competition rules.
   - Use the Kaggle API to download:
     ```bash
     kaggle competitions download -c landmark-recognition-2020
     unzip landmark-recognition-2020.zip -d dataset
     ```

2. **From Google Research**
   - Use `wget` to download specific subsets of the dataset.
   - Follow instructions provided in the official repository.

## Model Implementation
- **InceptionV3** was used for feature extraction, leveraging **transfer learning** for improved classification.
- **Data augmentation techniques**, including **Generative Adversarial Networks (GANs)**, were applied to balance the dataset.
- Achieved **82.03% Top-5 accuracy** using **GAN-based augmentation**.

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

## Results
- The model achieved **82.03% Top-5 accuracy**.
- **Confusion matrices** and **classification reports** provided insights into model performance.
- Data augmentation using **GANs** significantly improved classification accuracy.

## Contributing
Feel free to contribute by submitting issues or pull requests!

## License
This project is licensed under the **MIT License**.

## Acknowledgments
- **Google-Landmarks dataset** for providing high-quality labeled images.
- **TensorFlow & Keras** for deep learning frameworks.
- **InceptionV3 developers** for the pre-trained model.
- **GAN researchers** for improving data augmentation techniques.

