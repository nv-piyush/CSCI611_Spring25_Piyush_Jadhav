# **CSCI611 - Assignment 2**
## **Spring 2025**
### **Author:** Piyush Jadhav

---

## **Project Overview**
This project includes two primary tasks:
1. **Building a Convolutional Neural Network (CNN) for Image Classification**
2. **Implementing Image Filtering Using OpenCV**

Each task is contained within its own Jupyter Notebook, with detailed explanations, code implementation, and outputs.

---

## **1. Convolutional Neural Network (CNN)**
**Notebook:** `build_cnn_pajadhav.ipynb`

### **Description**
This notebook implements a deep learning model using **Convolutional Neural Networks (CNNs)** for image classification. The architecture includes:
- **Five convolutional layers** with ReLU activation and max pooling.
- **Fully connected layers** with dropout for regularization.
- **Input:** RGB images of size **32x32x3**.
- **Output:** Classification into 10 different categories.

### **Key Features**
- Uses **PyTorch** for building and training the CNN.
- Implements **data augmentation** for better generalization.
- Evaluates model performance and accuracy.

### **How to Run**
```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run the notebook
jupyter notebook build_cnn_pajadhav.ipynb
```

---

## **2. Image Filtering Using OpenCV**
**Notebook:** `image_filtering_pajadhav.ipynb`

### **Description**
This notebook applies **image filtering techniques using OpenCV** to process and analyze images. It demonstrates:
- **Edge detection** using convolution filters.
- **Feature extraction** with various kernels.
- **Blurring and Scaling** for noise reduction and enhancement.

### **Key Features**
- Uses **OpenCV and NumPy** for image processing.
- Demonstrates **image convolution** using standard kernels.
- Implements **edge detection filters** such as Sobel and Prewitt.

### **How to Run**
```bash
# Install dependencies
pip install opencv-python numpy matplotlib

# Run the notebook
jupyter notebook image_filtering_pajadhav.ipynb
```

---

## **Conclusion**
- The **CNN model** achieved an accuracy of **77%** on the test dataset.
- The **image filtering techniques** successfully extracted features like edges and textures.

---

### **Acknowledgments**
- Uses **PyTorch** for deep learning.
- Uses **OpenCV** for image processing.

---
