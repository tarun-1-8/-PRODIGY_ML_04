# 🍱 Task 04: Food Item Recognition & Calorie Estimation using CNN

This project is part of my Machine Learning Internship under **Prodigy InfoTech**.  
The objective is to build a deep learning model that can recognize food items from images and estimate their calorie content, helping users monitor dietary intake and make better nutritional choices.

---

## 📁 Dataset Information

- **Source:** [Kaggle: Food-101 Dataset](https://www.kaggle.com/dansbecker/food-101)
- **Total Images:** 101,000 (750 training + 250 test images per class)
- **Categories:** 101 different food items (e.g., Pizza, Burger, Sushi, etc.)
- **Format:** Images are divided into class-named folders inside the `images` directory

---

## 🎯 Objective

The aim is to develop a **Convolutional Neural Network (CNN)** that:
1. Classifies the food item shown in an image.
2. Estimates the calorie content based on the identified food category.

---

## 🧪 Project Steps

1. **Dataset Preparation**
   - Loaded images and organized using ImageDataGenerator
   - Resized and normalized all images
   - Created training and validation sets (80/20 split)

2. **CNN Model Building**
   - Built a custom CNN using Keras Sequential API
   - Used Conv2D, MaxPooling, Dropout, Flatten, and Dense layers
   - Applied Softmax activation for multi-class classification

3. **Model Training**
   - Trained the model for multiple epochs with early stopping
   - Visualized training and validation accuracy/loss

4. **Evaluation & Visualization**
   - Generated confusion matrix to evaluate performance
   - Plotted sample predictions
   - Displayed category-wise prediction accuracy

5. **Calorie Estimation**
   - Created a calorie mapping dictionary for 101 food classes
   - Used predicted class to display average calorie per serving

---

## 📊 Results

- Achieved **Top-1 Accuracy** around `70-80%` depending on model size and training time
- Confusion matrix shows good class separation in high-frequency food items
- Calorie estimates displayed alongside food name after prediction

---

## 📦 Tools & Libraries Used

- Python 3
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV (cv2), Matplotlib
- Scikit-learn

---

## 📘 Learning Outcomes

- Practiced image classification using CNNs
- Learned data preprocessing for image datasets
- Implemented real-world food recognition
- Integrated calorie prediction using class-based mapping
- Understood evaluation using accuracy, confusion matrix, and visualization

---

## 📁 Files Included

| File Name | Description |
|-----------|-------------|
| `Task02_Food_Recognition_Calorie_Estimation.ipynb` | Complete notebook with training and prediction |
| `sample_predictions.png` | Grid showing predicted food item and calories |
| `accuracy_loss_plot.png` | Model training vs validation accuracy/loss |
| `README.md` | This documentation file |

---

## 📬 Connect with Me:

**Tarun Sharma**  
B.Tech CSE | K.R. Mangalam University  
Machine Learning Intern @ Prodigy InfoTech  
GitHub: [tarun-1-8](https://github.com/tarun-1-8)  
LinkedIn: [Tarun Sharma](https://www.linkedin.com/in/tarun-sharma-987a6332b)

---
