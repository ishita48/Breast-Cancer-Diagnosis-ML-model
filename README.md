# 🎗️Breast Cancer Diagnosis Machine Learning Project🎗️

## Overview
Welcome to the Breast Cancer Diagnosis Machine Learning Project! This project aims to develop and evaluate machine learning models for predicting breast cancer diagnosis using the Wisconsin Breast Cancer dataset. The goal is to create accurate models that classify breast masses as benign or malignant based on their features.

For a detailed exploration of the project, view the Colab notebook [here](https://colab.research.google.com/github/ishita48/Breast-Cancer-Diagnosis-ML-model/blob/main/Breast_Cancer_Diagnosis.ipynb).

## Dataset 📊
The [Wisconsin Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) includes essential features such as:

- **Radius, texture, perimeter, area**: Measurements derived from digitized images.
- **Smoothness, compactness, concavity**: Texture and shape characteristics.
- **Symmetry, fractal dimension**: Geometric attributes.

These features are crucial for training models to predict whether a breast mass is benign or malignant.

## Technologies 🛠️
This project harnesses the power of several technologies:

- Python programming language
- Jupyter Notebook for interactive development
- Pandas and NumPy for data manipulation
- Scikit-Learn for machine learning modeling
- TensorFlow and Keras for building neural networks
- Google Colab for collaborative development
- GitHub for version control and collaboration

## Preprocessing 📋
The dataset undergoes preprocessing steps to ensure optimal model performance:

- **Data Loading**: Importing and loading data using Pandas.
- **Handling Categorical Variables**: Encoding categorical features.
- **Normalization**: Scaling numerical features to a standard range.
- **Train-Test Split**: Dividing data into training and validation sets.

## Models 🤖
A diverse set of models are evaluated:

- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting**
- **Support Vector Machine (SVM)**
- **Simple Neural Network**
- **Medium Neural Network**
- **Large Neural Network**

Each model is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Results 📈
### Model Performance Comparison
- **Support Vector Machine (SVM)** and **Logistic Regression** lead in performance:
  - **SVM**: 98.2% accuracy, perfect recall, and 97.2% precision.
  - **Logistic Regression**: 97.4% accuracy, 97.2% precision, and 98.6% recall.
  
- **Neural Networks** (Simple, Medium, Large) show competitive performance:
  - Accuracies range from 95.6% to 97.3%, with F1-scores from 96.4% to 97.2%.

### Conclusion 🎉
- **SVM**: Ideal for high recall and accuracy, crucial for identifying positive cases.
- **Logistic Regression**: Simple yet effective, offering interpretability.
- **Neural Networks**: Provide advanced modeling capabilities with potential for further enhancement through tuning.

## Usage 🚀

- **Data Preparation**: Load and preprocess the Wisconsin Breast Cancer dataset.
- **Model Training**: Train models using various algorithms and hyperparameters.
- **Evaluation**: Assess model performance with validation metrics and visualizations.
- **Contribution**: Contributions are welcome via pull requests. Fork the repository, create a branch, commit changes, and submit a pull request.

## Visualizations 📊
Visualize model performance with:

**Confusion Matrices**: Illustrating true positives, false positives, true negatives, and false negatives.
**Classification Reports**: Detailed metrics for precision, recall, and F1-score.

## Installation 🔧
Ensure Python 3.x and required libraries are installed:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
