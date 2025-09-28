# Diabetes Prediction using Machine Learning

## 📌 Overview

This project builds a machine learning model to predict whether a person
has diabetes based on health-related features. It uses **Support Vector
Machine (SVM)** for classification and evaluates the model's accuracy.

## ⚙️ Technologies Used

-   **Python 3**
-   **NumPy**
-   **Pandas**
-   **scikit-learn (sklearn)**

## 📂 Dataset

The dataset contains medical predictor variables such as glucose level,
BMI, insulin, age, etc., along with the target variable indicating
diabetes diagnosis.\
*(Replace this section with dataset source if available, e.g., Kaggle's
Pima Indians Diabetes Dataset).*

## 🧑‍💻 Steps in the Notebook

1.  **Import Libraries** -- Load Python libraries for data manipulation
    and ML.
2.  **Load Dataset** -- Read diabetes dataset into a pandas DataFrame.
3.  **Data Exploration** -- Inspect dataset shape, missing values, and
    distributions.
4.  **Data Preprocessing** -- Standardize numerical features using
    `StandardScaler`.
5.  **Train-Test Split** -- Split data into training and testing sets.
6.  **Model Training** -- Train an SVM classifier on the training data.
7.  **Model Evaluation** -- Test the model and calculate accuracy using
    `accuracy_score`.

## 🚀 How to Run

1.  Clone this repository or download the notebook.

2.  Install dependencies:

    ``` bash
    pip install numpy pandas scikit-learn
    ```

3.  Open the notebook:

    ``` bash
    jupyter notebook Diabetes_prediction.ipynb
    ```

4.  Run all cells to train and evaluate the model.

## 📊 Results

The model outputs accuracy scores on the training and testing datasets.\
*(Add specific accuracy numbers after running the notebook).*

## 📌 Future Improvements

-   Try different classifiers (Logistic Regression, Random Forest,
    XGBoost).
-   Perform hyperparameter tuning for SVM.
-   Use cross-validation for more robust evaluation.
-   Add data visualization (feature importance, correlation heatmap).
