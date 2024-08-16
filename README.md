# Credit Card Default Prediction

This repository contains a comprehensive machine learning project for predicting credit card defaults using various classification algorithms. The main objectives of the project include data preprocessing, feature engineering, model training, evaluation, and model selection.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Feature Engineering](#feature-engineering)
6. [Models](#models)
7. [Evaluation](#evaluation)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

The goal of this project is to predict whether a customer will default on their credit card payment in the next month. The dataset used for this project is from a credit card dataset provided in `.xls` format. The project includes steps such as data preprocessing, balancing the dataset using SMOTE, and training and evaluating various machine learning models.

## Dataset

The dataset used is the "Default of Credit Card Clients" dataset, which contains the following features:

- **Categorical Features:** Gender, Education, Marital Status.
- **Numerical Features:** Limit Balance, Age, Payment History, Bill Amounts, Payment Amounts.
- **Target Variable:** Default status (whether the client defaulted or not).

Some of the steps taken for cleaning the dataset include:
- Recoding incorrect or irrelevant values in categorical features.
- Renaming columns for clarity.
- Dropping unnecessary columns after feature engineering.

## Installation

### Requirements

To run the project, you need the following libraries:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `imbalanced-learn`
- `torch` (optional, for future deep learning model extensions)

You can install all the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Instructions

1. Clone the repository:

```bash
git clone https://github.com/your-username/credit-card-default-prediction.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Download the dataset (`default of credit card clients.xls`) and place it in the project directory.

4. Run the project using:

```bash
python main.py
```

## Project Structure

```
├── data/                          # Folder to store the dataset
├── notebooks/                     # Jupyter notebooks for experiments and EDA
├── src/                           # Source code files
│   ├── preprocessing.py           # Data loading and preprocessing functions
│   ├── feature_engineering.py     # Feature engineering functions
│   ├── model_training.py          # Model training and evaluation functions
│   ├── visualizations.py          # Visualization and plotting functions
├── main.py                        # Main script to run the project
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## Feature Engineering

The project includes several feature engineering steps, such as:

- **Payment Value:** Sum of payment delays across months.
- **Dues:** Difference between total billed amounts and total payment amounts.
- **One-Hot Encoding:** Applied to categorical features like Education and Marriage.

## Models

The following models have been implemented and evaluated:

1. **Logistic Regression:** A simple, interpretable baseline model.
2. **Decision Tree Classifier:** A non-linear model that works well for classification tasks.
3. **Random Forest Classifier:** An ensemble method that improves model performance through bagging.
4. (Optional) **Support Vector Classifier (SVC):** Uncomment the SVC code to include this model in the training pipeline.

For each model, hyperparameter tuning is performed using `GridSearchCV`.

## Evaluation

The performance of each model is evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC Score**

Confusion matrices and ROC curves are also plotted to visualize model performance. Additionally, revenue calculations based on true positives and false positives are included to assess the financial impact of the models.

## Usage

1. **Data Preprocessing:** The dataset is cleaned and preprocessed, including handling missing values, encoding categorical variables, and scaling numerical features.
2. **Model Training:** Various models are trained and evaluated using cross-validation to identify the best model.
3. **Model Evaluation:** The best model is evaluated on a test set, and its performance is reported using various metrics.
4. **Data Balancing:** SMOTE is used to balance the dataset and handle class imbalance issues.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
