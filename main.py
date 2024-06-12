import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import torch

# Load and clean the dataset
def load_and_clean_data(file_path):
    data = pd.read_excel(file_path)
    data.loc[data['EDUCATION'].isin([0, 5, 6]), 'EDUCATION'] = 4
    data.loc[data['MARRIAGE'] == 0, 'MARRIAGE'] = 3
    data = data.astype({'AGE': 'int', 'default': 'int'})
    data.rename(columns={
        'PAY_0':'PAY_SEPT','PAY_2':'PAY_AUG','PAY_3':'PAY_JUL',
        'PAY_4':'PAY_JUN','PAY_5':'PAY_MAY','PAY_6':'PAY_APR',
        'BILL_AMT1':'BILL_AMT_SEPT','BILL_AMT2':'BILL_AMT_AUG',
        'BILL_AMT3':'BILL_AMT_JUL','BILL_AMT4':'BILL_AMT_JUN',
        'BILL_AMT5':'BILL_AMT_MAY','BILL_AMT6':'BILL_AMT_APR',
        'PAY_AMT1':'PAY_AMT_SEPT','PAY_AMT2':'PAY_AMT_AUG',
        'PAY_AMT3':'PAY_AMT_JUL','PAY_AMT4':'PAY_AMT_JUN',
        'PAY_AMT5':'PAY_AMT_MAY','PAY_AMT6':'PAY_AMT_APR'
    }, inplace=True)
    return data

# Plot categorical features
def plot_categorical_features(data):
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    data_cat = data[categorical_features].copy()
    data_cat['default'] = data['default']
    data_cat = data_cat.astype(object)
    data_cat['SEX'] = data_cat['SEX'].replace({1: 'MALE', 2: 'FEMALE'})
    data_cat['EDUCATION'] = data_cat['EDUCATION'].replace({1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others'})
    data_cat['MARRIAGE'] = data_cat['MARRIAGE'].replace({1: 'married', 2: 'single', 3: 'others'})

    for col in categorical_features:
        plt.figure(figsize=(10, 5))
        fig, axes = plt.subplots(ncols=2, figsize=(13, 8))
        data[col].value_counts().plot(kind="pie", ax=axes[0], subplots=True)
        sns.countplot(x=col, hue='default', data=data_cat)
        plt.show()

# Plot numerical features
def plot_numerical_features(data):
    sns.boxplot(x="default", y="LIMIT_BAL", data=data)
    plt.show()

    fig, axes = plt.subplots(ncols=2, figsize=(40, 20))
    data['AGE'].value_counts().plot(kind="pie", ax=axes[0], subplots=True)
    sns.barplot(x='AGE', y='count', data=data['AGE'].value_counts().reset_index(), ax=axes[1], orient='v')
    plt.show()

    sns.boxplot(x="default", y="AGE", data=data)
    plt.show()

    sns.pairplot(data=data[['BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY', 'BILL_AMT_APR']])
    plt.show()

    pay_col = ['PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR']
    for col in pay_col:
        sns.countplot(x=col, hue='default', data=data)
        plt.show()

    sns.pairplot(data=data[['PAY_AMT_SEPT', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR', 'default']], hue='default')
    plt.show()

# Balance the dataset using SMOTE
def balance_data_with_smote(data):
    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(data.iloc[:, :-1], data['default'])
    balanced_data = pd.DataFrame(x_smote, columns=data.columns[:-1])
    balanced_data['default'] = y_smote
    sns.countplot(x='default', data=balanced_data)
    plt.show()
    return balanced_data

# Preprocess data
def preprocess_data(data):
    data['Payment_Value'] = data[['PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR']].sum(axis=1)
    data['Dues'] = (data[['BILL_AMT_APR', 'BILL_AMT_MAY', 'BILL_AMT_JUN', 'BILL_AMT_JUL', 'BILL_AMT_SEPT']].sum(axis=1) -
                    data[['PAY_AMT_APR', 'PAY_AMT_MAY', 'PAY_AMT_JUN', 'PAY_AMT_JUL', 'PAY_AMT_AUG', 'PAY_AMT_SEPT']].sum(axis=1))
    data.replace({'SEX': {1: 'MALE', 2: 'FEMALE'}, 'EDUCATION': {1: 'graduate school', 2: 'university', 3: 'high school', 4: 'others'}, 'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}}, inplace=True)
    data = pd.get_dummies(data, columns=['EDUCATION', 'MARRIAGE'])
    data.drop(['EDUCATION_others', 'MARRIAGE_others'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR'], drop_first=True)
    data.replace({'SEX': {'FEMALE': 0, 'MALE': 1}}, inplace=True)
    data.drop('ID', axis=1, inplace=True)
    return data

# Train and evaluate a model
def train_and_evaluate_model(X, y, model, param_grid):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    grid_clf = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, verbose=3, cv=3)
    grid_clf.fit(X_train, y_train)
    best_model = grid_clf.best_estimator_

    train_class_preds = best_model.predict(X_train)
    test_class_preds = best_model.predict(X_test)

    print("Best parameters:", grid_clf.best_params_)
    print("Best cross-validation score:", grid_clf.best_score_)

    # Evaluate the model
    evaluate_model(y_train, train_class_preds, y_test, test_class_preds)
    plot_roc_curve(best_model, X_test, y_test)

    return best_model

# Evaluate model performance
def evaluate_model(y_train, train_class_preds, y_test, test_class_preds):
    print("Train accuracy:", accuracy_score(y_train, train_class_preds))
    print("Test accuracy:", accuracy_score(y_test, test_class_preds))
    print("Test precision:", precision_score(y_test, test_class_preds))
    print("Test recall:", recall_score(y_test, test_class_preds))
    print("Test F1 score:", f1_score(y_test, test_class_preds))
    print("Test ROC AUC score:", roc_auc_score(y_test, test_class_preds))

    cm = confusion_matrix(y_train, train_class_preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    (tn, fp, fn, tp)

    revenue = tp * 1 +fp * -1
    print("Revenue:", revenue)

# Plot ROC curve
def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.legend(loc=4)
    plt.show()

# Main function
def main():
    data = load_and_clean_data('default of credit card clients.xls')
    plot_categorical_features(data)
    plot_numerical_features(data)
    balanced_data = balance_data_with_smote(data)
    preprocessed_data = preprocess_data(balanced_data)

    X = preprocessed_data.drop(['default', 'Payment_Value', 'Dues'], axis=1)
    y = preprocessed_data['default']

    logistic_regression_model = LogisticRegression(solver='liblinear')
    logistic_regression_param_grid = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    best_logistic_regression_model = train_and_evaluate_model(X, y, logistic_regression_model, logistic_regression_param_grid)

    # svc_model = SVC(probability=True)
    # svc_param_grid = {'C': [10], 'kernel': ['rbf']}
    # best_svc_model = train_and_evaluate_model(X, y, svc_model, svc_param_grid)

    decision_tree_model = DecisionTreeClassifier()
    decision_tree_param_grid = {'max_depth': [20, 30, 50, 100], 'min_samples_split': [0.1, 0.2, 0.4]}
    best_decision_tree_model = train_and_evaluate_model(X, y, decision_tree_model, decision_tree_param_grid)

    random_forest_model = RandomForestClassifier()
    random_forest_param_grid = {'n_estimators': [100, 150, 200], 'max_depth': [10, 20, 30]}
    best_random_forest_model = train_and_evaluate_model(X, y, random_forest_model, random_forest_param_grid)



if __name__ == "__main__":
    main()


