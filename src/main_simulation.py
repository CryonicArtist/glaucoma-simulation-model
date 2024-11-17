# main_simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report


# Virtual Population Creation
def generate_data():
    # Generate lactoferrin levels for healthy and glaucoma patients
    np.random.seed(42)
    healthy_lactoferrin = np.random.normal(loc=204, scale=20, size=1000)
    glaucoma_lactoferrin = np.random.normal(loc=218, scale=20, size=1000)

    # Generate serum ferritin levels for healthy and glaucoma patients
    # Ferritin: Healthy ~ 85.7 ng/mL, Glaucoma ~ 116.1 ng/mL
    healthy_ferritin = np.random.normal(loc=85.7, scale=20, size=1000)
    glaucoma_ferritin = np.random.normal(loc=116.1, scale=20, size=1000)

    # Combine the data into a DataFrame
    data = pd.DataFrame({
        'lactoferrin_level': np.concatenate([healthy_lactoferrin, glaucoma_lactoferrin]),
        'serum_ferritin_level': np.concatenate([healthy_ferritin, glaucoma_ferritin]),
        'label': [0] * 1000 + [1] * 1000  # 0 for healthy, 1 for glaucoma
    })
    return data


# Plotting the data (lactoferrin and serum ferritin distributions)
def plot_histogram(data):
    plt.hist(data[data['label'] == 0]['lactoferrin_level'], alpha=0.5, label='Healthy Lactoferrin', bins=30)
    plt.hist(data[data['label'] == 1]['lactoferrin_level'], alpha=0.5, label='Glaucoma Lactoferrin', bins=30)
    plt.hist(data[data['label'] == 0]['serum_ferritin_level'], alpha=0.5, label='Healthy Ferritin', bins=30)
    plt.hist(data[data['label'] == 1]['serum_ferritin_level'], alpha=0.5, label='Glaucoma Ferritin', bins=30)
    plt.legend()
    plt.xlabel('Biomarker Level')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lactoferrin and Serum Ferritin Levels')
    plt.show()


# Evaluating the model (Logistic Regression)
def evaluate_model(data):
    # Split the data into features and labels
    X = data[['lactoferrin_level', 'serum_ferritin_level']]  # Using both biomarkers
    y = data['label']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using logistic regression for simplicity
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test


# Generating and plotting ROC curve
def evaluate_roc_curve(model, X_test, y_test):
    # ROC curve for both biomarkers
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def main():
    data = generate_data()

    # Plot histograms for lactoferrin and serum ferritin levels
    plot_histogram(data)

    # Evaluate the model using logistic regression
    model, X_test, y_test = evaluate_model(data)

    # Evaluate and plot the ROC curve
    evaluate_roc_curve(model, X_test, y_test)


if __name__ == "__main__":
    main()