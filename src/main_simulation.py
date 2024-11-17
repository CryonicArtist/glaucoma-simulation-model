# main_simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# Virtual Population Creation
def generate_data():
    # Generate data for healthy and glaucoma patients
    # ng/mL
    np.random.seed(42)
    healthy_levels = np.random.normal(loc=204, scale=20, size=1000) 
    glaucoma_levels = np.random.normal(loc=218, scale=20, size=1000) 
    

# Labeling the data
    data = pd.DataFrame({
        'lactoferrin_level': np.concatenate([healthy_levels, glaucoma_levels]),
        'label': [0] * 1000 + [1] * 1000  # 0 for healthy, 1 for glaucoma
    })
    return data

# Plotting the data
def plot_histogram(data):
    plt.hist(data[data['label'] == 0]['lactoferrin_level'], alpha=0.5, label='Healthy', bins=30)
    plt.hist(data[data['label'] == 1]['lactoferrin_level'], alpha=0.5, label='Glaucoma', bins=30)
    plt.legend()
    plt.xlabel('Lactoferrin Level')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lactoferrin Levels')
    plt.show()


# Generating the ROC curve
def evaluate_model(data):
    fpr, tpr, _ = roc_curve(data['label'], data['lactoferrin_level'])
    roc_auc = roc_auc_score(data['label'], data['lactoferrin_level'])

    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def main():
    data = generate_data()
    plot_histogram(data)
    evaluate_model(data)

if __name__ == "__main__":
    main()
