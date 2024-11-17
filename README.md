# glaucoma-simulation-model

Current biomarkers being used as of 1.2.3: Lactoferrin Plasma Levels (ng/ml) |  Serum Ferritin Levels (ng/ml)

To download all dependencies in requirements.txt open terminal and type this cmd:

pip install -r requirements.txt

Current Data Results (1.2.3):

![image](https://github.com/user-attachments/assets/84b41a40-19d7-46b5-96ef-97cc98c7d727)

![image](https://github.com/user-attachments/assets/7324b0d0-fa70-43b0-8ae3-dc4161a34029)

![image](https://github.com/user-attachments/assets/765c8da8-a835-41c2-862a-93a331b5e9c8)


Columns Explained:

    Precision:
        The proportion of positive predictions that are actually correct.
        Precision for class 0 (healthy): Out of all the samples predicted to be healthy, how many were actually healthy.
        Precision for class 1 (glaucoma): Out of all the samples predicted to have glaucoma, how many were actually glaucoma patients.
        Formula:
        Precision=True PositivesTrue Positives + False Positives
        Precision=True Positives + False PositivesTrue Positives​

    Recall (Sensitivity or True Positive Rate):
        The proportion of actual positives that are correctly identified by the model.
        Recall for class 0 (healthy): Of all the healthy samples, how many were correctly identified as healthy.
        Recall for class 1 (glaucoma): Of all the glaucoma samples, how many were correctly identified as glaucoma.
        Formula:
        Recall=True PositivesTrue Positives + False Negatives
        Recall=True Positives + False NegativesTrue Positives​

    F1-Score:
        The harmonic mean of precision and recall. It balances precision and recall, especially useful when classes are imbalanced.
        F1 for class 0 (healthy): A combined measure of precision and recall for healthy individuals.
        F1 for class 1 (glaucoma): A combined measure of precision and recall for glaucoma patients.
        Formula:
        F1-Score=2×Precision×RecallPrecision + Recall
        F1-Score=2×Precision + RecallPrecision×Recall​

    Support:
        The number of true instances for each class in the test dataset.
        Support for class 0 (healthy): The number of healthy samples in the test set.
        Support for class 1 (glaucoma): The number of glaucoma samples in the test set.

    Accuracy:
        The overall proportion of correctly predicted instances (both healthy and glaucoma).
        Formula:
        Accuracy=True Positives + True NegativesTotal Samples
        Accuracy=Total SamplesTrue Positives + True Negatives​

    Macro Average:
        The average performance metrics (precision, recall, and F1-score) calculated across all classes without considering the class imbalance (equal weight to each class).
        Macro avg Precision: Average precision across all classes.
        Macro avg Recall: Average recall across all classes.
        Macro avg F1-Score: Average F1-Score across all classes.

    Weighted Average:
        The average performance metrics weighted by the number of samples in each class. This accounts for class imbalance, giving more importance to larger classes.
        Weighted avg Precision: Weighted average precision across all classes.
        Weighted avg Recall: Weighted average recall across all classes.
        Weighted avg F1-Score: Weighted average F1-Score across all classes.
