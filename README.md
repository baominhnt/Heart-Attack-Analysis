# Heart-Attack-Analysis
Heart Attack Prediction ProjectOverviewThis project focuses on predicting the presence of heart disease using clinical patient data. By leveraging multiple data mining techniques—including classification and association rule mining—the study identifies key physiological indicators that correlate with heart disease.

Dataset
Source: UCI Machine Learning Repository (Cleveland subset).
Size: 303 patient records.
Target: The variable num (presence of heart disease), commonly binarized into:
0: No Disease.
1+: Disease Present.

Key Features
Clinical Indicators: Age, Sex, Chest Pain Type (cp), Resting Blood Pressure (trestbps), and Cholesterol (chol).
Diagnostic Results: Fasting Blood Sugar (fbs), Resting ECG (restecg), Max Heart Rate (thalach), Exercise-Induced Angina (exang), and ST Depression (oldpeak).
Imaging Data: Number of Major Vessels (ca) and Thalassemia (thal).

Methodology
1. Data Preprocessing
Checked for and handled missing values.
Converted categorical variables into factors for better model processing.
Removed duplicate records to ensure data integrity.

2. ModelingThe project compared two distinct approaches:
- Classification (Supervised): Decision Tree, Naive Bayes, K-Nearest Neighbor (KNN), and Random Forest.
- Association Rules (Unsupervised): Apriori algorithm used to find common co-occurring symptoms.

Model Performance
The models were evaluated based on accuracy, with the Decision Tree emerging as the most effective for this clinical dataset.
RankModelAccuracy:
- Decision Tree: 89.2% Accuracy (Top Performer)
- Naive Bayes: 85.8% Accuracy
- K-Nearest Neighbor (KNN): 69.1% Accuracy
- Random Forest: 52.9% Accuracy


Key InsightsStrongest Indicator:
- Chest pain type (cp) is the primary screening factor. Specifically, "Atypical angina" showed an ~80% probability of heart disease.
- Structural Indicators: The number of major vessels (ca) is a strong anatomical signal. Patients with 0 vessels colored have a significantly higher chance of disease.
- Physiological Response: Max heart rate (thalach) and ST depression (oldpeak) are critical reflectors of heart performance under stress.

Conclusion
The Decision Tree model is recommended for this dataset because it naturally handles mixed data types and captures medical threshold patterns effectively. It provides a fully interpretable structure that is essential for clinical decision-making.
