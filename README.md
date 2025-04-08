# Project Summary

This project aimed to develop a system for predictive maintenance of automotive engines using machine learning. It addressed two key problems:
- Engine condition classification
- Anomaly detection in lubricant oil pressure

---

## Exploratory Data Analysis (EDA)

- **Data Quality:**  
  - The dataset was loaded and inspected for missing values and duplicates.  
  - No missing values or duplicates were found, indicating good data quality for analysis.

- **Data Distribution:**  
  - Histograms were used to visualize the distribution of each feature.  
  - This provided insights into the range and frequency of values for each parameter.

- **Feature Relationships:**  
  - A correlation heatmap was generated to identify relationships between features.  
  - Notably, lubricant oil pressure showed minimal correlation with other engine parameters.  
  - This suggested that predicting it accurately might be challenging.

- **Outlier Detection:**  
  - Box plots were used to identify potential outliers in the data.  
  - **While some outliers were observed, they were not removed as they could represent real-world scenarios.**

---

## Engine Condition Classification

- **Model Selection:**  
  Several classification models were trained and evaluated:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

- **Hyperparameter Tuning:**  
  - Used `GridSearchCV` to optimize model hyperparameters for better performance.

- **Performance Evaluation:**  
  - Models were evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC score
- **Best Model:**
  - The best model selected as **Random Forest** based on the highest ROC-AUC score for engine condition classification.
  - This model showed good discrimination between **Normal** and **Abnormal** engine conditions.


---

## Anomaly Detection in Lubricant Oil Pressure (Regression Approach)

- **Model Selection:**  
  Different regression models were used to predict Lubricant Oil Pressure:
  - Multiple Linear Regression
  - Random Forest Regression
  - XGBoost Regression

- **Key Observations from Analysis:**
  - The target variable *Lub oil pressure* was found to be almost constant across all records.
  - Standard Deviation was just 1.02
  - Correlation of Lub oil pressure with other features was negligible or very weak.

- **Reason for Similar Performance Across Models:**  
  - Due to the constant nature of Lub oil pressure and negligible correlation with features, all models mostly predicted the mean value (≈3.30).
  - Thus, advanced models or hyperparameter tuning did not add significant improvement.

---

## Anomaly Detection Logic

- For a new incoming data point:
  - Predict Lub oil pressure using the best-performing regression model.
  - Compare predicted value with actual measured value.
  - If deviation exceeds a defined threshold, flag it as an anomaly.

---

## Final Conclusion

- Engine classification models performed reasonably well after tuning and evaluation.
- Lub oil pressure anomaly detection using regression faced inherent challenges due to:
  - Very low variance in target variable.
  - Negligible correlation with other features.
- This use case is more suited for a rule-based anomaly detection approach rather than complex machine learning models due to the static behavior of Lub oil pressure.
