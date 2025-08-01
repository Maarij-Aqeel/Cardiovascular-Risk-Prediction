# Cardiovascular-Risk-Prediction


## Task Objective (X)

I developed a **supervised machine learning model** to predict the risk of heart disease based on key medical and demographic parameters. The goal was to assist in early risk identification by training classification models on historical patient data.

---

## How I Measured Success (Y)

- **Random Forest Classifier** achieved an accuracy of **81%**, outperforming logistic regression  
- Evaluated using metrics such as **precision, recall, f1-score**, and **support**  
- Model performance assessed through **classification reports** and **cross-validation**  
- Demonstrated strong predictive capability, particularly for high-risk patients

---

## How I Built It (Z)

### Dataset Used

- **Heart Disease UCI Dataset** (Available on [Kaggle](https://www.kaggle.com/))  
- Features include:
  - Demographic data (e.g., age, sex)
  - Medical indicators (e.g., trestbps, cholesterol, thalach, ca, oldpeak)
  - Categorical features such as chest pain type and fasting blood sugar
- **Missing values** were handled through **imputation techniques** to ensure data integrity  
- Target variable: `1` (disease present), `0` (no disease)

---

### Models Applied

- **Logistic Regression**
  - Interpretable baseline model for binary classification
  - Achieved ~75% accuracy after preprocessing and tuning

- **Random Forest Classifier**
  - Ensemble-based model with high generalization performance
  - Achieved **~81% accuracy** on the test set
  - Provided **feature importance** insights into top contributing factors

---

### Classification Report (Random Forest)
          precision    recall  f1-score   support

       0       0.78      0.77      0.77        73
       1       0.85      0.86      0.85       111
       accuracy                           0.82       184


---

## Key Findings

- **Random Forest** outperformed logistic regression in both accuracy and recall  
- The model was particularly strong in identifying **positive cases (label 1)**  
- Features like **age, chest pain type, thalach, oldpeak, and ca** showed high importance  
- Imputation helped preserve dataset size and boosted model stability  

---

## Use Cases

- Early detection tool for **cardiovascular risk assessment**  
- Can be integrated into **healthcare dashboards** for real-time prediction  
- Provides physicians with a **non-invasive, data-driven diagnostic aid**  

---

## Technical Stack

- **Language**: Python  
- **Libraries**: pandas, NumPy, scikit-learn, matplotlib, seaborn  
- **Models**: Logistic Regression, Random Forest Classifier  
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix  

---

## Future Work

- Experiment with **XGBoost and Gradient Boosting** for higher accuracy  
- Deploy the model using **Flask or FastAPI** with a simple UI  
- Apply **hyperparameter tuning** and **feature engineering** for performance improvement  

---


