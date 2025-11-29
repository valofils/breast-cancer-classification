# 🦋 Breast Cancer Classification with Machine Learning

## 📌 Overview
This project builds and evaluates multiple machine learning models to classify breast cancer tumors as **Benign** or **Malignant** using diagnostic features.  
The models are implemented in Python using **scikit-learn**, and the analysis is documented in a Jupyter Notebook.
---

## 📊 Dataset
The dataset used is the **Breast Cancer Diagnostic Wisconsin Dataset**, available on Kaggle:

🔗 https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

It contains:
- **569 samples**
- **30 numeric features**
- Target variable:  
  - *M* = Malignant  
  - *B* = Benign  

---

## 🤖 Models Evaluated
This notebook explores and compares several machine learning algorithms:

- Logistic Regression  
- Support Vector Machine (RBF kernel)  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Naive Bayes  
- Neural Network (MLPClassifier)

Each model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC  

---

## 📈 Example Results
Best-performing models:

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|-------|----------|-----------|--------|----------|------|
| SVM (RBF) | 0.97 | 0.97 | 0.96 | 0.96 | 0.99 |
| Neural Network | 0.97 | 0.96 | 0.95 | 0.95 | 0.99 |


---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Open the notebook
```bash
jupyter notebook notebooks/breast_cancer.ipynb
```

---

## 🧪 Future Improvements
- Hyperparameter tuning  
- Streamlit web app deployment  
- Explainability using SHAP  
- Stratified cross-validation improvements  

---

## 👨‍💻 Author
**Mariel**  
Statistician Data scientist
