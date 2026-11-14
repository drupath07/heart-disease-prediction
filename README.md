# Heart Disease Prediction Using Machine Learning

##  Project Overview

This project demonstrates an end-to-end Machine Learning workflow for classifying heart disease status using patient-related dataset features.

It uses a **Random Forest Classifier** with hyperparameter tuning and a **Streamlit web application** to demonstrate model predictions through an interactive interface.

**Note:** This is an educational machine learning project, not a medical diagnostic tool. Its predictions must not be used for medical decisions.

##  Project Objectives

- Perform data cleaning and preprocessing.
- Explore the dataset using Exploratory Data Analysis (EDA).
- Train and compare multiple machine learning algorithms.
- Tune the Random Forest model.
- Evaluate model performance.
- Build an interactive Streamlit application.
- Deploy the application using Streamlit Community Cloud.

##  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning and preprocessing |
| Random Forest | Selected classification algorithm |
| Joblib | Model serialization |
| Streamlit | Web application |
| Google Colab | Model development |
| GitHub | Version control and project hosting |

##  Dataset Description

- Total records: 10,000
- Total columns: 21
- Input features: 20
- Target variable: `Heart Disease Status`
- Target classes: Yes and No

The dataset contains numerical and categorical features, including age, gender, blood pressure, cholesterol, exercise habits, smoking, diabetes, BMI and other attributes.

## ⚙️ Machine Learning Workflow

1. Dataset loading
2. Data inspection and cleaning
3. Missing-value handling
4. Exploratory Data Analysis
5. Feature and target separation
6. Categorical encoding and numerical scaling
7. Train-test splitting
8. Model training and comparison
9. Random Forest hyperparameter tuning
10. Model evaluation
11. Model saving using Joblib
12. Streamlit application development
13. Cloud deployment

## 🤖 Models Explored

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees Classifier

**Selected model:** Tuned Random Forest Classifier.

##  Model Evaluation

The tuned Random Forest model produced the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 77.45% |
| ROC-AUC | 0.4817 |
| Positive-class precision | 16% |
| Positive-class recall | 3% |
| Positive-class F1-score | 0.05 |

**Important limitation:** The model has weak discrimination between the two classes and identifies very few positive-class records. Accuracy is affected by the dataset's 80:20 class imbalance. These results do not establish clinical usefulness.

##  Streamlit Application

The application provides:

- Interactive input form with 20 features
- Numerical inputs and categorical dropdowns
- Random Forest classification output
- Model score visualization
- Educational-use disclaimer

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── app.py
├── Heart_Disease_Prediction.ipynb
├── heart_disease_model.pkl
├── feature_config.json
├── requirements.txt
├── runtime.txt
└── README.md
```

##  Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/drupath07/heart-disease-prediction.git
```

### 2. Navigate to the project directory

```bash
cd heart-disease-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

To rerun the notebook, open `Heart_Disease_Prediction.ipynb` in Google Colab and provide the original dataset file.

##  Repository

[GitHub Repository](https://github.com/drupath07/heart-disease-prediction)

##  Disclaimer

This project is developed strictly for academic and educational purposes. It is an experimental machine learning demonstration and is not intended for medical diagnosis, treatment, or health-risk assessment. Do not use its outputs to make health decisions.

##  Author

Drupath Reddy

GitHub: [@drupath07](https://github.com/drupath07)
