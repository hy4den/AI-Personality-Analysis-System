# 🧬 AI Personality Analysis System
### 🎓 CENG313 - Introduction to Data Science (Term Project)

**Developers:**
*   Caghan Tutku Uzundurukan
*   Ali Isakoca

---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 1. Project Overview

This project involves the development of a machine learning system designed to predict an individual's **MBTI (Myers-Briggs Type Indicator)** personality type based on demographic data and specific personality scores (Introversion, Sensing, Thinking, Judging).

The system utilizes an **Elite Voting Ensemble** architecture, combining **Random Forest** and **Gradient Boosting** algorithms to achieve high predictive performance. The final model is deployed via a **Streamlit** web interface, providing real-time analysis and interactive visualizations.

### 🏆 Key Results
| Model | Accuracy |
|-------|----------|
| Baseline (Logistic Regression) | 76.65% |
| **Final (Voting Ensemble RF + GBM)** | **90.51%** |

---

## 📂 2. Project Structure

```bash
CENG313_Project/
├── 📂 data/
│   └── data.csv                  # Raw dataset for training/testing
├── 📂 models/
│   ├── personality_model.joblib  # Trained Voting Ensemble model
│   └── scaler.joblib             # Fitted StandardScaler
├── 📄 app.py                     # Main Streamlit web application
├── 📓 Personality_Analysis_Project.ipynb  # EDA, Preprocessing & Training
├── 📄 requirements.txt           # Project dependencies
└── 📝 README.md                  # Project documentation
```

---

## 🛠 3. Prerequisites & Requirements

To run this project, you need **Python 3.8+**.

**Key Libraries:**
*   `Streamlit` (Web Interface)
*   `Scikit-Learn` (Machine Learning)
*   `Pandas` & `NumPy` (Data Processing)
*   `Plotly` (Interactive Visualizations)

---

## 🚀 4. Installation & Execution

### **Step 1: Navigate to Project**
```bash
cd /path/to/CENG313_Project
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Application**
```bash
streamlit run app.py
```
_The app should open automatically at `http://localhost:8501`_

---

## 🧠 5. Model Training (Optional)
The pre-trained models are included in `models/`. To retrain:
1.  Open `Personality_Analysis_Project.ipynb`.
2.  Run all cells.
3.  New models will be saved to the `models/` directory.