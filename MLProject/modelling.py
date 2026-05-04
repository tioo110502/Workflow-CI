import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import dagshub
dagshub.init(repo_owner='tioo110502', repo_name='Eksperimen_SML_Hafiz-Satria', mlflow=True)

# 1. MLflow Setup
mlflow.set_tracking_uri("https://dagshub.com/tioo110502/Eksperimen_SML_Hafiz-Satria.mlflow")
mlflow.set_experiment("Latihan_Model_Basic")

# 2. Data Loading
df = pd.read_csv("Analisis_sentimen_timnas_STY_cleaned.csv")
# Pastikan kolom X adalah string dan isi yang kosong (NaN) diganti dengan teks kosong ""
X = df[df.columns[0]].astype(str) 
y = df[df.columns[-1]]

# 3. Preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Model Training & Logging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogReg_Basic_Local"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # 5. Membuat dan menyimpan file PNG
    cm = confusion_matrix(y_test, model.predict(X_test_vec))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Sentiment Model Prediction")
    plt.savefig("sentiment_model_plot.png")
    
    # Log artifact 
    mlflow.log_artifact("sentiment_model_plot.png")
    
    # Simpan model 
    mlflow.sklearn.log_model(model, "sentiment_model")