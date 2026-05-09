import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import dagshub


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

# Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression(max_iter=1000))
])

# MLflow autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="LogReg_Basic_Local"):

    # Training
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Sentiment Model Prediction")

    plt.savefig("sentiment_model_plot.png")

    # Log artifact
    mlflow.log_artifact("sentiment_model_plot.png")

    # Simpan pipeline lengkap
    mlflow.sklearn.log_model(
        model,
        "sentiment_model"
    )

print("model berhasil disimpan!")
