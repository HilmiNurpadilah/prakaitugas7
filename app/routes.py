from flask import render_template, request
from app import app
import pandas as pd
import os

# Fungsi prediksi sederhana
from tensorflow.keras.models import load_model
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/model_diabetes.h5')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../data/scaler_diabetes.save')

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_diabetes_simple(insulin, glukosa, keturunan):
    # Default value untuk fitur lain
    X = np.array([[0, float(glukosa), 0, 0, float(insulin), 0, float(keturunan), 0]])
    X = scaler.transform(X)
    pred = model.predict(X)[0][0]
    hasil = 'Positif' if pred >= 0.5 else 'Negatif'
    # Golongan berdasarkan glukosa
    glukosa_val = float(glukosa)
    if glukosa_val < 140:
        gol = 'Normal'
    elif glukosa_val < 200:
        gol = 'Pra-diabetes'
    else:
        gol = 'Diabetes Tinggi'
    return hasil, gol, float(pred)

@app.route('/', methods=['GET', 'POST'])
def index():
    tabel = None
    prediksi = None
    golongan = None
    probabilitas = None
    if os.path.exists('data/diabetes_selected.csv'):
        df = pd.read_csv('data/diabetes_selected.csv', delimiter=';')
        tabel = df.head(30).to_html(classes='table table-striped table-bordered', index=False, border=0, justify='center')
    if request.method == 'POST':
        insulin = request.form.get('Insulin', 0)
        glukosa = request.form.get('Glukosa', 0)
        keturunan = request.form.get('Fungsi_Keturunan_Diabetes', 0)
        prediksi, golongan, probabilitas = predict_diabetes_simple(insulin, glukosa, keturunan)
    return render_template('index.html', tabel=tabel, prediksi=prediksi, golongan=golongan, probabilitas=probabilitas)
