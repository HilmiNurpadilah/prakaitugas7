import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

scaler = None
model = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/model_diabetes.h5')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../data/scaler_diabetes.save')

def load_artifacts():
    global scaler, model
    if scaler is None and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if model is None and os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)

def predict_diabetes(form_data):
    load_artifacts()
    input_features = [
        float(form_data.get('Kehamilan', 0)),
        float(form_data.get('Glukosa', 0)),
        float(form_data.get('Tekanan Darah', 0)),
        float(form_data.get('Ketebalan Kulit', 0)),
        float(form_data.get('Insulin', 0)),
        float(form_data.get('BMI', 0)),
        float(form_data.get('Fungsi Keturunan Diabetes', 0)),
        float(form_data.get('Usia', 0)),
    ]
    X = np.array(input_features).reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)
    pred = model.predict(X)[0][0]
    return 'Positif Diabetes' if pred >= 0.5 else 'Negatif Diabetes'
