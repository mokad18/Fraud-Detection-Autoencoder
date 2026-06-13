from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import io

app = FastAPI(title="Kreditkarten-Betrugserkennung API")

print("Lade Modell und Scaler...")
model = tf.keras.models.load_model('autoencoder_model.keras')
amount_scaler = joblib.load('amount_scaler.save')
hour_scaler = joblib.load('hour_scaler.save')  # Den neuen Scaler laden!

THRESHOLD = 3000.0


@app.post("/predict")
async def detect_anomalies(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # --- NEU: Das gleiche Feature Engineering wie im Training! ---
        if 'Time' in df.columns:
            df['Hour'] = (df['Time'] / 3600) % 24
            df = df.drop(['Time'], axis=1)

        if 'Class' in df.columns:
            df = df.drop(['Class'], axis=1)

        df_original = df.copy()

        # --- NEU: Beide Scaler anwenden ---
        df['Amount'] = amount_scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['Hour'] = hour_scaler.transform(df['Hour'].values.reshape(-1, 1))

        predictions = model.predict(df)
        mse = np.mean(np.power(df - predictions, 2), axis=1)

        df_original['MSE'] = mse
        df_original['Is_Fraud'] = mse > THRESHOLD

        anomalies = df_original[df_original['Is_Fraud']].copy()

        fraud_list = anomalies.head(10).to_dict(orient="records")

        return {
            "status": "Erfolgreich analysiert",
            "total_transactions_checked": int(len(df_original)),
            "anomalies_detected": int(len(anomalies)),
            "fraud_details": fraud_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))