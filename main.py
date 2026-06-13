from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import io

app = FastAPI(title="Kreditkarten-Betrugserkennung API")

print("Lade Modell und Scaler...")
model = tf.keras.models.load_model('autoencoder_model.keras')
scaler = joblib.load('amount_scaler.save')

THRESHOLD = 3.2949


@app.post("/predict")
async def detect_anomalies(file: UploadFile = File(...)):
    try:
        # 1. Datei einlesen
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # 2. Unnötige Spalten entfernen (genau wie im Training!)
        # Wir müssen sicherstellen, dass die Spalten 'Time' und 'Class'
        # (falls sie in der Upload-Datei existieren) NICHT an das Modell geschickt werden.
        columns_to_drop = []
        if 'Time' in df.columns:
            columns_to_drop.append('Time')
        if 'Class' in df.columns:
            columns_to_drop.append('Class')

        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)

        df_original = df.copy()

        # 3. Skalierung
        df['Amount'] = scaler.transform(df['Amount'].values.reshape(-1, 1))

        # 4. Vorhersage
        predictions = model.predict(df)
        mse = np.mean(np.power(df - predictions, 2), axis=1)

        # 5. Auswertung
        df_original['MSE'] = mse
        df_original['Is_Fraud'] = mse > THRESHOLD

        # 6. Ergebnisse (Wir wandeln die Numpy-Booleans um, da JSON sonst oft crasht)
        anomalies = df_original[df_original['Is_Fraud']].copy()

        # Sicherstellen, dass das Dictionary sauber formatiert wird
        fraud_list = anomalies.head(10).to_dict(orient="records")
        return {
            "status": "Erfolgreich analysiert",
            "total_transactions_checked": int(len(df_original)),
            "anomalies_detected": int(len(anomalies)),
            "fraud_details": fraud_list
        }

    except Exception as e:
        # Falls es wieder crasht, spuckt die Website dir jetzt genau aus, WARUM!
        raise HTTPException(status_code=500, detail=str(e))