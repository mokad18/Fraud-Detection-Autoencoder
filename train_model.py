import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.layers import Input, Dense, Dropout

# --- 1. REPRODUZIERBARKEIT ---
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- 2. DATEN VORBEREITEN ---
print("Lade Daten...")
df = pd.read_csv('creditcard.csv')

# NEU: Feature Engineering - Sekunden in Tagesstunde umwandeln
df['Hour'] = (df['Time'] / 3600) % 24
df = df.drop(['Time'], axis=1)

# Amount UND die neue Hour-Spalte skalieren
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Hour'] = StandardScaler().fit_transform(df['Hour'].values.reshape(-1, 1))

normal_data = df[df['Class'] == 0]
fraud_data  = df[df['Class'] == 1]

train_data, val_normal = train_test_split(
    normal_data.drop(['Class'], axis=1),
    test_size=0.2,
    random_state=42
)

print(f"Trainingsdaten (Normal):    {len(train_data)}")
print(f"Validierungsdaten (Normal): {len(val_normal)}")
print(f"Betrugsfälle (Test):        {len(fraud_data)}")

# --- 3. AUTOENCODER BAUEN ---
input_dim = train_data.shape[1]

input_layer  = Input(shape=(input_dim,))

# --- 3. AUTOENCODER BAUEN ---
input_dim = train_data.shape[1]

input_layer  = Input(shape=(input_dim,))
encoder      = Dense(14, activation="relu")(input_layer)
bottleneck   = Dense(7,  activation="relu")(encoder)
decoder      = Dense(14, activation="relu")(bottleneck)
output_layer = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# --- 4. TRAINING ---
print("\nStarte Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    train_data, train_data,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_data=(val_normal, val_normal),
    callbacks=[early_stop],
    verbose=1
)

# --- 5. EVALUIERUNG ---
print("\n--- TESTPHASE ---")

fraud_features     = fraud_data.drop(['Class'], axis=1)
fraud_predictions  = autoencoder.predict(fraud_features)
mse_fraud          = np.mean(np.power(fraud_features - fraud_predictions, 2), axis=1)

normal_predictions = autoencoder.predict(val_normal)
mse_normal         = np.mean(np.power(val_normal - normal_predictions, 2), axis=1)

print(f"Ø Fehler (Normal): {np.mean(mse_normal):.4f}")
print(f"Ø Fehler (Betrug): {np.mean(mse_fraud):.4f}")

# Schwellenwert
threshold      = np.percentile(mse_normal, 99.8273)
detected_fraud = mse_fraud > threshold
fraud_count    = np.sum(detected_fraud)

# Metriken
y_true = [0]*len(mse_normal) + [1]*len(mse_fraud)
y_pred = [1 if e > threshold else 0 for e in list(mse_normal) + list(mse_fraud)]

print(f"\n--- ERGEBNISSE ---")
print(f"Schwellenwert:    {threshold:.4f}")
print(f"Erkannter Betrug: {fraud_count} von {len(fraud_data)}")
print(f"Recall:           {100 * fraud_count / len(fraud_data):.2f}%")
print(f"Genauigkeit:      {accuracy_score(y_true, y_pred)*100:.1f}%")
print(f"MCC:              {matthews_corrcoef(y_true, y_pred):.2f}")


# --- 6. VISUALISIERUNG ---
x_max = np.percentile(np.concatenate([mse_normal, mse_fraud]), 99.5)
bins_range = np.linspace(0, x_max, 100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
fig.suptitle('Anomaliedetektion mittels Autoencodern\nAnwendungsbeispiel: Kreditkartenbetrug', fontsize=14)

# --- Oberer Plot: Normale Transaktionen ---
ax1.hist(mse_normal, bins=bins_range, alpha=0.8, color='darkgreen', label='Normale Transaktionen')
ax1.axvspan(0,         threshold, facecolor='lightgreen', alpha=0.3)
ax1.axvspan(threshold, x_max,     facecolor='lightcoral', alpha=0.3)
ax1.axvline(threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Schwellenwert ({threshold:.4f})')
ax1.set_ylabel('Häufigkeit', fontsize=11)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# --- Unterer Plot: Betrugsfälle ---
ax2.hist(mse_fraud, bins=bins_range, alpha=0.8, color='darkred', label='Betrugsfälle')
ax2.axvspan(0,         threshold, facecolor='lightgreen', alpha=0.3)
ax2.axvspan(threshold, x_max,     facecolor='lightcoral', alpha=0.3)
ax2.axvline(threshold, color='blue', linestyle='dashed', linewidth=2, label=f'Schwellenwert ({threshold:.4f})')
ax2.set_xlabel('Rekonstruktionsfehler (MSE)', fontsize=11)
ax2.set_ylabel('Häufigkeit', fontsize=11)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.xlim(0, x_max)
plt.tight_layout()
plt.savefig('bild2.png', dpi=300, bbox_inches='tight')
print("\nGrafik gespeichert als 'bild2.png'")
plt.show()

joblib.dump(StandardScaler().fit(df['Amount'].values.reshape(-1, 1)), 'amount_scaler.save')
joblib.dump(StandardScaler().fit(df['Hour'].values.reshape(-1, 1)), 'hour_scaler.save')
autoencoder.save('autoencoder_model.keras')