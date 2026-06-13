# Credit Card Fraud Detection API

Ein Machine Learning Backend zur Erkennung von Anomalien und Kreditkartenbetrug, basierend auf tiefen neuronalen Netzen. Entwickelt im Rahmen einer Seminararbeit an der Technischen Hochschule Ingolstadt .

## Über das Projekt
Dieses Projekt nutzt einen **Autoencoder** (Unsupervised Learning), um betrügerische Transaktionen in hochdimensionalen Finanzdaten zu identifizieren. Das Modell lernt die Repräsentation normaler Transaktionen und erkennt Betrug anhand eines stark abweichenden Rekonstruktionsfehlers (MSE). 

Die Modell-Architektur wurde in einem separaten Skript trainiert, evaluiert und exportiert. Die Auswertung neuer Daten erfolgt über eine performante **FastAPI-Schnittstelle**.

### Tech Stack
* **Machine Learning:** Python, TensorFlow / Keras, Scikit-Learn, Pandas, NumPy
* **Backend:** FastAPI, Uvicorn
* **Visualisierung:** Matplotlib

## Modell-Performance & Visualisierung
![img_1.png](img_1.png)
Schwellenwert:    2.5426
Erkannter Betrug: 398 von 492
Recall:           80.89%
Genauigkeit:      99.7%
MCC:              0.80

## Lokale Installation & Start

1. Repository klonen und Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt