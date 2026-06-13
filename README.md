# 💳 Credit Card Fraud Detection API

Ein Machine-Learning-Backend zur Erkennung von Anomalien und Kreditkartenbetrug, basierend auf tiefen neuronalen Netzen. Entwickelt im Rahmen einer Seminararbeit an der Technischen Hochschule Ingolstadt (THI).

## 🧠 Über das Projekt
Dieses Projekt nutzt einen **Autoencoder** (Unsupervised Learning), um betrügerische Transaktionen in hochdimensionalen Finanzdaten zu identifizieren. Das Modell lernt die Repräsentation normaler Transaktionen und erkennt Betrug anhand eines stark abweichenden Rekonstruktionsfehlers (MSE). 

Die Modell-Architektur wurde in einem separaten Skript trainiert, evaluiert und exportiert. Die Auswertung neuer Daten erfolgt über eine performante **FastAPI-Schnittstelle**.

### 🛠 Tech Stack
* **Machine Learning:** Python, TensorFlow / Keras, Scikit-Learn, Pandas, NumPy
* **Backend:** FastAPI, Uvicorn
* **Visualisierung:** Matplotlib

## 📊 Modell-Performance & Visualisierung
*(Füge hier später auf GitHub einfach dein generiertes `bild2.png` per Drag & Drop ein, um die Trennung von normalen Transaktionen und Betrugsfällen zu zeigen!)*

## 🚀 Lokale Installation & Start

1. Repository klonen und Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt