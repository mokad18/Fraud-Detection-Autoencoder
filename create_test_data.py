import pandas as pd

print("Lade große Datenbank...")
df = pd.read_csv('creditcard.csv')

# Wir schnappen uns 50 echte Betrugsfälle und 1000 normale Transaktionen
frauds = df[df['Class'] == 1].sample(50, random_state=99)
normals = df[df['Class'] == 0].sample(1000, random_state=99)

# Wir kleben beides zusammen und mischen es gut durch
test_data = pd.concat([frauds, normals]).sample(frac=1)

# Wir speichern das als neue, winzige Datei ab
test_data.to_csv('api_test_daten.csv', index=False)
print("Fertig! Die Datei 'api_test_daten.csv' wurde erstellt.")