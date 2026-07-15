import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree

print("--- 1. GENERAZIONE DATASET 'ZOMBIE VS UMANO' ---")
np.random.seed(42)
n_soggetti = 150

# Generazione dati divertenti
velocita_kmh = np.random.uniform(2.0, 25.0, size=n_soggetti).round(1)
ha_arma = np.random.choice([0, 1], size=n_soggetti, p=[0.4, 0.6]) # 0 = No, 1 = Sì
livello_paura = np.random.randint(1, 10, size=n_soggetti)

# Regola logica per definire l'Esito: 1 = Sopravvissuto (Umano), 0 = Eliminato (Zombie)
esito = []
for v, a, p in zip(velocita_kmh, ha_arma, livello_paura):
    if (v > 12 and p < 7) or (a == 1 and p < 8):
        esito.append(1)  # Sopravvive
    else:
        esito.append(0)  # Diventa uno zombie

df_zombie = pd.DataFrame({
    "ID": [f"SUBJ-{i:03d}" for i in range(1, n_soggetti + 1)],
    "Velocita_KMH": velocita_kmh,
    "Ha_Arma": ha_arma,
    "Livello_Paura": livello_paura,
    "Sopravvissuto": esito
})

# Salva il file CSV
df_zombie.to_csv("apocalisse_zombie.csv", index=False)
print(" File 'apocalisse_zombie.csv' creato con successo!\n")

# --- 2. ADDESTRAMENTO MODELLO DI CLASSIFICAZIONE ---
X = df_zombie[["Velocita_KMH", "Ha_Arma", "Livello_Paura"]]
y = df_zombie["Sopravvissuto"]

# Creiamo l'Albero di Decisione
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

print(f"Accuratezza del modello nel classificare i sopravvissuti: {clf.score(X, y)*100:.1f}%\n")

# --- 3. GRAFICO DELL'ALBERO DECISIONALE ---
plt.figure(figsize=(12, 6))
plot_tree(
    clf, 
    feature_names=["Velocita (km/h)", "Ha Arma (0/1)", "Livello Paura (1-10)"], 
    class_names=["Zombie 🧟", "Sopravvissuto 🏃"], 
    filled=True, 
    rounded=True,
    fontsize=10
)

plt.title("Albero Decisionale: Regole di Sopravvivenza nell'Apocalisse", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("albero_sopravvivenza_zombie.png", dpi=300)
print(" Grafico salvato come 'albero_sopravvivenza_zombie.png'\n")