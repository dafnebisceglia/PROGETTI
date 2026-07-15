import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. CREAZIONE E SALVATAGGIO DEL DATASET CSV
data_inter = {
    "Giocatore": [
        "Lautaro Martínez",
        "Nicolò Barella",
        "Alessandro Bastoni",
        "Marcus Thuram",
        "Hakan Çalhanoğlu",
        "Federico Dimarco",
        "Benjamin Pavard",
        "Denzel Dumfries",
        "Davide Frattesi",
        "Yann Sommer",
    ],
    "Eta": [28, 29, 27, 28, 32, 28, 30, 30, 26, 37],
    "Gol_Assist": [32, 12, 6, 22, 15, 14, 3, 9, 10, 0],
    "Minuti_Giocati": [3100, 3400, 3200, 2800, 3000, 2900, 2500, 2200, 1800, 3300],
    "Valore_Mercato_Mln": [110, 80, 70, 65, 45, 50, 40, 28, 35, 5],
}

df = pd.DataFrame(data_inter)

# Salva il file CSV nella cartella
df.to_csv("giocatori_inter.csv", index=False)
print(" File 'giocatori_inter.csv' creato con successo!\n")

# 2. PREPARAZIONE PER IL MACHINE LEARNING
# Usiamo Gol+Assist e Minuti Giocati per prevedere il Valore di Mercato
X = df[["Gol_Assist", "Minuti_Giocati"]]  # Feature (Variabili indipendenti)
y = df["Valore_Mercato_Mln"]  # Target (Variabile da prevedere)

# 3. ADDESTRAMENTO MODELLO (REGRESSIONE LINEARE)
modello = LinearRegression()
modello.fit(X, y)

# Previsioni del modello
df["Valore_Predetto_Mln"] = modello.predict(X).round(1)

# 4. VALUTAZIONE DEL MODELLO
r2 = r2_score(y, df["Valore_Predetto_Mln"])
mae = mean_absolute_error(y, df["Valore_Predetto_Mln"])

print("--- RISULTATI MODELLO ML ---")
print(f"Punteggio R² (Accuratezza): {r2:.2f}")
print(f"Errore Medio Assoluto (MAE): €{mae:.2f} Mln\n")

print(
    df[["Giocatore", "Valore_Mercato_Mln", "Valore_Predetto_Mln"]].to_string(
        index=False
    )
)

# 5. GRAFICO CONFRONTO VALORE REALE VS PREDETTO
plt.figure(figsize=(9, 5))
sns.set_theme(style="whitegrid")

plt.scatter(
    df["Valore_Mercato_Mln"],
    df["Valore_Predetto_Mln"],
    color="blue",
    s=100,
    edgecolor="black",
)
plt.plot(
    [0, 120], [0, 120], color="red", linestyle="--", label="Previsione Perfetta"
)

plt.title(
    "ML: Valore Reale vs Valore Predetto (Giocatori Inter)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Valore di Mercato Reale (€ Mln)", fontsize=11)
plt.ylabel("Valore Predetto dal Modello (€ Mln)", fontsize=11)
plt.legend()

plt.tight_layout()
plt.savefig("predizione_valore_inter.png", dpi=300)
print("\n Grafico salvato come 'predizione_valore_inter.png'")