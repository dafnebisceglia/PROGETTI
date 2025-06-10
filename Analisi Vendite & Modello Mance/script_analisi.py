import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# 1. IMPOSTAZIONI GRAFICHE (Seaborn)
sns.set_theme(style="whitegrid")

print("--- 1. CARICAMENTO DATI ---")
# Usiamo un dataset integrato in Seaborn (dati reali su scontrini e mance)
df = sns.load_dataset("tips")
print(f"Dataset caricato con successo! Numero di righe: {len(df)}\n")

print("--- 2. ANTEPRIMA DATI ---")
print(df.head())
print("\n" + "=" * 40 + "\n")

# 3. ANALISI GRAFICA CON SEABORN
print("--- 3. CREAZIONE GRAFICO ---")
plt.figure(figsize=(8, 5))

# Grafico a dispersione con linea di tendenza
sns.regplot(
    data=df,
    x="total_bill",
    y="tip",
    scatter_kws={"alpha": 0.6, "color": "teal"},
    line_kws={"color": "coral", "linewidth": 2},
)

plt.title("Relazione tra Conto Totale e Mancia", fontsize=14, fontweight="bold")
plt.xlabel("Conto Totale ($)", fontsize=12)
plt.ylabel("Mancia ($)", fontsize=12)

# Salva il grafico come immagine per il tuo repository
plt.tight_layout()
plt.savefig("grafico_mance.png", dpi=300)
print(" Grafico salvato come 'grafico_mance.png'\n")

# 4. MODELLO STATISTICO CON STATSMODELS
print("--- 4. MODELLO REGRESSIONE OLS (Statsmodels) ---")
# Creazione e fitting del modello OLS: prevede la mancia in base al conto totale
modello = smf.ols("tip ~ total_bill", data=df).fit()

# Stampa del riepilogo statistico
print(modello.summary())