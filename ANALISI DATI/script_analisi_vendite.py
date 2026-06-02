import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Impostazioni stile grafici
sns.set_theme(style="ticks", palette="muted")

print("--- 1. GENERAZIONE E PULIZIA DATI ---")
# Creiamo un dataset realistico di vendite aziendali
np.random.seed(42)
categorie = [
    "Elettronica",
    "Abbigliamento",
    "Casa e Cucina",
    "Libri",
    "Sport",
    "Giardinaggio",
]

data = {
    "Categoria": categorie,
    "Fatturato_EUR": np.random.randint(15000, 85000, size=len(categorie)),
    "Costi_EUR": np.random.randint(8000, 50000, size=len(categorie)),
    "Unita_Vendute": np.random.randint(100, 1200, size=len(categorie)),
}

df = pd.DataFrame(data)

# Calcolo KPI aziendali (Profitto e Margine %)
df["Profitto_EUR"] = df["Fatturato_EUR"] - df["Costi_EUR"]
df["Margine_ %"] = (df["Profitto_EUR"] / df["Fatturato_EUR"]) * 100

# Ordiniamo per profitto decrescente
df = df.sort_values(by="Profitto_EUR", ascending=False).reset_index(drop=True)

print("\nTabella Performance Categorie:")
print(
    df[["Categoria", "Fatturato_EUR", "Profitto_EUR", "Margine_ %"]].to_string(
        index=False
    )
)
print("\n" + "=" * 50 + "\n")

print("--- 2. CREAZIONE REPORT GRAFICO ---")
fig, ax1 = plt.subplots(figsize=(9, 5))

# Grafico a barre per il Profitto
bars = sns.barplot(
    data=df,
    x="Categoria",
    y="Profitto_EUR",
    ax=ax1,
    palette="Viridis",
    hue="Categoria",
    legend=False,
)

# Linea di target profitto medio
profitto_medio = df["Profitto_EUR"].mean()
ax1.axhline(
    profitto_medio,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Profitto Medio (€{profitto_medio:,.0f})",
)

# Titoli e etichette
ax1.set_title(
    "Profitto Netto per Categoria di Prodotto", fontsize=14, fontweight="bold"
)
ax1.set_xlabel("Categoria", fontsize=11)
ax1.set_ylabel("Profitto (€)", fontsize=11)
ax1.legend(loc="upper right")

# Etichette sui valori delle barre
for p in bars.patches:
    bars.annotate(
        f"€{p.get_height():,.0f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=9,
        xytext=(0, 3),
        textcoords="offset points",
    )

plt.xticks(rotation=15)
plt.tight_layout()

# Salva l'immagine report
plt.savefig("report_profitto.png", dpi=300)
print(" Grafico salvato come 'report_profitto.png'\n")