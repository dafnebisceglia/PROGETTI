import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

print("--- 1. ANALISI CAMPAIGN MARKETING A/B ---")
# Simulo le performance di 2 varianti di una pagina web
campagna_data = {
    "Variante": ["Pagina Originale (A)", "Nuovo Layout (B)"],
    "Utenti_Totali": [5000, 5000],
    "Conversioni": [320, 480],  # Il layout B ha convertito di più
}

df_mkt = pd.DataFrame(campagna_data)

# Calcolo del Tasso di Conversione (%)
df_mkt["Tasso_Conversione_%"] = (
    df_mkt["Conversioni"] / df_mkt["Utenti_Totali"]
) * 100

print(df_mkt.to_string(index=False))
print("\n" + "=" * 45 + "\n")

print("--- 2. CREAZIONE GRAFICO A/B TEST ---")
plt.figure(figsize=(7, 5))
palette_custom = ["#4c72b0", "#55a868"]

ax = sns.barplot(
    data=df_mkt,
    x="Variante",
    y="Tasso_Conversione_%",
    palette=palette_custom,
    hue="Variante",
    legend=False,
)

# Etichette sopra le barre
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.2f}%",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        xytext=(0, 3),
        textcoords="offset points",
    )

plt.title("Risultati A/B Test: Tasso di Conversione", fontsize=13, fontweight="bold")
plt.ylabel("Tasso di Conversione (%)", fontsize=11)
plt.xlabel("")
plt.ylim(0, 12)

plt.tight_layout()
plt.savefig("report_ab_test.png", dpi=300)
print(" Grafico salvato come 'report_ab_test.png'\n")