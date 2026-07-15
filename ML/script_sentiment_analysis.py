import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Impostazione tema
sns.set_theme(style="whitegrid", palette="Set2")

print("--- 1. GENERAZIONE E ANALISI TEXT MINING ---")

# Dataset di recensioni clienti simulate
data_recensioni = {
    "ID_Recensione": [f"REV-{i:03d}" for i in range(1, 11)],
    "Testo_Recensione": [
        "Prodotto fantastico, spedizione velocissima!",
        "Pessimo servizio, arrivato rotto e in ritardo.",
        "Nella norma, fa il suo dovere senza infamia e senza lode.",
        "Ottima qualità del materiale, lo consiglio a tutti.",
        "Assistenza clienti inesistente, soldi buttati.",
        "Servizio ok, ma il prezzo è un po' troppo alto.",
        "Eccezionale! Supera ogni mia aspettativa.",
        "Non funziona affatto, chiederò il rimborso.",
        "Discreto, spedizione nei tempi previsti.",
        "Molto soddisfatto del mio acquisto, 5 stelle!"
    ],
    # Assegniamo uno score di sentiment stimato da -1 (molto negativo) a +1 (molto positivo)
    "Score_Sentiment": [0.9, -0.8, 0.1, 0.85, -0.9, -0.2, 0.95, -0.85, 0.2, 0.9]
}

df_reviews = pd.DataFrame(data_recensioni)

# Calcolo features testuali
df_reviews["Lunghezza_Testo"] = df_reviews["Testo_Recensione"].apply(len)
df_reviews["Numero_Parole"] = df_reviews["Testo_Recensione"].apply(lambda x: len(x.split()))

# Classificazione del Sentiment in categorie
def classifica_sentiment(score):
    if score > 0.3:
        return "Positivo"
    elif score < -0.3:
        return "Negativo"
    else:
        return "Neutro"

df_reviews["Categoria_Sentiment"] = df_reviews["Score_Sentiment"].apply(classifica_sentiment)

# Salva il dataset in CSV
df_reviews.to_csv("recensioni_analizzate.csv", index=False)
print(" File 'recensioni_analizzate.csv' creato con successo!\n")

print("--- TABELLA RIASSUNTIVA ---")
print(df_reviews[["ID_Recensione", "Categoria_Sentiment", "Score_Sentiment", "Numero_Parole"]].to_string(index=False))
print("\n" + "=" * 50 + "\n")

# --- 2. GRAFICO ANALISI SENTIMENT ---
plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df_reviews, 
    x="Categoria_Sentiment", 
    order=["Positivo", "Neutro", "Negativo"],
    hue="Categoria_Sentiment",
    legend=False
)

plt.title("Distribuzione del Sentiment nelle Recensioni Clienti", fontsize=13, fontweight="bold")
plt.xlabel("Categoria Sentiment", fontsize=11)
plt.ylabel("Numero di Recensioni", fontsize=11)

# Aggiunta conteggio sulle barre
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(
            f"{int(p.get_height())}", 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 3), textcoords='offset points'
        )

plt.tight_layout()
plt.savefig("grafico_sentiment.png", dpi=300)
print(" Grafico salvato come 'grafico_sentiment.png'\n")