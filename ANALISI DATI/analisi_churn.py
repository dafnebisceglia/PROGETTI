import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="coolwarm")

print("--- 1. CREAZIONE DATASET CLIENTE ---")
np.random.seed(101)
n_clienti = 200

data = {
    "ID_Cliente": [f"CLI-{i:03d}" for i in range(1, n_clienti + 1)],
    "Mesi_Anzianita": np.random.randint(1, 48, size=n_clienti),
    "Spesa_Mensile_EUR": np.random.uniform(20.0, 120.0, size=n_clienti).round(2),
    "Ticket_Assistenza": np.random.randint(0, 8, size=n_clienti),
}

df = pd.DataFrame(data)

# Regola empirica: più ticket d'assistenza si aprono, più è alto il rischio di churn (abbandono)
df["Abbandonato"] = np.where(
    (df["Ticket_Assistenza"] > 3) & (df["Mesi_Anzianita"] < 12), "Sì", "No"
)

print(
    f"Distribuzione Churn:\n{df['Abbandonato'].value_counts(normalize=True) * 100}\n"
)

print("--- 2. CREAZIONE GRAFICO CHURN ---")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Mesi_Anzianita",
    y="Spesa_Mensile_EUR",
    hue="Abbandonato",
    style="Abbandonato",
    s=100,
    alpha=0.8,
)

plt.title(
    "Analisi Churn: Anzianità vs Spesa Mensile", fontsize=13, fontweight="bold"
)
plt.xlabel("Mesi di Anzianità del Cliente", fontsize=11)
plt.ylabel("Spesa Mensile (€)", fontsize=11)
plt.axvline(12, color="gray", linestyle="--", alpha=0.7, label="Soglia 1 Anno")
plt.legend(title="Cliente Abbandonato")

plt.tight_layout()
plt.savefig("grafico_churn.png", dpi=300)
print(" Grafico salvato come 'grafico_churn.png'\n")