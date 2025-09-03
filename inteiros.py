import pandas as pd
import numpy as np

# Caminho do dataset (troca para o teu ficheiro)
dataset_path = "../StudyCase/cicids2017-original/original-cicids2017-train.csv"

# Ler dataset
df = pd.read_csv(dataset_path)

# Guardar resultados
apenas_inteiros = []
tem_decimais = {}
tem_negativos = {}

# Iterar só pelas colunas numéricas
for col in df.select_dtypes(include=[np.number]).columns:
    serie = df[col].dropna()

    # parte decimal
    frac = serie - np.floor(serie)

    if (frac == 0).all():
        apenas_inteiros.append(col)
    else:
        exemplo_decimal = serie[frac != 0].iloc[0]
        tem_decimais[col] = exemplo_decimal

    # valores negativos
    if (serie < 0).any():
        exemplo_neg = serie[serie < 0].iloc[0]
        tem_negativos[col] = exemplo_neg

print("🔢 Colunas onde todos os valores são inteiros (parte decimal = 0):")
print(apenas_inteiros)

print("\n💡 Colunas onde existe pelo menos um valor com parte decimal ≠ 0 (com exemplo):")
for col, exemplo in tem_decimais.items():
    print(f"{col}: exemplo decimal = {exemplo}")

print("\n⚠️ Colunas que têm valores negativos (com exemplo):")
for col, exemplo in tem_negativos.items():
    print(f"{col}: exemplo negativo = {exemplo}")
