import os

BASE_DIR = "data/splits/split_01_interval_10"

conjuntos = ["train", "val", "test"]
classes = ["real", "fake"]

dados = {}
total_geral = 0

# Contagem
for conjunto in conjuntos:
    dados[conjunto] = {}
    total_conjunto = 0

    for classe in classes:
        pasta = os.path.join(BASE_DIR, conjunto, classe)

        quantidade = len([
            arq for arq in os.listdir(pasta)
            if arq.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        dados[conjunto][classe] = quantidade
        total_conjunto += quantidade

    dados[conjunto]["total"] = total_conjunto
    total_geral += total_conjunto

# Exibição
for conjunto in conjuntos:
    total = dados[conjunto]["total"]
    porcentagem = (total / total_geral) * 100

    print(f"\n=== {conjunto.upper()} ===")
    print(f"Real: {dados[conjunto]['real']}")
    print(f"Fake: {dados[conjunto]['fake']}")
    print(f"Total: {total}")
    print(f"Percentual: {porcentagem:.2f}%")

print(f"\nTOTAL GERAL: {total_geral}")