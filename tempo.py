from pathlib import Path
import json

ROOT = Path(r"C:\tcc mesonet\experiments")

total_segundos = 0

for json_file in ROOT.rglob("*.json"):

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "total_training_time_sec" in data:
            total_segundos += float(data["total_training_time_sec"])

    except:
        pass

horas = total_segundos / 3600

print(f"Tempo total: {total_segundos:.2f} segundos")
print(f"Tempo total: {horas:.2f} horas")