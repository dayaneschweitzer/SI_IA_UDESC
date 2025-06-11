import pickle

# Carregar o metadados.pkl
with open("vetores/metadados.pkl", "rb") as f:
    metadados = pickle.load(f)

# Verificar se algum chunk da 032 fala 'desligamento'
found = False
for m in metadados:
    if "032" in m["arquivo"]:
        if "desligamento" in m["texto"].lower():
            print(">>>> Encontrado chunk relevante!")
            print(f"Arquivo: {m['arquivo']}, Chunk ID: {m['chunk_id']}")
            print(m["texto"])
            print("=" * 80)
            found = True

if not found:
    print(">>> Nenhum chunk da Resolução 032 com 'desligamento' foi encontrado.")
