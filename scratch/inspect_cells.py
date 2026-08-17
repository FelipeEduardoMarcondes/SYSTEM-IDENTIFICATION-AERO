import json

for nb_name in ['ANN_NARX_AEROPENDULO.ipynb']:
    print(f"\n==== {nb_name} ====")
    with open('identificação-collab/' + nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code' and 'carregar_experimento' in "".join(cell['source']):
                print(f"Cell {i}:")
                print("".join(cell['source']))
