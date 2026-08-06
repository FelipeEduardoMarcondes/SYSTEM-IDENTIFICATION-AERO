import json

notebook_path = r'c:\Users\vicio\Documents\AEROPENDULO\identificação-collab\NODE_AEROPENDULO_v2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- Salvando os modelos na máquina ---\n",
        "import torch\n",
        "\n",
        "caminho_cinza = 'node_caixa_cinza.pth'\n",
        "torch.save(phys_model.state_dict(), caminho_cinza)\n",
        "print(f'Modelo Físico (Caixa-Cinza) salvo em: {caminho_cinza}')\n",
        "\n",
        "caminho_preta = 'node_caixa_preta.pth'\n",
        "torch.save(bb_model.state_dict(), caminho_preta)\n",
        "print(f'Modelo Neural (Caixa-Preta) salvo em: {caminho_preta}')"
    ]
}

nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Célula de salvamento adicionada com sucesso!")

