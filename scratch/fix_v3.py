import json

notebook_path = "c:/Users/vicio/Documents/AEROPENDULO/identificação-collab/ANN_NARX_AEROPENDULO_v3.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The new cell source
model_source = [
    "class NARXModel(nn.Module):\n",
    "    def __init__(self, ny, nu, hidden_dim1=256, hidden_dim2=128, dropout=0.1):\n",
    "        super().__init__()\n",
    "        self.linear_bypass = nn.Linear(ny + nu, 1)\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(ny + nu, hidden_dim1),\n",
    "            nn.Tanh(),\n",
    "            nn.Dropout(dropout),\n",
    "            nn.Linear(hidden_dim1, hidden_dim2),\n",
    "            nn.Tanh(),\n",
    "            nn.Dropout(dropout),\n",
    "            nn.Linear(hidden_dim2, 1)\n",
    "        )\n",
    "    \n",
    "    def forward(self, x):\n",
    "        return self.linear_bypass(x) + self.net(x)\n",
    "    \n",
    "    def predict(self, x):\n",
    "        self.eval()\n",
    "        with torch.no_grad():\n",
    "            return self.forward(x)\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": model_source
}

target_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "model = NARXModel(ny, nu, 128, 64).to(device)" in source:
            target_idx = i
            break

if target_idx != -1:
    # Check if the cell right before it is already the class definition
    prev_source = "".join(nb["cells"][target_idx - 1].get("source", []))
    if "class NARXModel(nn.Module):" not in prev_source:
        nb["cells"].insert(target_idx, new_cell)
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Successfully inserted NARXModel cell.")
    else:
        print("NARXModel cell already exists before the target cell.")
else:
    print("Could not find the target cell to insert before.")
