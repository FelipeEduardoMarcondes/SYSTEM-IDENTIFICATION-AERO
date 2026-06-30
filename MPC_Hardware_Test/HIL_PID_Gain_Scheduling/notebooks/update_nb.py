import json

with open("pid_gain_scheduling_14_drone.ipynb", "r") as f:
    nb = json.load(f)

# Create the new cell
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json\n",
        "import os\n",
        "\n",
        "gains = {\n",
        "    \"SCHED_NODES\": angles_deg.tolist(),\n",
        "    \"KP_NODES\": Kp_list,\n",
        "    \"KI_NODES\": Ki_list,\n",
        "    \"KD_NODES\": Kd_list,\n",
        "    \"VEQ_NODES\": v_eq_list\n",
        "}\n",
        "\n",
        "out_path = os.path.join(\"..\", \"pid_gains.json\")\n",
        "with open(out_path, \"w\") as f:\n",
        "    json.dump(gains, f, indent=4)\n",
        "print(f\"Ganhos salvos em {out_path}\")\n"
    ]
}

# Append cell if it doesn't already exist
found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "pid_gains.json" in "".join(cell.get("source", [])):
        found = True
        break

if not found:
    nb["cells"].append(new_cell)
    with open("pid_gain_scheduling_14_drone.ipynb", "w") as f:
        json.dump(nb, f, indent=2)
    print("Celular adicionada ao notebook!")
else:
    print("Celular ja existia no notebook.")
