import json

fpath = "notebooks/mpc_grey_box_14_drone.ipynb"
with open(fpath, "r") as f:
    data = json.load(f)

for cell in data.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell["source"])
        if "def get_ref(t):" in source and "x2ref_full" not in source:
            cell["source"].append("\nx2ref_full = np.array([get_ref(t) for t in t_steps])\n")
            cell["source"].append("ref_signal = x2ref_full[:steps]\n")

with open(fpath, "w") as f:
    json.dump(data, f, indent=1)
