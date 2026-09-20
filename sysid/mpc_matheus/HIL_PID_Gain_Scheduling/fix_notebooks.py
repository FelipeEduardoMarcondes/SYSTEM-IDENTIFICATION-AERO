import os
import json
import re

for fname in os.listdir("notebooks"):
    if fname.endswith(".ipynb"):
        fpath = os.path.join("notebooks", fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        
        changed = False
        for cell in data.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell["source"])
                if "def get_ref(t):" in source and "step = int(t" in source:
                    # Usar regex para ser flexível
                    pattern = r"def get_ref\(t\):.*?return max[^}]*"
                    new_ref = "def get_ref(t):\n    if t <= 95.0:\n        return (180.0 / 95.0) * t\n    else:\n        return max(180.0 - (180.0 / 95.0) * (t - 95.0), 0.0)"
                    new_source = re.sub(pattern, new_ref, source, flags=re.DOTALL)
                    if new_source != source:
                        # Split by newline but keep newline character using a list comprehension
                        cell["source"] = [line + '\n' for line in new_source.split('\n')]
                        # Remover a última newline adicionada para ser exato
                        cell["source"][-1] = cell["source"][-1][:-1]
                        changed = True

        if changed:
            with open(fpath, "w") as f:
                json.dump(data, f, indent=1)
            print(f"Updated {fpath}")
            os.system(f"jupyter nbconvert --execute --to notebook --inplace {fpath}")
