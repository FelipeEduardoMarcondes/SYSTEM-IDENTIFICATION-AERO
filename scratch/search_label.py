import os

root = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main'
for dirpath, dirnames, filenames in os.walk(root):
    if '.git' in dirpath: continue
    for f in filenames:
        if f.endswith(('.py', '.sh', '.bat', '.m', '.ipynb', '.tex')):
            p = os.path.join(dirpath, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    if 'STM32 em C (SIL)' in content:
                        print("Found in:", p)
            except Exception:
                pass
