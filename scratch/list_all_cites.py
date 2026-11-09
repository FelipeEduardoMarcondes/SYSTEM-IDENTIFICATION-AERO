import re

def extract_citations(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    citations = []
    for idx, line in enumerate(lines, 1):
        cites = re.findall(r'\\cite(?:online)?\{([^}]+)\}', line)
        for c in cites:
            keys = [k.strip() for k in c.split(',')]
            for k in keys:
                citations.append((idx, k, line.strip()))
    return citations

print("--- CAP 01 ---")
for line_no, key, line_text in extract_citations("tcc/capitulos/cap_01.tex"):
    print(f"L{line_no}: {key}")

print("\n--- CAP 02 ---")
for line_no, key, line_text in extract_citations("tcc/capitulos/cap_02.tex"):
    print(f"L{line_no}: {key}")

