import re

with open("tcc/bibliografia.bib", "r", encoding="utf-8") as f:
    bib_text = f.read()

keys = [
    "nguyen2020", "lima2025", "mayne2000", "karg2020", "david2021",
    "bennett1996", "bemporad2002", "alsmeier2024", "chen2018approximating",
    "hertneck2018", "hose2025approximate", "aguirre2007", "jiang2022",
    "jacob2018quantization", "banbury2021mlperf", "hewing2020learning",
    "nubert2020safe", "drgona2022differentiable", "sanyal2024", "li2024"
]

for key in keys:
    pattern = rf"@(\w+)\s*\{{\s*{key}\s*,([\s\S]*?)(?=\n@|\Z)"
    m = re.search(pattern, bib_text)
    if m:
        entry_type = m.group(1)
        fields = m.group(2).strip()
        print(f"=== KEY: {key} (Type: @{entry_type}) ===")
        # extract year, author, title, journal/booktitle
        y = re.search(r'year\s*=\s*\{?([^,}\n]+)\}?', fields, re.I)
        t = re.search(r'title\s*=\s*\{?([^}\n]+)\}?', fields, re.I)
        a = re.search(r'author\s*=\s*\{?([^}\n]+)\}?', fields, re.I)
        print(f"  Title: {t.group(1) if t else 'N/A'}")
        print(f"  Author: {a.group(1) if a else 'N/A'}")
        print(f"  Year in bib: {y.group(1) if y else 'N/A'}")
    else:
        print(f"=== KEY: {key} NOT FOUND IN BIB ===")

