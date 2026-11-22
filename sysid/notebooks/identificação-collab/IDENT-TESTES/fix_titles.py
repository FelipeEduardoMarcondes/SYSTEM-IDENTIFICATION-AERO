import sys
import os

file_path = r'c:\Users\vicio\Documents\AEROPENDULO\sysid\notebooks\identificação-collab\IDENT-HELON\mpc_v4.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace buggy characters
content = content.replace('â€”', '-')
content = content.replace('RÂ²', 'R2')
content = content.replace('RA²', 'R2')
content = content.replace('Â²', '2')
content = content.replace('Ã²', '2')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Buggy titles fixed.')
