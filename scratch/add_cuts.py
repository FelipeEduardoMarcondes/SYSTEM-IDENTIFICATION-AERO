import json
import re

def add_cuts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Sub function definition
            if 'def carregar_experimento' in source:
                new_def = """def carregar_experimento(url, decimacao=1, start_idx=0, end_idx=None):
    df = pd.read_csv(url)
    df_sub = df.iloc[::decimacao].copy().reset_index(drop=True)
    
    # Suporte para cabeçalhos antigos (motor_percent) e novos (u_pct)
    if 'u_pct' in df_sub.columns:
        u_raw = df_sub['u_pct'].values
    else:
        u_raw = df_sub['motor_percent'].values
        
    y_raw = df_sub['angulo_deg'].values
    
    if end_idx is None:
        end_idx = len(u_raw)
        
    u = u_raw[start_idx:end_idx]
    y = y_raw[start_idx:end_idx]
    return u, y"""
                source = re.sub(
                    r"def carregar_experimento\(url, decimacao=1\):.*?return u, y", 
                    new_def, 
                    source, 
                    flags=re.DOTALL
                )

            # Sub call for BLS
            if 'BLS' in filepath:
                bls_target = r"u_TRA,\s*y_TRA\s*=\s*carregar_experimento\(BASE2 \+ \"RODADA-2/multi-seno-2_0804_19-31\.csv\"\).*?\nu_TEST,\s*y_TEST\s*=\s*carregar_experimento\(BASE2 \+ \"multi-seno-1_0807_16-57\.csv\"\).*?"
                new_bls = """# Define limites de corte para remover ruído inicial e final
CUT_TRA_START = 2000
CUT_TRA_END = -1500

CUT_TEST_START = 2000
CUT_TEST_END = -1000

u_TRA,  y_TRA  = carregar_experimento(BASE2 + "RODADA-2/multi-seno-2_0804_19-31.csv", start_idx=CUT_TRA_START, end_idx=CUT_TRA_END) # Estimação
u_TEST, y_TEST = carregar_experimento(BASE2 + "multi-seno-1_0807_16-57.csv", start_idx=CUT_TEST_START, end_idx=CUT_TEST_END) # Teste """
                source = re.sub(bls_target, new_bls, source, flags=re.DOTALL)
                
            # Sub call for RLS
            if 'RLS' in filepath:
                # First clean up the double assignment
                rls_target = r"u_TRA,\s*y_TRA\s*=\s*carregar_experimento\(BASE2 \+ \"RODADA-2/multi-seno-1_0804_19-03\.csv\"\)\nu_TEST,\s*y_TEST\s*=\s*carregar_experimento\(BASE2 \+ \"RODADA-2/multi-seno-2_0804_19-28\.csv\"\)\n\nu_TRA,\s*y_TRA\s*=\s*carregar_experimento\(BASE2 \+ \"multi-seno-2_0807_17-13\.csv\"\).*?\nu_TEST,\s*y_TEST\s*=\s*carregar_experimento\(BASE2 \+ \"multi-seno-1_0807_16-57\.csv\"\).*?"
                new_rls = """# Define limites de corte para remover ruído inicial e final
CUT_TRA_START = 2000
CUT_TRA_END = -1500

CUT_TEST_START = 2000
CUT_TEST_END = -1000

u_TRA,  y_TRA  = carregar_experimento(BASE2 + "RODADA-2/multi-seno-2_0804_19-31.csv", start_idx=CUT_TRA_START, end_idx=CUT_TRA_END) # Estimação
u_TEST, y_TEST = carregar_experimento(BASE2 + "multi-seno-1_0807_16-57.csv", start_idx=CUT_TEST_START, end_idx=CUT_TEST_END) # Teste """
                source = re.sub(rls_target, new_rls, source, flags=re.DOTALL)
                
            cell['source'] = [line + '\n' if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

add_cuts('identificação-collab/BLS_AEROPENDULO.ipynb')
add_cuts('identificação-collab/RLS_AEROPENDULO.ipynb')
