import pandas as pd
import numpy as np

def readData(dataset_name="multiseno", decimar=1, return_ref=False, start_idx=0, end_idx=None, start_time=None, end_time=None):
    """
    Funcao oficial para baixar os dados do Aeropendulo direto do GitHub.
    
    Parametros
    ----------
    dataset_name : str
        Nome do ensaio ("multiseno", "degraus", "chirp", "mpc_100hz", etc)
    decimar : int
        Fator de subamostragem (1 = 100 Hz, 2 = 50 Hz, 5 = 20 Hz, etc)
    return_ref : bool
        Se True, retorna a referencia (se disponivel) como 4o elemento.
    start_idx : int
        Indice inicial de corte do dataset (padrao 0).
    end_idx : int
        Indice final de corte do dataset (padrao None).
    start_time : float
        Tempo inicial em segundos para o corte (sobrepoe start_idx se fornecido).
    end_time : float
        Tempo final em segundos para o corte (sobrepoe end_idx se fornecido).
        
    Retorna
    -------
    y, u, t (se return_ref=False)
    y, u, t, ref (se return_ref=True)
    """
    
    import os
    import subprocess
    
    # URL base para os raw files do repositorio (dinâmico baseado na branch atual)
    branch = "main"
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=current_dir, stderr=subprocess.DEVNULL).decode('utf-8').strip()
        if not branch:
            branch = "main"
    except Exception:
        pass
    
    base_url = f"https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/{branch}/"
    
    urls = {
        "multiseno": "sysid_multiseno_validation_ref_0909_14-42.csv",
        "degraus": "degraus_0908_23-01.csv",
        "chirp": "chirp-60-amp50_0908_23-09.csv",
        "mpc_100hz": "referencia_mpc_0920_19-19.csv",
        "multiseno_2": "multi-seno-60-030Hz_0908_23-13.csv"
    }
    
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_root, "data", "experimentos", "coleta_dados")
    
    normalized_name = dataset_name.replace('\\', '/')
    local_path = ""
    url = ""
    
    # Se for uma URL remota direta
    if normalized_name.startswith("http://") or normalized_name.startswith("https://"):
        url = normalized_name
    else:
        # Se for um dos apelidos tabelados
        rel_target = urls.get(normalized_name, normalized_name)
        
        # Remove prefixos redundantes caso o usuário tenha passado o caminho relativo completo
        for prefix in ["data/experimentos/coleta_dados/", "data/experimentos/", "coleta_dados/"]:
            if rel_target.startswith(prefix):
                rel_target = rel_target[len(prefix):]
                break
                
        # 1. Candidato direto dentro de data/experimentos/coleta_dados/
        candidate = os.path.join(data_dir, rel_target)
        if os.path.exists(candidate) and os.path.isfile(candidate):
            local_path = candidate
        elif os.path.isabs(dataset_name) and os.path.exists(dataset_name):
            local_path = dataset_name
        else:
            # 2. Busca recursiva EXCLUSIVAMENTE dentro de data/experimentos/coleta_dados
            base_fname = os.path.basename(rel_target)
            for root_folder, _, files in os.walk(data_dir):
                if base_fname in files:
                    local_path = os.path.join(root_folder, base_fname)
                    break
        
        # URL remota no GitHub baseada na pasta canônica
        if local_path and os.path.exists(local_path):
            rel_github = os.path.relpath(local_path, repo_root).replace('\\', '/')
        else:
            rel_github = f"data/experimentos/coleta_dados/{rel_target}"
        url = base_url + rel_github

    try:
        if local_path and os.path.exists(local_path):
            print(f"Carregando dataset '{dataset_name}' localmente ({local_path})...")
            df = pd.read_csv(local_path, on_bad_lines='skip')
        else:
            print(f"Baixando dataset '{dataset_name}' do GitHub ({url})...")
            df = pd.read_csv(url, on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar '{dataset_name}': {e}")
        
    # Tempo em segundos
    if 'tempo_ms' in df.columns:
        tempo_s = df['tempo_ms'].values / 1000.0
    elif 'tempo_s' in df.columns:
        tempo_s = df['tempo_s'].values
    else:
        tempo_s = np.arange(len(df)) * 0.01 # Assume 100 Hz como padrao
        
    # Controle
    if 'u_pct' in df.columns:
        u = df['u_pct'].values
    elif 'motor_percent' in df.columns:
        u = df['motor_percent'].values
    else:
        u = np.zeros(len(df))
        
    # Saida
    y = df['angulo_deg'].values
    
    # Referencia
    ref = np.array([])
    if 'referencia' in df.columns:
        ref = df['referencia'].values
    elif 'referencia_deg' in df.columns:
        ref = df['referencia_deg'].values
        
    # Converte tempos (se fornecidos) para indices
    if start_time is not None:
        start_idx = np.searchsorted(tempo_s, start_time)
    if end_time is not None:
        end_idx = np.searchsorted(tempo_s, end_time)
        
    # Corte inicial e final (Trimming)
    u = u[start_idx:end_idx]
    y = y[start_idx:end_idx]
    tempo_s = tempo_s[start_idx:end_idx]
    if len(ref) > 0:
        ref = ref[start_idx:end_idx]
        
    # Subamostragem (Decimacao por slicing simples para manter degraus)
    if decimar > 1:
        u = u[::decimar]
        y = y[::decimar]
        tempo_s = tempo_s[::decimar]
        if len(ref) > 0:
            ref = ref[::decimar]
            
    if return_ref:
        return y, u, tempo_s, ref
    return y, u, tempo_s

def list_datasets():
    """Lista os datasets registrados no aerodata."""
    return ["multiseno", "degraus", "chirp", "mpc_100hz", "multiseno_2"]
