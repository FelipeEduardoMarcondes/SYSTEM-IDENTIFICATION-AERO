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
    
    # URL base para os raw files do repositorio
    base_url = "https://raw.githubusercontent.com/FelipeEduardoMarcondes/SYSTEM-IDENTIFICATION-AERO/main/"
    
    urls = {
        "multiseno": base_url + "data/experimentos/sysid_multiseno_validation_ref_0909_14-42.csv",
        "degraus": base_url + "data/experimentos/degraus_0908_23-01.csv",
        "chirp": base_url + "data/experimentos/chirp-60-amp50_0908_23-09.csv",
        "mpc_100hz": base_url + "data/experimentos/referencia_mpc_0920_19-19.csv",
        "multiseno_2": base_url + "data/experimentos/multi-seno-60-030Hz_0908_23-13.csv"
    }
    
    if dataset_name in urls:
        url = urls[dataset_name]
    elif dataset_name.startswith("http"):
        url = dataset_name
    elif dataset_name.endswith(".csv"):
        # Trata caminhos com subpastas (ex: "RODADA-1/ensaio.csv" ou "data/controle/ref.csv")
        # Se o usuario ja especificou "data/", vai direto da raiz do repo
        if dataset_name.startswith("data/"):
            url = base_url + dataset_name
        else:
            # Senao, assume que esta dentro de data/experimentos/ por padrao
            url = base_url + "data/experimentos/" + dataset_name
    else:
        raise ValueError(f"Dataset nao encontrado. Escolha um destes: {list(urls.keys())}\n"
                         "Ou passe o nome do arquivo exato (ex: 'ensaio.csv') ou a URL.")
    
    try:
        print(f"Baixando dataset '{dataset_name}' do GitHub...")
        df = pd.read_csv(url, on_bad_lines='skip')
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar {url}: {e}")
        
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
