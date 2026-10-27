import json

base = 'sysid/notebooks/identificação-collab/IDENT-FELIPE/'
for fname in ['modelo_greybox_v1.json', 'modelo_greybox_v7.json', 'modelo_greybox_v8.json']:
    try:
        with open(base + fname, 'r', encoding='utf-8') as f:
            d = json.load(f)
            print(f'=== {fname} ===')
            for k in ['u0_static', 'tau_acc_fixed', 'tau_m_fixed', 'k_static']:
                if k in d:
                    print(f"  {k}: {d[k]}")
            if 'models' in d:
                for mk, mv in d['models'].items():
                    print(f"  Model: {mk}")
                    if isinstance(mv, dict):
                        for pk in ['theta', 'params', 'c1', 'c2', 'c3', 'c4', 'Fc']:
                            if pk in mv:
                                print(f"    {pk}: {mv[pk]}")
            elif 'theta' in d:
                print('  theta:', d['theta'])
    except Exception as e:
        print(f'=== {fname} === Error: {e}')
