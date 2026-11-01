import os
import re

# 1. Update tcc/capitulos/cap_03.tex
cap03_path = 'tcc/capitulos/cap_03.tex'
with open(cap03_path, 'r', encoding='utf-8') as f:
    c3 = f.read()

# Replace motor/helice in line 19
c3 = c3.replace(
    'haste pendular contendo o motor \\textit{brushless} (BLDC) acoplado.',
    'haste pendular contendo a unidade propulsora acoplada.'
)

# Replace paragraph in line 30
target_p30 = '''A haste rosqueada também permite a fixação e remoção facilitada de contrapesos, que foram usinados e rosqueados para se encaixar diretamente na haste. Um suporte impresso em 3D acopla o motor BLDC à extremidade da haste. A propulsão é gerada por uma hélice bipá de passo fixo. Um alojamento impresso em 3D para a Unidade de Medição Inercial (IMU) foi desenvolvido e fixado ao eixo de rotação, responsável por capturar o ângulo do sistema. A Figura~\\ref{fig:montagem-bancada} ilustra o formato atual da bancada. Os principais parâmetros construtivos do sistema são resumidos na Tabela~\\ref{tab:componentes_mecanicos}.'''

new_p30 = '''A haste rosqueada também permite a fixação e remoção facilitada de contrapesos, que foram usinados e rosqueados para se encaixar diretamente na haste. Um suporte impresso em 3D acopla a unidade propulsora à extremidade da haste. A propulsão é gerada por uma turbina fan dutada (EDF, do inglês \\textit{Electric Ducted Fan}) da fabricante FMS de 50\\,mm de diâmetro (unidade \\textit{FMS 50mm 11-Blade EDF Unit}), dotada de rotor de 11 pás e acionada por um motor elétrico \\textit{brushless} outrunner 2627 de $4500$\\,KV (compatível com alimentação LiPo de 3S a 4S). O confinamento do fluxo de ar pelo duto cilíndrico reduz perdas de vórtice de ponta de pá, assegura um empuxo colimado e rápido ao longo do plano pendular e confere maior segurança física durante os ensaios de bancada. Um alojamento impresso em 3D para a Unidade de Medição Inercial (IMU) foi desenvolvido e fixado ao eixo de rotação, responsável por capturar o ângulo do sistema. A Figura~\\ref{fig:montagem-bancada} ilustra o formato atual da bancada. Os principais parâmetros construtivos do sistema são resumidos na Tabela~\\ref{tab:componentes_mecanicos}.'''

assert target_p30 in c3, 'target_p30 not found'
c3 = c3.replace(target_p30, new_p30)

# Replace Table 3.1 entries
old_tab = '''        Suporte do motor          & PLA (impressão 3D)                      \\\\
        Alojamento da IMU         & PLA (impressão 3D)                      \\\\
        Motor                     & EMAX RS2306                             \\\\
        Hélice                    & Bipá, passo fixo                        \\\\'''

new_tab = '''        Suporte do propulsor      & PLA (impressão 3D)                      \\\\
        Alojamento da IMU         & PLA (impressão 3D)                      \\\\
        Unidade Propulsora (EDF)  & FMS 50\\,mm, 11 pás (\\textit{Ducted Fan}) \\\\
        Motor Elétrico            & Brushless Outrunner 2627 KV4500 (3S--4S)\\\\'''

assert old_tab in c3, 'old_tab not found'
c3 = c3.replace(old_tab, new_tab)

# Replace line 143 and 145
c3 = c3.replace(
    'arranjo motor-hélice introduz não-linearidades',
    'arranjo propulsor (turbina EDF) introduz não-linearidades'
)
c3 = c3.replace(
    'massa do conjunto motor-hélice $m_1$',
    'massa da unidade propulsora (turbina EDF) $m_1$'
)

# Replace line 480
c3 = c3.replace(
    'induzido pela hélice em altas rotações.',
    'induzido pelo rotor da turbina dutada em altas rotações.'
)

with open(cap03_path, 'w', encoding='utf-8', newline='') as f:
    f.write(c3)
print('Updated cap_03.tex successfully!')

# 2. Update tcc/capitulos/cap_01.tex
cap01_path = 'tcc/capitulos/cap_01.tex'
with open(cap01_path, 'r', encoding='utf-8') as f:
    c1 = f.read()

c1 = c1.replace(
    'pelo torque aerodinâmico do conjunto motor-hélice',
    'pelo empuxo aerodinâmico da turbina fan dutada (EDF)'
)
with open(cap01_path, 'w', encoding='utf-8', newline='') as f:
    f.write(c1)
print('Updated cap_01.tex successfully!')

# 3. Update tcc/capitulos/cap_02.tex
cap02_path = 'tcc/capitulos/cap_02.tex'
with open(cap02_path, 'r', encoding='utf-8') as f:
    c2 = f.read()

c2 = c2.replace(
    'operadas por motores BLDC e hélices, fenômenos como atrito',
    'operadas por turbinas elétricas EDF (\\textit{Electric Ducted Fan}), fenômenos como atrito'
)
with open(cap02_path, 'w', encoding='utf-8', newline='') as f:
    f.write(c2)
print('Updated cap_02.tex successfully!')

# 4. Update tcc/capitulos/cap_04.tex
cap04_path = 'tcc/capitulos/cap_04.tex'
with open(cap04_path, 'r', encoding='utf-8') as f:
    c4 = f.read()

c4 = c4.replace(
    'relação de empuxo da hélice ($T \\propto \\omega^2$)',
    'relação de empuxo aerodinâmico de rotores e turbinas fan ($T \\propto \\omega^2$)'
)
with open(cap04_path, 'w', encoding='utf-8', newline='') as f:
    f.write(c4)
print('Updated cap_04.tex successfully!')

# 5. Update tcc/pretextual.tex
pre_path = 'tcc/pretextual.tex'
with open(pre_path, 'r', encoding='utf-8') as f:
    cp = f.read()

# Add EDF to siglas
if r'\item[EDF]' not in cp:
    target_sigla = r'\item[DLPF] Filtro Passa-Baixas Digital (\textit{Digital Low-Pass Filter})'
    edf_sigla = target_sigla + '\n    \\item[EDF] Ventilador Elétrico Dutado (\\textit{Electric Ducted Fan})'
    cp = cp.replace(target_sigla, edf_sigla)

# Update symbols
cp = cp.replace(
    r'\item[$ L_1 $] Distância do conjunto motor-hélice ao pivô de rotação',
    r'\item[$ L_1 $] Distância da unidade propulsora (turbina EDF) ao pivô de rotação'
)
cp = cp.replace(
    r'\item[$ m_1 $] Massa do conjunto motor-hélice',
    r'\item[$ m_1 $] Massa da unidade propulsora (turbina EDF)'
)

# Update resumo / abstract
cp = cp.replace(
    'O sistema, operado por um motor BLDC e monitorado por uma IMU',
    'O sistema, impulsionado por uma turbina fan dutada (EDF) acionada por motor BLDC e monitorado por uma IMU'
)
cp = cp.replace(
    'The system, driven by a BLDC motor and monitored by an IMU',
    'The system, propelled by an Electric Ducted Fan (EDF) unit driven by a BLDC motor and monitored by an IMU'
)

with open(pre_path, 'w', encoding='utf-8', newline='') as f:
    f.write(cp)
print('Updated pretextual.tex successfully!')

print('ALL FILES UPDATED WITH EDF SPECIFICATIONS!')

