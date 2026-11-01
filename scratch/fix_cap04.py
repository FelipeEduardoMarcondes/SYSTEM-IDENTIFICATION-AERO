import os

path = r'tcc/capitulos/cap_04.tex'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the corrupted text segment with the clean text
corrupted = """Para comprovar que a acurácia preditiva se mantém frente a diferentes perfis operacionais, o modelo foi submetido a ensaios independentes de validação envolvendo transitórios abruptos e varredura espectral contínua:

A Figura~\\ref{fig:narx_val_steps} ilustra a simulação livre do modelo NARX frente a uma sequência experimental de degraus múltiplos na faixa de ^\\circ$ a ^\\circ$, com duração superior a 125\\,s (12.656 passos de amostragem a 100\\,Hz).

\\begin{figure}[H]
    \\centering
    \\caption{Simulação livre (\\textit{free-run}) do modelo NARX frente a uma sequência de degraus de validação independente em torno de ^\\circ$.}
    \\label{fig:narx_val_steps}
    \\includegraphics[width=0.95\\textwidth]{img/resultados/narx_val_steps.png}
    \\sourceFig{o autor, 2026.}
\\end{figure}

Nesse ensaio de degraus, o modelo obteve $\\text{RMSE} = 5{,}678^\\circ$, $\\text{MAE} = 4{,}238^\\circ$ e índice de aderência $\\text{FIT} = 74{,}7\\%$ (^2 = 0{,}936$). O modelo reproduz com fidelidade tanto a velocidade da resposta transitória quanto os patamares de regime permanente, sem apresentar divergência numérica ao longo de todo o horizonte temporal.

Em seguida, avaliou-se a resposta em frequência por meio de um ensaio com sinal de varredura senoidal (\\textit{chirp}) em torno de ^\\circ$ com amplitude de $\\pm 35^\\circ$ cobrindo o espectro até {,}50$\\,Hz, conforme apresentado na Figura~\\ref{fig:narx_val_chirp}.

\\begin{figure}[H]
    \\centering
    \\caption{Simulação livre (\\textit{free-run}) do modelo NARX frente ao sinal de varredura (\\textit{chirp}) com amplitude de $\\pm 35^\\circ$ em torno de ^\\circ$.}
    \\label{fig:narx_val_chirp}
    \\includegraphics[width=0.95\\textwidth]{img/resultados/narx_val_chirp.png}
    \\sourceFig{o autor, 2026.}
\\end{figure}

No ensaio de \\textit{chirp}, o modelo obteve $\\text{RMSE} = 6{,}270^\\circ$, $\\text{MAE} = 5{,}117^\\circ$ e $\\text{FIT} = 72{,}2\\%$ (^2 = 0{,}922$). Observa-se que, à medida que a frequência instantânea de excitação se eleva e atinge a largura de banda do sistema, o modelo NARX preserva o casamento estrito de fase com as medições experimentais da IMU e reproduz a atenuação de amplitude característica da dinâmica aeroelástica. Essa consistência tanto no regime transitório quanto em varreduras harmônicas consolida o modelo polinomial como uma base preditiva sólida para a formulação do controle preditivo ótimo."""

clean = """Para comprovar que a acurácia preditiva se mantém frente a diferentes perfis operacionais, o modelo foi submetido a ensaios independentes de validação envolvendo transitórios abruptos e varredura espectral contínua:

A Figura~\\ref{fig:narx_val_steps} ilustra a simulação livre do modelo NARX frente a uma sequência experimental de degraus múltiplos na faixa de $25^\\circ$ a $65^\\circ$, com duração superior a 125\\,s (12.656 passos de amostragem a 100\\,Hz).

\\begin{figure}[H]
    \\centering
    \\caption{Simulação livre (\\textit{free-run}) do modelo NARX frente a uma sequência de degraus de validação independente em torno de $45^\\circ$.}
    \\label{fig:narx_val_steps}
    \\includegraphics[width=0.95\\textwidth]{img/resultados/narx_val_steps.png}
    \\sourceFig{o autor, 2026.}
\\end{figure}

Nesse ensaio de degraus, o modelo obteve $\\text{RMSE} = 5{,}678^\\circ$, $\\text{MAE} = 4{,}238^\\circ$ e índice de aderência $\\text{FIT} = 74{,}7\\%$ ($R^2 = 0{,}936$). O modelo reproduz com fidelidade tanto a velocidade da resposta transitória quanto os patamares de regime permanente, sem apresentar divergência numérica ao longo de todo o horizonte temporal.

Em seguida, avaliou-se a resposta em frequência por meio de um ensaio com sinal de varredura senoidal (\\textit{chirp}) em torno de $45^\\circ$ com amplitude de $\\pm 35^\\circ$ cobrindo o espectro até $0{,}50$\\,Hz, conforme apresentado na Figura~\\ref{fig:narx_val_chirp}.

\\begin{figure}[H]
    \\centering
    \\caption{Simulação livre (\\textit{free-run}) do modelo NARX frente ao sinal de varredura (\\textit{chirp}) com amplitude de $\\pm 35^\\circ$ em torno de $45^\\circ$.}
    \\label{fig:narx_val_chirp}
    \\includegraphics[width=0.95\\textwidth]{img/resultados/narx_val_chirp.png}
    \\sourceFig{o autor, 2026.}
\\end{figure}

No ensaio de \\textit{chirp}, o modelo obteve $\\text{RMSE} = 6{,}270^\\circ$, $\\text{MAE} = 5{,}117^\\circ$ e $\\text{FIT} = 72{,}2\\%$ ($R^2 = 0{,}922$). Observa-se que, à medida que a frequência instantânea de excitação se eleva e atinge a largura de banda do sistema, o modelo NARX preserva o casamento estrito de fase com as medições experimentais da IMU e reproduz a atenuação de amplitude característica da dinâmica aeroelástica. Essa consistência tanto no regime transitório quanto em varreduras harmônicas consolida o modelo polinomial como uma base preditiva sólida para a formulação do controle preditivo ótimo."""

# normalize
text_norm = text.replace('\r\n', '\n')
corrupted_norm = corrupted.replace('\r\n', '\n')
clean_norm = clean.replace('\r\n', '\n')

assert corrupted_norm in text_norm, 'Corrupted block not found!'

new_text = text_norm.replace(corrupted_norm, clean_norm)
with open(path, 'w', encoding='utf-8', newline='') as f:
    f.write(new_text)

print('SUCCESS: Clean text written without any corruption!')

