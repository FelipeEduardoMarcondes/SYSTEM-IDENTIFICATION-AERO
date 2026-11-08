import os

tex_path = r"c:\Users\vicio\Documents\AEROPENDULO\tcc\capitulos\cap_04.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()

target = """    \\item \\textbf{Comportamento da Ação de Controle:} A taxa média de variação da ação de controle por ciclo de amostragem ($|\\Delta u|$ a cada $10$\\,ms) registrou valores de $0{,}1036\\%$ (Validação~1), $0{,}1276\\%$ (Validação~2) e $0{,}0298\\%$ (Validação~3), níveis equivalentes aos observados no ensaio nominal ($0{,}1027\\%$, $0{,}1426\\%$ e $0{,}0349\\%$). Esses valores indicam que a inclusão do integrador residual com $K_{i,\\text{res}} = 0{,}20$ operou sem introduzir oscilações de alta frequência no sinal de comando.
    
\\end{enumerate}"""

addition = """    \\item \\textbf{Comportamento da Ação de Controle:} A taxa média de variação da ação de controle por ciclo de amostragem ($|\\Delta u|$ a cada $10$\\,ms) registrou valores de $0{,}1036\\%$ (Validação~1), $0{,}1276\\%$ (Validação~2) e $0{,}0298\\%$ (Validação~3), níveis equivalentes aos observados no ensaio nominal ($0{,}1027\\%$, $0{,}1426\\%$ e $0{,}0349\\%$). Esses valores indicam que a inclusão do integrador residual com $K_{i,\\text{res}} = 0{,}20$ operou sem introduzir oscilações de alta frequência no sinal de comando.
    
\\end{enumerate}

\\subsection{Rejeição de Perturbações Externas na Bancada}
\\label{subsec:rejeicao_disturbios}

Para avaliar a capacidade de recuperação da malha fechada frente a perturbações externas não modeladas, submeteu-se a haste do aeropêndulo a perturbações mecânicas manuais (deslocamentos forçados e tentativas de emperramento) durante a regulação estática em $60^\\circ$. A Figura~\\ref{fig:rejeicao_disturbio} apresenta a resposta temporal correspondente na janela de atuação do controlador neural ($5{,}0$\\,s a $69{,}0$\\,s, suprimindo as fases de partida e recolhimento sob PID).

\\begin{figure}[H]
    \\centering
    \\caption{Resposta experimental da bancada física à rejeição de perturbações externas em $60^\\circ$.}
    \\label{fig:rejeicao_disturbio}
    \\includegraphics[width=0.95\\textwidth]{img/resultados/rejeicao_disturbio_bancada.png}
    \\sourceFig{o autor, 2026.}
\\end{figure}

Observa-se que, a cada perturbação descendente (emperramento da haste com deflexões de até $-17^\\circ$), a ação de controle reage elevando o comando $u(t)$ até cerca de $68\\%$ de PWM para restaurar o torque propulsivo. Nas perturbações ascendentes (deslocamentos manuais de até $+15{,}4^\\circ$), o comando de controle é reduzido para aproximadamente $20\\%$, permitindo a descida por gravidade. Em todas as ocorrências, o sistema retornou à referência de $60^\\circ$ em aproximadamente $1{,}0$\\,s a $1{,}8$\\,s após a cessação do distúrbio, sem oscilações sustentadas e com erro estacionário nulo."""

if target in content:
    content = content.replace(target, addition)
    print("Sucesso (LF)")
elif target.replace("\n", "\r\n") in content:
    content = content.replace(target.replace("\n", "\r\n"), addition.replace("\n", "\r\n"))
    print("Sucesso (CRLF)")
else:
    print("ERRO: target não encontrado")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(content)

print("cap_04.tex atualizado.")

