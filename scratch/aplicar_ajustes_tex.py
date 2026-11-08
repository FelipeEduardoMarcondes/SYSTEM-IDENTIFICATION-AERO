import os

tex_path = r"c:\Users\vicio\Documents\AEROPENDULO\tcc\capitulos\cap_04.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replacement 1: Subsection 4.4 items and boxplot figure
old_sec4_4 = """A análise comparativa do controlador híbrido frente ao controlador nominal revelou ganhos expressivos em todos os quesitos de avaliação:

\\begin{enumerate}
    \\item \\textbf{Atenuação Expressiva do Erro Estacionário (\\textit{Bias}):} Na Validação~2 (degraus aleatórios), o erro médio persistente de regime permanente reduziu-se de $+2{,}67^\\circ$ no controlador nominal para apenas ${+0{,}07^\\circ}$ no controlador híbrido, correspondendo a uma redução de ${97{,}5\\%}$. Na Validação~1, o viés médio diminuiu de $+2{,}29^\\circ$ para ${+0{,}03^\\circ}$ (${98{,}8\\%}$ de redução). Nos patamares de degrau, o braço do aeropêndulo aproximou-se da linha de referência, indicando que o integrador compensou os desvios estáticos de sustentação e pequenas assimetrias aerodinâmicas da planta.
    
    \\item \\textbf{Atenuação do Atrito Estático (\\textit{Stick-Slip}) na Senoide:} Na Validação~3 (Figura~\\ref{fig:cmp_hibrido_val3}), os patamares horizontais de travamento mecânico foram substancialmente atenuados. O integrador residual acumulou suavemente torque propulsivo nos instantes de velocidade nula, auxiliando a vencer o atrito estático do rolamento sem ocasionar sobressinais expressivos. Como resultado direto, o RMSE na trajetória senoidal caiu de $3{,}55^\\circ$ para ${1{,}75^\\circ}$, o que representa uma redução de ${50{,}8\\%}$.
    
    \\item \\textbf{Redução Global do Erro de Rastreamento:} Na Validação~1, o RMSE angular foi reduzido de $3{,}13^\\circ$ para ${2{,}20^\\circ}$ (melhoria de ${29{,}8\\%}$). Na Validação~2, o RMSE caiu de $4{,}61^\\circ$ para ${3{,}60^\\circ}$ (melhoria de ${21{,}8\\%}$), atenuando desvios nos degraus de amplitude intermediária.
    
    \\item \\textbf{Preservação da Suavidade de Controle:} A taxa média de variação da ação de controle permaneceu contida entre $0{,}04\\%$ e $0{,}11\\%$/passo, patamar indistinguível do ensaio puramente nominal. Isso sugere que o ganho $K_{i,\\text{res}} = 0{,}20$ operou em regime estável e amortecido, sem introduzir comutações de alta frequência ou degradar a integridade do atuador.
    
\\end{enumerate}"""

new_sec4_4 = """Para avaliar a repetibilidade e a consistência estatística do controlador híbrido na bancada física, analisou-se um conjunto de $N = 10$ ensaios experimentais independentes para cada trajetória de validação (composto pelo ensaio inicial e 9 repetições adicionais), avaliados na janela ativa de atuação da rede neural ($5{,}0$\\,s a $70{,}0$\\,s). A Figura~\\ref{fig:boxplot_reteste} apresenta a distribuição do erro quadrático médio (RMSE) obtido nas $10$ repetições em comparação ao ensaio nominal sem integrador.

\\begin{figure}[H]
    \\centering
    \\caption{Distribuição estatística do RMSE na bancada real ($N = 10$ repetições) para as três trajetórias de validação.}
    \\label{fig:boxplot_reteste}
    \\includegraphics[width=0.85\\textwidth]{img/resultados/boxplot_reteste_estatistico.png}
    \\sourceFig{o autor, 2026.}
\\end{figure}

A análise comparativa do controlador híbrido frente ao controlador nominal revelou os seguintes resultados:

\\begin{enumerate}
    \\item \\textbf{Atenuação do Erro Estacionário (\\textit{Bias}):} Na Validação~2 (degraus aleatórios), o erro médio de regime permanente reduziu-se de $+2{,}67^\\circ$ no controlador nominal para $+0{,}05^\\circ \\pm 0{,}01^\\circ$ no conjunto híbrido ($N = 10$), correspondendo a uma redução de $98{,}1\\%$. Na Validação~1, o viés médio diminuiu de $+2{,}29^\\circ$ para $+0{,}01^\\circ \\pm 0{,}01^\\circ$ ($99{,}5\\%$ de redução). Nos patamares de degrau, o braço do aeropêndulo convergiu para a linha de referência, indicando que a ação integradora compensou os desvios estáticos de sustentação e assimetrias mecânicas da planta.
    
    \\item \\textbf{Atenuação do Efeito de Atrito Estático na Senoide:} Na Validação~3 (Figura~\\ref{fig:cmp_hibrido_val3}), os patamares de travamento decorrentes de atrito estático de rolamento nos pontos de reversão de velocidade foram atenuados pela ação integradora residual. Como resultado, o RMSE médio na trajetória senoidal reduziu-se de $3{,}55^\\circ$ no modo nominal para $1{,}80^\\circ \\pm 0{,}03^\\circ$ nas $10$ repetições híbridas (redução de $49{,}3\\%$, com intervalo de confiança de $95\\%$ entre $1{,}78^\\circ$ e $1{,}82^\\circ$).
    
    \\item \\textbf{Redução do Erro Global e Repetibilidade:} Na Validação~1, o RMSE angular médio foi de $2{,}26^\\circ \\pm 0{,}03^\\circ$ (redução de $27{,}8\\%$ frente aos $3{,}13^\\circ$ do ensaio nominal; IC $95\\%$: $[2{,}24^\\circ; 2{,}28^\\circ]$), com coeficiente de variação amostral de $1{,}14\\%$. Na Validação~2, o RMSE médio situou-se em $3{,}80^\\circ \\pm 0{,}17^\\circ$ (redução de $17{,}6\\%$ frente a $4{,}61^\\circ$; IC $95\\%$: $[3{,}67^\\circ; 3{,}92^\\circ]$), com coeficiente de variação de $4{,}58\\%$. A maior dispersão observada na Validação~2 decorre da dinâmica transitória no degrau descendente em $t \\approx 41$\\,s, instante em que o sinal de comando atinge a saturação inferior e o sistema opera momentaneamente sob dinâmica livre de desaceleração.
    
    \\item \\textbf{Comportamento da Ação de Controle:} A taxa média de variação da ação de controle por ciclo de amostragem ($|\\Delta u|$ a cada $10$\\,ms) registrou valores de $0{,}1036\\%$ (Validação~1), $0{,}1276\\%$ (Validação~2) e $0{,}0298\\%$ (Validação~3), níveis equivalentes aos observados no ensaio nominal ($0{,}1027\\%$, $0{,}1426\\%$ e $0{,}0349\\%$). Esses valores indicam que a inclusão do integrador residual com $K_{i,\\text{res}} = 0{,}20$ operou sem introduzir oscilações de alta frequência no sinal de comando.
    
\\end{enumerate}"""

# Replacement 2: Table tab:desempenho_rmse
old_table = """    \\begin{tabular}{lccccccc}
        \\toprule
        \\textbf{Cenário de Teste} & \\textbf{Python ($^\\circ$)} & \\textbf{SIL Nom. ($^\\circ$)} & \\textbf{SIL Híb. ($^\\circ$)} & \\textbf{Bancada Nom. ($^\\circ$)} & \\textbf{Bancada Híb. ($^\\circ$)} & \\textbf{Redução RMSE} & \\textbf{Redução Viés (\\textit{Bias})} \\\\ \\midrule
        Multisseno + Degraus & 1,49 & 2,48 & 1,53 & 3,13 & \\textbf{2,20} & \\textbf{-29,8\\%} & $+2{,}29^\\circ \\to \\mathbf{+0{,}03^\\circ}$ (-98,8\\%) \\\\
        Degraus Aleatórios   & 3,63 & 3,78 & 3,38 & 4,61 & \\textbf{3,60} & \\textbf{-21,8\\%} & $+2{,}67^\\circ \\to \\mathbf{+0{,}07^\\circ}$ (-97,5\\%) \\\\
        Senoidal Suave       & 1,34 & 1,49 & 0,62 & 3,55 & \\textbf{1,75} & \\textbf{-50,8\\%} & $+2{,}13^\\circ \\to \\mathbf{+0{,}08^\\circ}$ (-96,2\\%) \\\\ \\bottomrule
    \\end{tabular}"""

new_table = """    \\begin{tabular}{lccccccc}
        \\toprule
        \\textbf{Cenário de Teste} & \\textbf{Python ($^\\circ$)} & \\textbf{SIL Nom. ($^\\circ$)} & \\textbf{SIL Híb. ($^\\circ$)} & \\textbf{Bancada Nom. ($^\\circ$)} & \\textbf{Bancada Híb. ($N=10$) ($^\\circ$)} & \\textbf{Redução RMSE} & \\textbf{Redução Viés (\\textit{Bias})} \\\\ \\midrule
        Multisseno + Degraus & 1,49 & 2,48 & 1,53 & 3,13 & $\\mathbf{2{,}26 \\pm 0{,}03}$ & \\textbf{-27,8\\%} & $+2{,}29^\\circ \\to \\mathbf{+0{,}01^\\circ \\pm 0{,}01^\\circ}$ (-99,5\\%) \\\\
        Degraus Aleatórios   & 3,63 & 3,78 & 3,38 & 4,61 & $\\mathbf{3{,}80 \\pm 0{,}17}$ & \\textbf{-17,6\\%} & $+2{,}67^\\circ \\to \\mathbf{+0{,}05^\\circ \\pm 0{,}01^\\circ}$ (-98,1\\%) \\\\
        Senoidal Suave       & 1,34 & 1,49 & 0,62 & 3,55 & $\\mathbf{1{,}80 \\pm 0{,}03}$ & \\textbf{-49,3\\%} & $+2{,}13^\\circ \\to \\mathbf{+0{,}08^\\circ \\pm 0{,}00^\\circ}$ (-96,4\\%) \\\\ \\bottomrule
    \\end{tabular}"""

# Replacement 3: Subsection 4.7 text
old_sec4_7_item3 = """    \\item \\textbf{Mitigação do Gap pelo Controlador Híbrido:} A inclusão em paralelo do integrador residual de baixa autoridade ($K_{i,\\text{res}} = 0{,}20$, saturação anti-windup em $\\pm 10\\%$) reduziu o erro persistente de sustentação: o viés médio diminuiu entre $96{,}2\\%$ e $98{,}8\\%$, situando-se entre ${+0{,}03^\\circ}$ e ${+0{,}08^\\circ}$. Simultaneamente, a atenuação do atrito estático (\\textit{stick-slip}) reduziu o erro na trajetória senoidal de $3{,}55^\\circ$ para $1{,}75^\\circ$ (redução de $50{,}8\\%$). Cumpre observar, contudo, que na Validação~3 o RMSE na bancada ($1{,}75^\\circ$) permanece superior ao obtido em SIL ($0{,}62^\\circ$), indicando que dinâmicas de alta ordem não modeladas e atritos residuais mantêm um gap dinâmico remanescente, embora o viés estático médio tenha sido praticamente eliminado."""

new_sec4_7_item3 = """    \\item \\textbf{Mitigação do Gap pelo Controlador Híbrido:} A inclusão em paralelo do integrador residual de baixa autoridade ($K_{i,\\text{res}} = 0{,}20$, saturação anti-windup em $\\pm 10\\%$) reduziu o erro persistente de sustentação: no conjunto de $N=10$ ensaios, o viés médio diminuiu entre $96{,}4\\%$ e $99{,}5\\%$, situando-se entre $+0{,}01^\\circ$ e $+0{,}08^\\circ$. Simultaneamente, a atenuação do atrito estático (\\textit{stick-slip}) reduziu o RMSE médio na trajetória senoidal de $3{,}55^\\circ$ para $1{,}80^\\circ \\pm 0{,}03^\\circ$ (redução de $49{,}3\\%$). Cumpre observar que na Validação~3 o RMSE na bancada ($1{,}80^\\circ$) permanece superior ao obtido em SIL ($0{,}62^\\circ$), indicando que dinâmicas não modeladas e atritos residuais mantêm um desvio dinâmico remanescente, embora o viés estático tenha sido atenuado."""

old_balanco = """alcançando valores de RMSE na bancada entre $1{,}75^\\circ$ e $3{,}60^\\circ$."""
new_balanco = """alcançando valores médios de RMSE na bancada entre $1{,}80^\\circ$ e $3{,}80^\\circ$ ($N = 10$)."""

# Normalize newlines for matching
def do_replace(text, old, new, desc):
    old_crlf = old.replace("\n", "\r\n")
    new_crlf = new.replace("\n", "\r\n")
    if old_crlf in text:
        text = text.replace(old_crlf, new_crlf)
        print(f"Substituição com sucesso (CRLF): {desc}")
    elif old in text:
        text = text.replace(old, new)
        print(f"Substituição com sucesso (LF): {desc}")
    else:
        print(f"ERRO: alvo não encontrado para: {desc}")
    return text

content = do_replace(content, old_sec4_4, new_sec4_4, "Seção 4.4 e Boxplot")
content = do_replace(content, old_table, new_table, "Tabela tab:desempenho_rmse")
content = do_replace(content, old_sec4_7_item3, new_sec4_7_item3, "Seção 4.7 item 3")
content = do_replace(content, old_balanco, new_balanco, "Balanço final")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Arquivo tcc/capitulos/cap_04.tex atualizado.")

