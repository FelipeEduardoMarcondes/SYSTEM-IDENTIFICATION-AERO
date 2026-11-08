import os

# Check pretextual.tex
p_path = r"c:\Users\vicio\Documents\AEROPENDULO\tcc\pretextual.tex"
with open(p_path, "r", encoding="utf-8") as f:
    p_content = f.read()

# Portuguese Resumo update
old_pt = "Ensaios experimentais na bancada demonstraram reduções de até 81% no RMSE dinâmico e atenuação superior a 96% no erro estacionário frente a um controlador PID clássico."
new_pt = "Ensaios experimentais na bancada demonstraram que o controlador preditivo neural reduziu o RMSE em até 81,7% (faixa de 69,2% a 81,7%) frente ao PID linear clássico sob manobras dinâmicas rápidas. Por sua vez, a adição do compensador integral residual na arquitetura híbrida reduziu o viés de regime permanente entre 96,4% e 99,5% (erro estático residual $\\le 0{,}08^\\circ$) e o RMSE em até 49,3% frente ao controlador neural puramente nominal, com repetibilidade estatística verificada em $N = 10$ ensaios independentes."

# English Abstract update
old_en = "Bench experimental trials demonstrated dynamic RMSE reductions of up to 81% and over 96% attenuation of steady-state error compared to a classical PID controller."
new_en = "Bench experimental trials demonstrated that the neural predictive controller reduced tracking RMSE by up to 81.7% (ranging from 69.2% to 81.7%) compared to a classical linear PID under fast dynamic maneuvers. Furthermore, the addition of the residual integral compensator in the hybrid architecture reduced steady-state bias by 96.4% to 99.5% (residual static error $\\le 0.08^\\circ$) and RMSE by up to 49.3% compared to the purely nominal neural controller, with statistical repeatability verified over $N = 10$ independent runs."

def update_file(content, old, new, name):
    if old in content:
        content = content.replace(old, new)
        print(f"Sucesso (LF): {name}")
    elif old.replace("\n", "\r\n") in content:
        content = content.replace(old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))
        print(f"Sucesso (CRLF): {name}")
    else:
        print(f"ERRO: não encontrado em {name}")
    return content

p_content = update_file(p_content, old_pt, new_pt, "Resumo PT")
p_content = update_file(p_content, old_en, new_en, "Abstract EN")

with open(p_path, "w", encoding="utf-8") as f:
    f.write(p_content)

# Now check cap_05.tex
c5_path = r"c:\Users\vicio\Documents\AEROPENDULO\tcc\capitulos\cap_05.tex"
with open(c5_path, "r", encoding="utf-8") as f:
    c5_content = f.read()

old_c5_item5 = """    \\item \\textbf{Comparação Experimental entre Controladores e Mitigação do Gap Sim-to-Real:} Os ensaios espelhados na bancada física frente aos sinais evidenciaram vantagens expressivas da compensação preditiva neural sobre o PID clássico, com redução na raiz do erro quadrático médio (RMSE) de rastreamento entre $69{,}2\\%$ e $81{,}7\\%$ em manobras dinâmicas de alta frequência e amplitude. Concomitantemente, a transição para a bancada revelou o viés estacionário e o travamento mecânico por atrito estático de rolamento (\\textit{stick-slip}) decorrentes da ausência de ação integradora na MLP pura; a introdução da arquitetura híbrida (MLP com integrador residual paralelo saturado) atenuou o viés em mais de $96\\%$ e reduziu o erro senoidal pela metade, mitigando expressivamente o componente estático do \\textit{Sim-to-Real gap}."""

new_c5_item5 = """    \\item \\textbf{Comparação Experimental entre Controladores e Mitigação do Gap Sim-to-Real:} Os ensaios na bancada física permitiram avaliar com precisão duas comparações complementares:
    \\begin{itemize}
        \\item \\textit{Controlador Neural Preditivo versus PID Clássico:} Em manobras dinâmicas rápidas (sinais multisseno e \\textit{chirp} em $45^\\circ$ e $60^\\circ$), a ação preditiva antecipatória da rede neural reduziu a raiz do erro quadrático médio (RMSE) entre $69{,}2\\%$ e $81{,}7\\%$ frente ao controlador PID linear sintonizado na bancada;
        \\item \\textit{Controlador Híbrido versus Controlador Neural Nominal:} Na comparação entre as topologias neurais na bancada ($N=10$ repetições independentes), a inclusão do integrador residual de baixa autoridade em paralelo com a MLP atenuou o erro de regime permanente (viés estático) entre $96{,}4\\%$ e $99{,}5\\%$ (reduzindo o viés de $+2{,}67^\\circ$ para $+0{,}05^\\circ \\pm 0{,}01^\\circ$ nos degraus e para $+0{,}01^\\circ \\pm 0{,}01^\\circ$ no multisseno) e reduziu o RMSE em $49{,}3\\%$ na senoide (de $3{,}55^\\circ$ para $1{,}80^\\circ \\pm 0{,}03^\\circ$), $27{,}8\\%$ no multisseno com degraus e $17{,}6\\%$ nos degraus aleatórios, mitigando o componente de erro estático associado ao \\textit{Sim-to-Real gap}.
    \\end{itemize}"""

c5_content = update_file(c5_content, old_c5_item5, new_c5_item5, "Cap 05 item 5")

with open(c5_path, "w", encoding="utf-8") as f:
    f.write(c5_content)

print("Arquivos atualizados.")

