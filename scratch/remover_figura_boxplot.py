import os

tex_path = r"c:\Users\vicio\Documents\AEROPENDULO\tcc\capitulos\cap_04.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()

old_block = """Para avaliar a repetibilidade e a consistência estatística do controlador híbrido na bancada física, analisou-se um conjunto de $N = 10$ ensaios experimentais independentes para cada trajetória de validação (composto pelo ensaio inicial e 9 repetições adicionais), avaliados na janela ativa de atuação da rede neural ($5{,}0$\\,s a $70{,}0$\\,s). A Figura~\\ref{fig:boxplot_reteste} apresenta a distribuição do erro quadrático médio (RMSE) obtido nas $10$ repetições em comparação ao ensaio nominal sem integrador.

\\begin{figure}[H]
    \\centering
    \\caption{Distribuição estatística do RMSE na bancada real ($N = 10$ repetições) para as três trajetórias de validação.}
    \\label{fig:boxplot_reteste}
    \\includegraphics[width=0.85\\textwidth]{img/resultados/boxplot_reteste_estatistico.png}
    \\sourceFig{o autor, 2026.}
\\end{figure}

A análise comparativa do controlador híbrido frente ao controlador nominal revelou os seguintes resultados:"""

new_block = """Para avaliar a repetibilidade e a consistência estatística do controlador híbrido na bancada física, analisou-se um conjunto de $N = 10$ ensaios experimentais independentes para cada trajetória de validação (composto pelo ensaio inicial e 9 repetições adicionais), avaliados na janela ativa de atuação da rede neural ($5{,}0$\\,s a $70{,}0$\\,s).

A análise comparativa do controlador híbrido frente ao controlador nominal revelou os seguintes resultados:"""

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Substituição com sucesso (LF)")
elif old_block.replace("\n", "\r\n") in content:
    content = content.replace(old_block.replace("\n", "\r\n"), new_block.replace("\n", "\r\n"))
    print("Substituição com sucesso (CRLF)")
else:
    print("ERRO: bloco não encontrado")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(content)

print("cap_04.tex atualizado.")

