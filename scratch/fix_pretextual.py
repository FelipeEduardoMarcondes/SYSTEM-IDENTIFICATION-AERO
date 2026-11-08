p_path = r"c:\Users\vicio\Documents\AEROPENDULO\tcc\pretextual.tex"
with open(p_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Line 77 (index 76)
old_pt_end = "Ensaios experimentais na bancada demonstraram reduções de até 81\\% no RMSE dinâmico e atenuação superior a 96\\% no erro estacionário frente a um controlador PID clássico."
new_pt_end = "Ensaios experimentais na bancada demonstraram que o controlador preditivo neural reduziu o RMSE em até 81,7\\% (faixa de 69,2\\% a 81,7\\%) frente ao PID linear clássico sob manobras dinâmicas rápidas. Por sua vez, a adição do compensador integral residual na arquitetura híbrida reduziu o viés de regime permanente entre 96,4\\% e 99,5\\% (erro estático residual $\\le 0{,}08^\\circ$) e o RMSE em até 49,3\\% frente ao controlador neural puramente nominal, com repetibilidade estatística verificada em $N = 10$ ensaios independentes."

# Line 91 (index 90)
old_en_end = "Bench experimental trials demonstrated dynamic RMSE reductions of up to 81\\% and over 96\\% attenuation of steady-state error compared to a classical PID controller."
new_en_end = "Bench experimental trials demonstrated that the neural predictive controller reduced tracking RMSE by up to 81.7\\% (ranging from 69.2\\% to 81.7\\%) compared to a classical linear PID under fast dynamic maneuvers. Furthermore, the addition of the residual integral compensator in the hybrid architecture reduced steady-state bias by 96.4\\% to 99.5\\% (residual static error $\\le 0.08^\\circ$) and RMSE by up to 49.3\\% compared to the purely nominal neural controller, with statistical repeatability verified over $N = 10$ independent runs."

assert old_pt_end in lines[76], f"Not in line 77: {lines[76]}"
lines[76] = lines[76].replace(old_pt_end, new_pt_end)

assert old_en_end in lines[90], f"Not in line 91: {lines[90]}"
lines[90] = lines[90].replace(old_en_end, new_en_end)

with open(p_path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("pretextual.tex atualizado com sucesso!")

