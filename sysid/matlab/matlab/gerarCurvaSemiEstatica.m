function [u1, t1] = gerarCurvaSemiEstatica(Ts, zeropad, Tf, amp)
% GERARCURVASEMIESTATICA Gera a curva semi-estatica (rampa subida/descida)
%
%   [u1, t1] = gerarCurvaSemiEstatica(Ts, zeropad, Tf, amp)
%
%   Entradas:
%     Ts       - periodo de amostragem [s]
%     zeropad  - vetor de zeros usado no inicio/fim do sinal
%     Tf       - duracao da rampa [s]
%     amp      - amplitude maxima da rampa
%
%   Saidas:
%     u1, t1   - sinal gerado e vetor de tempo correspondente
%
%   Ao final, salva u1 e t1 em 'dados_curva_semi_estatica.mat'

t_rampa = 0:Ts:Tf;
N1 = length(t_rampa);
u_rampa = linspace(0, amp, N1);

u1 = [zeropad u_rampa u_rampa(end:-1:1) zeropad];
N1 = length(u1);
t1 = ((1:N1)-1)*Ts;

figure
plot(t1, u1)
title('Curva semi-estatica')
xlabel('Tempo [s]')
ylabel('Amplitude')

dados = [t1(:) u1(:)];
writematrix(dados, 'dados_curva_semi_estatica.csv');

end
