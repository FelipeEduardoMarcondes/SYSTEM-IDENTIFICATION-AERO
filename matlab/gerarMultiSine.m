function [u4, t4] = gerarMultiSine(Ts, zeropad, valorDC, Fmax, Amp, Tf, ninp)
% GERARMULTISINE Gera o sinal multi-seno (n realizacoes)
%
%   [u4, t4] = gerarMultiSine(Ts, zeropad, valorDC, Fmax, Amp, Tf, ninp)
%
%   Entradas:
%     Ts       - periodo de amostragem [s]
%     zeropad  - vetor de zeros usado no inicio/fim do sinal
%     valorDC  - valor medio somado ao sinal
%     Fmax     - frequencia maxima das senoides [Hz]
%     Amp      - amplitude do sinal
%     Tf       - duracao do sinal [s]
%     ninp     - numero de entradas (realizacoes) a gerar
%
%   Saidas:
%     u4, t4   - sinal gerado e vetor de tempo correspondente
%
%   Requer a funcao multiSine.m no mesmo caminho.
%
%   Ao final, salva u4 e t4 em 'dados_multi_sine.mat'

rng(545)
[u, t] = multiSine(1/Ts, Fmax, Tf, Amp, ninp);

u4 = [zeropad zeropad ([zeropad zeropad u' zeropad zeropad] + valorDC) zeropad zeropad];
N4 = length(u4);
t4 = ((1:N4)-1)*Ts;

figure
plot(t4, u4)
title('Multi-seno')
xlabel('Tempo [s]')
ylabel('Amplitude')

ut = timetable(u4', 'SampleRate', 1/Ts);
figure
pspectrum(ut)
xlim([0 10])

dados = [t4(:) u4(:)];
writematrix(dados, 'dados_multi_sine.csv');

end
