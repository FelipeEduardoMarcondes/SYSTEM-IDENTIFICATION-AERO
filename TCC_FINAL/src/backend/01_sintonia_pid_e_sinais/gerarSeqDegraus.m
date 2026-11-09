function [u2, t2] = gerarSeqDegraus(Ts, zeropad, valorDC, Tf, Tduracao, amp)
% GERARSEQDEGRAUS Gera uma sequencia de degraus aleatorios
%
%   [u2, t2] = gerarSeqDegraus(Ts, zeropad, valorDC, Tf, Tduracao, amp)
%
%   Entradas:
%     Ts        - periodo de amostragem [s]
%     zeropad   - vetor de zeros usado no inicio/fim do sinal
%     valorDC   - valor medio somado ao sinal
%     Tf        - duracao total do sinal [s]
%     Tduracao  - duracao de cada degrau [s]
%     amp       - amplitude dos degraus
%
%   Saidas:
%     u2, t2    - sinal gerado e vetor de tempo correspondente
%
%   Ao final, salva u2 e t2 em 'dados_seq_degraus.mat'

rng('shuffle');

t = (0:Ts:Tf);
N = length(t);
Ndeg = round(Tduracao/Ts);
Nrand = ceil((N-1)/Ndeg);
randSteps = amp*(2*rand(Nrand,1)-1);
Umat = repmat(randSteps,1,Ndeg)';

u2 = [zeropad zeropad ([zeropad zeropad Umat(:)' zeropad zeropad] + valorDC) zeropad zeropad];

% Garante que nenhum valor fique negativo (clamp em 0)
u2 = max(u2, 0);

N2 = length(u2);
t2 = ((1:N2)-1)*Ts;

figure
plot(t2, u2)
title('Sequencia de degraus')
xlabel('Tempo [s]')
ylabel('Amplitude')

dados = [t2(:) u2(:)];
writematrix(dados, 'dados_seq_degraus.csv');

end
