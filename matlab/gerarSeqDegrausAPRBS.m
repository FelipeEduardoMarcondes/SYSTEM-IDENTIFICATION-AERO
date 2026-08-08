function [u2, t2] = gerarSeqDegrausAPRBS(Ts, zeropad, valorDC, Tf, TduracaoMin, TduracaoMax, amp)
% GERARSEQDEGRAUSAPRBS Gera uma sequencia APRBS (Amplitude Pseudo-Random Binary Sequence)
%
%   [u2, t2] = gerarSeqDegrausAPRBS(Ts, zeropad, valorDC, Tf, TduracaoMin, TduracaoMax, amp)
%
%   Entradas:
%     Ts          - periodo de amostragem [s]
%     zeropad     - vetor de zeros usado no inicio/fim do sinal
%     valorDC     - valor medio (ponto de operacao) somado ao sinal
%     Tf          - duracao total da parte dinamica do sinal [s]
%     TduracaoMin - duracao minima de cada degrau [s]
%     TduracaoMax - duracao maxima de cada degrau [s]
%     amp         - variacao maxima de amplitude dos degraus (ao redor de valorDC)
%
%   Saidas:
%     u2, t2      - sinal gerado e vetor de tempo correspondente
%
%   Ao final, salva u2 e t2 em 'dados_seq_degraus_aprbs.csv'

rng('shuffle'); % Usa semente baseada no relogio para ser diferente a cada vez

u_aprbs = [];
tempo_acumulado = 0;

% Gera os blocos de degraus com tempos e amplitudes sorteados
while tempo_acumulado < Tf
    % Sorteia a duracao do degrau atual
    duracao_atual = TduracaoMin + (TduracaoMax - TduracaoMin) * rand();
    passos_atual = round(duracao_atual / Ts);
    
    % Sorteia a amplitude do degrau atual
    amp_atual = amp * (2 * rand() - 1);
    
    % Concatena no vetor
    u_aprbs = [u_aprbs, amp_atual * ones(1, passos_atual)];
    
    % Atualiza o tempo acumulado
    tempo_acumulado = tempo_acumulado + passos_atual * Ts;
end

% Corta o vetor para ter exatamente o tamanho de Tf para garantir consistencia
N_esperado = round(Tf / Ts) + 1;
u_aprbs = u_aprbs(1:N_esperado);

% Constroi o sinal final mantendo as margens de zeropad e valorDC originais
u2 = [zeropad zeropad ([zeropad zeropad u_aprbs zeropad zeropad] + valorDC) zeropad zeropad];
N2 = length(u2);
t2 = ((1:N2)-1)*Ts;

% Plot do sinal gerado
figure
plot(t2, u2, 'LineWidth', 1.5)
title('Sequencia APRBS (Amplitudes e Tempos Aleatorios)')
xlabel('Tempo [s]')
ylabel('Amplitude (Sinal de Controle)')
grid on

% Salva os dados em CSV
dados = [t2(:) u2(:)];
writematrix(dados, 'dados_seq_degraus_aprbs.csv');

end
