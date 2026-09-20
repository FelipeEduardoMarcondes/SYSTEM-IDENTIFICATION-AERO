clear
close all
clc

Ts = 1/1e2;                 % Amostragem 100 Hz
zeropad = zeros(1, 500);    % 5 segundos de repouso no início e no fim
valorDC = 60;               % valor médio em graus

%% 1. Sequencia de Degraus (APRBS)
Tf_aprbs = 50;              % 50 segundos de duração
TduracaoMin = 0.5;
TduracaoMax = 3.5;
amp_aprbs = 50;

[u_aprbs, t_aprbs] = gerarSeqDegrausAPRBS(Ts, zeropad, valorDC, Tf_aprbs, TduracaoMin, TduracaoMax, amp_aprbs);

%% 2. Swept Sine (Chirp)
Fmax_chirp = 0.5;
A_chirp = 30;               % amplitude
T0_chirp = 50;              % 50 segundos de duração

[u_chirp, t_chirp] = gerarSweptSine(Ts, zeropad, valorDC, Fmax_chirp, A_chirp, T0_chirp);

%% 3. Multi-seno
Fmax_multi = 0.15;
Amp_multi = 45;
Tf_multi = 45;              % 45 segundos de duração
ninp = 1;

[u_multi, t_multi] = gerarMultiSine(Ts, zeropad, valorDC, Fmax_multi, Amp_multi, Tf_multi, ninp);

%% 4. MONTAGEM DO SUPER EXPERIMENTO (Limitado a ~155s)
% Vamos extrair só a parte "ativa" dos sinais 2 e 3 (tirando os zeropads deles)
% para que não o pêndulo não pare no meio do experimento.

parte_ativa_chirp = u_chirp(length(zeropad)+1 : end-length(zeropad));
parte_ativa_multi = u_multi(length(zeropad)+1 : end-length(zeropad));

% u_aprbs já vem com zeropad no começo e no fim. Inserimos as outras partes no meio:
u_super = [u_aprbs(1 : end-length(zeropad)), parte_ativa_chirp, parte_ativa_multi, zeropad];

% Recalcula o vetor de tempo pro novo tamanho
N_super = length(u_super);
t_super = ((1:N_super)-1)*Ts;

figure;
plot(t_super, u_super, 'LineWidth', 1.2);
title(sprintf('Super Experimento - Tamanho: %d amostras (Max: 16500)', N_super));
xlabel('Tempo [s]');
ylabel('Sinal de Controle');
grid on;

% Salva o arquivo CSV que vai ser lido pelo Python e enviado para o STM32
dados = [t_super(:) u_super(:)];
writematrix(dados, 'controle/dados_super_experimento.csv');

disp(['Experimento gerado com sucesso! Total de tempo: ' num2str(t_super(end)) 's']);
