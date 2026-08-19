% SYSID_GENERATE_TESTS.m
% Script para gerar conjuntos de teste mais ricos e testes com ângulos negativos
% 
% Os arquivos gerados terao nomenclaturas especificas como multi-seno-1, 
% aprbs-1, etc., conforme solicitado.

clear
close all
clc

Ts = 1/1e2;
zeropad = zeros(1,500); % 5 segundos de repouso no inicio e no fim

disp('=======================================================')
disp(' GERANDO CONJUNTOS DE TESTE RICOS E ÂNGULOS NEGATIVOS ')
disp('=======================================================')

% -------------------------------------------------------------------------
% 1. CONJUNTOS DE TESTE RICOS (Sinais maiores, diferentes dinâmicas)
% -------------------------------------------------------------------------
valorDC_rico = 60; 
Tf_rico = 120; % Tempo de duracao (120s) para sinais mais ricos

disp('-> Gerando multi-seno-1 (Foco em baixas freq e grande amp)...')
[~, ~] = gerarMultiSine(Ts, zeropad, valorDC_rico, 0.2, 55, Tf_rico, 1);
movefile('dados_multi_sine.csv', 'multi-seno-1.csv');
% close all;

disp('-> Gerando multi-seno-2 (Foco em freq mais altas e moderada amp)...')
[~, ~] = gerarMultiSine(Ts, zeropad, valorDC_rico, 0.6, 40, Tf_rico, 1);
movefile('dados_multi_sine.csv', 'multi-seno-2.csv');
% close all;

disp('-> Gerando aprbs-1 (Variações rápidas, degraus curtos)...')
[~, ~] = gerarSeqDegrausAPRBS(Ts, zeropad, valorDC_rico, Tf_rico, 0.5, 2.5, 55);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-1.csv');
% close all;

disp('-> Gerando aprbs-2 (Variações lentas, degraus longos)...')
[~, ~] = gerarSeqDegrausAPRBS(Ts, zeropad, valorDC_rico, Tf_rico, 2.0, 7.0, 55);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-2.csv');
% close all;

disp('-> Gerando swept-sine-1 (Banda ampla de frequencias)...')
[~, ~] = gerarSweptSine(Ts, zeropad, valorDC_rico, 0.8, 45, Tf_rico);
movefile('dados_swept_sine.csv', 'swept-sine-1.csv');
% close all;


% -------------------------------------------------------------------------
% 2. CONJUNTOS DE TESTE ABORDANDO ÂNGULOS NEGATIVOS
% -------------------------------------------------------------------------
% Para gerar ângulos negativos (ou seja, o sinal de controle passando por zero e
% assumindo valores negativos), podemos alterar o valorDC para 0 (oscilando
% igualmente entre positivo e negativo) ou para um valor negativo.

valorDC_zero = 0;
valorDC_neg = -30;
Tf_neg = 120; % 120s

disp('-> Gerando multi-seno-neg-1 (Centrado em 0, amplitude 60)...')
[~, ~] = gerarMultiSine(Ts, zeropad, valorDC_zero, 0.25, 60, Tf_neg, 1);
movefile('dados_multi_sine.csv', 'multi-seno-neg-1.csv');
% close all;

disp('-> Gerando multi-seno-neg-2 (Centrado em -30, amplitude 50)...')
[~, ~] = gerarMultiSine(Ts, zeropad, valorDC_neg, 0.35, 50, Tf_neg, 1);
movefile('dados_multi_sine.csv', 'multi-seno-neg-2.csv');
% close all;

disp('-> Gerando aprbs-neg-1 (Centrado em 0, amplitude max 70)...')
[~, ~] = gerarSeqDegrausAPRBS(Ts, zeropad, valorDC_zero, Tf_neg, 1.0, 4.0, 70);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-neg-1.csv');
% close all;

disp('-> Gerando aprbs-neg-2 (Regime 100% negativo, centrado em -50)...')
[~, ~] = gerarSeqDegrausAPRBS(Ts, zeropad, -50, Tf_neg, 1.0, 5.0, 30);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-neg-2.csv');
% close all;

disp('-> Gerando swept-sine-neg-1 (Centrado em 0)...')
[~, ~] = gerarSweptSine(Ts, zeropad, valorDC_zero, 0.5, 60, Tf_neg);
movefile('dados_swept_sine.csv', 'swept-sine-neg-1.csv');
% close all;

disp('=======================================================')
disp(' SUCESSO! Todos os testes iniciais foram gerados. ')
disp('=======================================================')

% -------------------------------------------------------------------------
% 3. CONJUNTOS DE TESTE ESPECÍFICOS PARA ATRITO DE STRIBECK
% -------------------------------------------------------------------------
disp('=======================================================')
disp(' GERANDO CONJUNTOS DE TESTE PARA ATRITO (STRIBECK) ')
disp('=======================================================')

% Micro-degraus (Escada):
% Amplitude indo de 0 até 90 (e depois descendo)
% Incrementos de 5 em 5.
% O sistema estabiliza em 4s, então deixamos cada degrau por 5s.
disp('-> Gerando micro-degraus (escada-atrito-1)...')
[~, ~] = gerarEscada(Ts, zeropad, 90, 5, 5.0);
movefile('dados_escada.csv', 'escada-atrito-1.csv');

% Queda Livre (Coast-Down):
% Leva o pêndulo a um ângulo alto (u=90) por 10s para estabilizar
% Desliga o motor (u=0) e deixa cair por 20s
disp('-> Gerando queda livre (queda-livre-1)...')
[~, ~] = gerarQuedaLivre(Ts, zeropad, 90, 10.0, 20.0);
movefile('dados_queda_livre.csv', 'queda-livre-1.csv');
% close all;

disp('=======================================================')
disp(' SUCESSO TOTAL! Todos os testes foram gerados e salvos. ')
disp('=======================================================')
