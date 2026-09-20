clear
close all
clc

Ts = 1/1e2;
zeropad = zeros(1,500);
valorDC = 60; % valor medio

%% Curva semi-estatica
Tf1 = 75;
amp1 = 120;

[u1, t1] = gerarCurvaSemiEstatica(Ts, zeropad, Tf1, amp1);

%% Sequencia de degraus
Tf2 = 120;
Tduracao = 4; % quantos seg de duracao cada degrau
% Tduracao = 4; % quantos seg de duracao cada degrau
% Tduracao = 6; % quantos seg de duracao cada degrau

% amp2 = 15;
amp2 = 50;

[u2, t2] = gerarSeqDegraus(Ts, zeropad, valorDC, Tf2, Tduracao, amp2);

%% Swept sine
Fmax3 = 0.75;
A3 = 55; % amplitude
T03 = 60; % periodo [s]

[u3, t3] = gerarSweptSine(Ts, zeropad, valorDC, Fmax3, A3, T03);

%% Multi-seno (n realizacoes)
Fmax4 = 0.15;
% Amp4 = 15;
Amp4 = 60;
% Amp4 = 45;
Tf4 = 140; % mesma duracao da seq de degraus
ninp = 1;

[u4, t4] = gerarMultiSine(Ts, zeropad, valorDC, Fmax4, Amp4, Tf4, ninp);

%% Sequencia Degraus Variado (APRBS)
Tf5 = 140;           % Duracao total do sinal dinamico [s]
TduracaoMin5 = 2.0;  % Duracao minima de cada degrau (degraus curtos/rapidos) [s]
TduracaoMax5 = 4.0;  % Duracao maxima de cada degrau (degraus longos/lentos) [s]
amp5 = 55;           % Variacao de amplitude (vai saltar entre valorDC - 50 e valorDC + 50)
[u5, t5] = gerarSeqDegrausAPRBS(Ts, zeropad, valorDC, Tf5, TduracaoMin5, TduracaoMax5, amp5);


