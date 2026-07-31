clear
close all
clc

Ts = 1/1e2;
zeropad = zeros(1,500);
valorDC = 45; % valor medio

%% Curva semi-estatica
Tf1 = 60;
amp1 = 90;

[u1, t1] = gerarCurvaSemiEstatica(Ts, zeropad, Tf1, amp1);

%% Sequencia de degraus
Tf2 = 60;
Tduracao = 4; % quantos seg de duracao cada degrau
% Tduracao = 4; % quantos seg de duracao cada degrau
% Tduracao = 6; % quantos seg de duracao cada degrau

% amp2 = 15;
amp2 = 30;

[u2, t2] = gerarSeqDegraus(Ts, zeropad, valorDC, Tf2, Tduracao, amp2);

%% Swept sine
Fmax3 = 0.5;
A3 = 30; % amplitude
T03 = 60; % periodo [s]

[u3, t3] = gerarSweptSine(Ts, zeropad, valorDC, Fmax3, A3, T03);

%% Multi-seno (n realizacoes)
Fmax4 = 0.50;
% Amp4 = 15;
Amp4 = 60;
% Amp4 = 45;
Tf4 = Tf2; % mesma duracao da seq de degraus
ninp = 1;

[u4, t4] = gerarMultiSine(Ts, zeropad, valorDC, Fmax4, Amp4, Tf4, ninp);
