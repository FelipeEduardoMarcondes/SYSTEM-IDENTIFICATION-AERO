% SYSID_GENERATE_TESTS.m
% Script para gerar sinais de excitacao para identificacao do aeropendulo
%
% Especificacoes:
%   - 4 variantes de cada tipo de sinal
%   - Frequencia maxima: 0.75 Hz
%   - Nivel DC: 45 e 60 graus
%   - APRBS: degraus entre 2 e 5 segundos
%   - SEM angulos negativos (amplitude limitada)

clear
close all
clc

Ts = 1/1e2;              % Periodo de amostragem (100 Hz)
zeropad = zeros(1, 300); % 5 segundos de repouso no inicio e no fim
Tf = 120;                % Duracao dos sinais dinamicos [s]

disp('=============================================================')
disp(' GERANDO SINAIS DE IDENTIFICACAO - RODADA 4                  ')
disp(' Fmax=0.75Hz | DC=45/60 | Sem angulos negativos              ')
disp('=============================================================')

% =========================================================================
% 1. MULTI-SENO (4 variantes)
% =========================================================================
disp('--- MULTI-SENO ---')

rng(55);
[ms_u1, ms_t1] = gerarMultiSine(Ts, zeropad, 45, 0.30, 45, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-1.csv'); close all;

rng(350);
[ms_u2, ms_t2] = gerarMultiSine(Ts, zeropad, 60, 0.30, 60, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-2.csv'); close all;

rng(55);
[ms_u3, ms_t3] = gerarMultiSine(Ts, zeropad, 45, 0.60, 45, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-3.csv'); close all;

rng(350);
[ms_u4, ms_t4] = gerarMultiSine(Ts, zeropad, 60, 0.75, 60, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-4.csv'); close all;

% =========================================================================
% 2. APRBS (4 variantes) - degraus entre 2 e 5 segundos
% =========================================================================
disp('--- APRBS ---')

[ap_u1, ap_t1] = gerarSeqDegrausAPRBS(Ts, zeropad, 45, Tf, 2.0, 4.0, 45);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-1.csv'); close all;

[ap_u2, ap_t2] = gerarSeqDegrausAPRBS(Ts, zeropad, 60, Tf, 2.0, 4.0, 60);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-2.csv'); close all;

[ap_u3, ap_t3] = gerarSeqDegrausAPRBS(Ts, zeropad, 45, Tf, 2.0, 4.0, 45);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-3.csv'); close all;

[ap_u4, ap_t4] = gerarSeqDegrausAPRBS(Ts, zeropad, 60, Tf, 2.0, 4.0, 60);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-4.csv'); close all;

% =========================================================================
% 3. SWEPT-SINE / CHIRP (4 variantes) - Fmax = 0.75 Hz
% =========================================================================
disp('--- SWEPT-SINE ---')

[ss_u1, ss_t1] = gerarSweptSine(Ts, zeropad, 45, 0.75, 35, 120);
movefile('dados_swept_sine.csv', 'swept-sine-1.csv'); close all;

[ss_u2, ss_t2] = gerarSweptSine(Ts, zeropad, 60, 0.75, 50, 120);
movefile('dados_swept_sine.csv', 'swept-sine-2.csv'); close all;

[ss_u3, ss_t3] = gerarSweptSine(Ts, zeropad, 45, 0.75, 40, 80);
movefile('dados_swept_sine.csv', 'swept-sine-3.csv'); close all;

[ss_u4, ss_t4] = gerarSweptSine(Ts, zeropad, 60, 0.75, 55, 80);
movefile('dados_swept_sine.csv', 'swept-sine-4.csv'); close all;

% =========================================================================
% 4. SEQUENCIA DE DEGRAUS (4 variantes)
% =========================================================================
disp('--- SEQUENCIA DE DEGRAUS ---')

[sd_u1, sd_t1] = gerarSeqDegraus(Ts, zeropad, 45, Tf, 4, 45);
movefile('dados_seq_degraus.csv', 'seq-degraus-1.csv'); close all;

[sd_u2, sd_t2] = gerarSeqDegraus(Ts, zeropad, 60, Tf, 4, 60);
movefile('dados_seq_degraus.csv', 'seq-degraus-2.csv'); close all;

[sd_u3, sd_t3] = gerarSeqDegraus(Ts, zeropad, 45, Tf, 4, 45);
movefile('dados_seq_degraus.csv', 'seq-degraus-3.csv'); close all;

[sd_u4, sd_t4] = gerarSeqDegraus(Ts, zeropad, 60, Tf, 4, 60);
movefile('dados_seq_degraus.csv', 'seq-degraus-4.csv'); close all;

% =========================================================================
% PLOTAR TUDO (4 figuras, cada uma com 4 subplots)
% =========================================================================
disp('--- PLOTANDO ---')

figure('Name','Multi-Seno (4 variantes)','Position',[50 50 1400 700])
subplot(4,1,1); plot(ms_t1,ms_u1,'b'); title('multi-seno-1 | DC=45, Fmax=0.30Hz, Amp=40'); ylabel('graus'); grid on; ylim([min(ms_u1)-5 max(ms_u1)+5])
subplot(4,1,2); plot(ms_t2,ms_u2,'b'); title('multi-seno-2 | DC=60, Fmax=0.30Hz, Amp=55'); ylabel('graus'); grid on; ylim([min(ms_u2)-5 max(ms_u2)+5])
subplot(4,1,3); plot(ms_t3,ms_u3,'b'); title('multi-seno-3 | DC=45, Fmax=0.60Hz, Amp=35'); ylabel('graus'); grid on; ylim([min(ms_u3)-5 max(ms_u3)+5])
subplot(4,1,4); plot(ms_t4,ms_u4,'b'); title('multi-seno-4 | DC=60, Fmax=0.75Hz, Amp=50'); ylabel('graus'); grid on; ylim([min(ms_u4)-5 max(ms_u4)+5])
xlabel('Tempo [s]')

figure('Name','APRBS (4 variantes)','Position',[100 50 1400 700])
subplot(4,1,1); plot(ap_t1,ap_u1,'b'); title('aprbs-1 | DC=45, T=[2,4]s, Amp=40'); ylabel('graus'); grid on; ylim([min(ap_u1)-5 max(ap_u1)+5])
subplot(4,1,2); plot(ap_t2,ap_u2,'b'); title('aprbs-2 | DC=60, T=[2,4]s, Amp=55'); ylabel('graus'); grid on; ylim([min(ap_u2)-5 max(ap_u2)+5])
subplot(4,1,3); plot(ap_t3,ap_u3,'b'); title('aprbs-3 | DC=45, T=[3,5]s, Amp=40'); ylabel('graus'); grid on; ylim([min(ap_u3)-5 max(ap_u3)+5])
subplot(4,1,4); plot(ap_t4,ap_u4,'b'); title('aprbs-4 | DC=60, T=[3,5]s, Amp=55'); ylabel('graus'); grid on; ylim([min(ap_u4)-5 max(ap_u4)+5])
xlabel('Tempo [s]')

figure('Name','Swept-Sine (4 variantes)','Position',[150 50 1400 700])
subplot(4,1,1); plot(ss_t1,ss_u1,'b'); title('swept-sine-1 | DC=45, Amp=35, T0=120s'); ylabel('graus'); grid on; ylim([min(ss_u1)-5 max(ss_u1)+5])
subplot(4,1,2); plot(ss_t2,ss_u2,'b'); title('swept-sine-2 | DC=60, Amp=50, T0=120s'); ylabel('graus'); grid on; ylim([min(ss_u2)-5 max(ss_u2)+5])
subplot(4,1,3); plot(ss_t3,ss_u3,'b'); title('swept-sine-3 | DC=45, Amp=40, T0=80s'); ylabel('graus'); grid on; ylim([min(ss_u3)-5 max(ss_u3)+5])
subplot(4,1,4); plot(ss_t4,ss_u4,'b'); title('swept-sine-4 | DC=60, Amp=55, T0=80s'); ylabel('graus'); grid on; ylim([min(ss_u4)-5 max(ss_u4)+5])
xlabel('Tempo [s]')

figure('Name','Seq Degraus (4 variantes)','Position',[200 50 1400 700])
subplot(4,1,1); plot(sd_t1,sd_u1,'b'); title('seq-degraus-1 | DC=45, Tdeg=3s, Amp=40'); ylabel('graus'); grid on; ylim([min(sd_u1)-5 max(sd_u1)+5])
subplot(4,1,2); plot(sd_t2,sd_u2,'b'); title('seq-degraus-2 | DC=60, Tdeg=3s, Amp=55'); ylabel('graus'); grid on; ylim([min(sd_u2)-5 max(sd_u2)+5])
subplot(4,1,3); plot(sd_t3,sd_u3,'b'); title('seq-degraus-3 | DC=45, Tdeg=5s, Amp=35'); ylabel('graus'); grid on; ylim([min(sd_u3)-5 max(sd_u3)+5])
subplot(4,1,4); plot(sd_t4,sd_u4,'b'); title('seq-degraus-4 | DC=60, Tdeg=5s, Amp=50'); ylabel('graus'); grid on; ylim([min(sd_u4)-5 max(sd_u4)+5])
xlabel('Tempo [s]')

disp('=============================================================')
disp(' SUCESSO! 16 sinais gerados. 4 figuras abertas.              ')
disp('=============================================================')
