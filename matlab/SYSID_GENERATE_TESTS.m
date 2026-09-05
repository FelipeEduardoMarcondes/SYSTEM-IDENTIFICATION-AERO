%% 
% SYSID_GENERATE_TESTS.m
% Script para gerar sinais de excitacao para identificacao do aeropendulo
% Baseado na tabela final de parâmetros.
%
% Especificacoes:
%   - 3 variantes de cada tipo de sinal (exceto Chirp/MultiSeno que variam parâmetros)
%   - Nivel DC: 45 e 60 graus
%   - APRBS: degraus fixos e aleatorios (2 a 4 segundos)
%   - Chirp: Simetricos ao redor do DC, tempos definidos (Fmax = 0.50 Hz).
%   - Multi Seno: Frequencias maximas 0.30, 0.40, 0.50 Hz.

clear
close all
clc

Ts = 1/1e2;              % Periodo de amostragem (100 Hz)
zeropad = zeros(1, 300); % 3 segundos de repouso no inicio e no fim (adaptável)
Tf = 120;                % Duracao padrao dos sinais dinamicos [s]
Tf_chirp = 45;           % Duracao especifica dos chirps

disp('=============================================================')
disp(' GERANDO SINAIS DE IDENTIFICACAO - TABELA FINAL              ')
disp('=============================================================')

% =========================================================================
% 1. SEQ DEGRAUS (Tempo Fixo - assumindo ser o primeiro bloco de APRBS)
% =========================================================================
disp('--- SEQ DEGRAUS FIXOS (APRBS bloco 1) ---')

% DC 45
[sd_u1, sd_t1] = gerarSeqDegraus(Ts, zeropad, 45, Tf, 4, 45);
movefile('dados_seq_degraus.csv', 'seq-degraus-45-1.csv'); 
[sd_u2, sd_t2] = gerarSeqDegraus(Ts, zeropad, 45, Tf, 4, 45);
movefile('dados_seq_degraus.csv', 'seq-degraus-45-2.csv');
[sd_u3, sd_t3] = gerarSeqDegraus(Ts, zeropad, 45, Tf, 4, 45);
movefile('dados_seq_degraus.csv', 'seq-degraus-45-3.csv');

% DC 60
[sd_u4, sd_t4] = gerarSeqDegraus(Ts, zeropad, 60, Tf, 4, 60);
movefile('dados_seq_degraus.csv', 'seq-degraus-60-1.csv');
[sd_u5, sd_t5] = gerarSeqDegraus(Ts, zeropad, 60, Tf, 4, 60);
movefile('dados_seq_degraus.csv', 'seq-degraus-60-2.csv');
[sd_u6, sd_t6] = gerarSeqDegraus(Ts, zeropad, 60, Tf, 4, 60);
movefile('dados_seq_degraus.csv', 'seq-degraus-60-3.csv');
close all;

% =========================================================================
% 2. APRBS (Tempo Aleatório: 2 a 4 s)
% =========================================================================
disp('--- APRBS ALEATORIO ---')

% DC 45
[ap_u1, ap_t1] = gerarSeqDegrausAPRBS(Ts, zeropad, 45, Tf, 2.0, 4.0, 45);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-45-1.csv');
[ap_u2, ap_t2] = gerarSeqDegrausAPRBS(Ts, zeropad, 45, Tf, 2.0, 4.0, 45);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-45-2.csv');
[ap_u3, ap_t3] = gerarSeqDegrausAPRBS(Ts, zeropad, 45, Tf, 2.0, 4.0, 45);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-45-3.csv');

% DC 60
[ap_u4, ap_t4] = gerarSeqDegrausAPRBS(Ts, zeropad, 60, Tf, 2.0, 4.0, 60);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-60-1.csv');
[ap_u5, ap_t5] = gerarSeqDegrausAPRBS(Ts, zeropad, 60, Tf, 2.0, 4.0, 60);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-60-2.csv');
[ap_u6, ap_t6] = gerarSeqDegrausAPRBS(Ts, zeropad, 60, Tf, 2.0, 4.0, 60);
movefile('dados_seq_degraus_aprbs.csv', 'aprbs-60-3.csv');
close all;

% =========================================================================
% 3. CHIRP (Symmetric)
% =========================================================================
disp('--- CHIRP ---')

% DC 45
[ss_u1, ss_t1] = gerarSweptSine(Ts, zeropad, 45, 0.50, 25, Tf_chirp); % 20-70
movefile('dados_swept_sine.csv', 'chirp-45-amp25.csv');
[ss_u2, ss_t2] = gerarSweptSine(Ts, zeropad, 45, 0.50, 35, Tf_chirp); % 10-80
movefile('dados_swept_sine.csv', 'chirp-45-amp35.csv');
[ss_u3, ss_t3] = gerarSweptSine(Ts, zeropad, 45, 0.50, 45, Tf_chirp); % 0-90
movefile('dados_swept_sine.csv', 'chirp-45-amp45.csv');

% DC 60
[ss_u4, ss_t4] = gerarSweptSine(Ts, zeropad, 60, 0.50, 40, Tf_chirp); % 20-100
movefile('dados_swept_sine.csv', 'chirp-60-amp40.csv');
[ss_u5, ss_t5] = gerarSweptSine(Ts, zeropad, 60, 0.50, 50, Tf_chirp); % 10-110
movefile('dados_swept_sine.csv', 'chirp-60-amp50.csv');
[ss_u6, ss_t6] = gerarSweptSine(Ts, zeropad, 60, 0.50, 60, Tf_chirp); % 0-120
movefile('dados_swept_sine.csv', 'chirp-60-amp60.csv');
close all;

% =========================================================================
% 4. MULTI-SENO
% =========================================================================
disp('--- MULTI-SENO ---')

% DC 45
rng(55);
[ms_u1, ms_t1] = gerarMultiSine(Ts, zeropad, 45, 0.30, 45, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-45-030Hz.csv');
rng(65);
[ms_u2, ms_t2] = gerarMultiSine(Ts, zeropad, 45, 0.40, 45, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-45-040Hz.csv');
rng(75);
[ms_u3, ms_t3] = gerarMultiSine(Ts, zeropad, 45, 0.50, 45, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-45-050Hz.csv');

% DC 60
rng(55);
[ms_u4, ms_t4] = gerarMultiSine(Ts, zeropad, 60, 0.30, 60, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-60-030Hz.csv');
rng(65);
[ms_u5, ms_t5] = gerarMultiSine(Ts, zeropad, 60, 0.40, 60, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-60-040Hz.csv');
rng(75);
[ms_u6, ms_t6] = gerarMultiSine(Ts, zeropad, 60, 0.50, 60, Tf, 1);
movefile('dados_multi_sine.csv', 'multi-seno-60-050Hz.csv');
close all;

% =========================================================================
% PLOTAR TUDO (4 figuras, cada uma com 6 subplots)
% =========================================================================
disp('--- PLOTANDO ---')

% --- 1. SEQ DEGRAUS FIXOS ---
figure('Name','Seq Degraus (Tempo Fixo)','Position',[50 50 1400 800])
subplot(3,2,1); plot(sd_t1,sd_u1,'b'); title('DC=45 | Amp=45 | #1'); ylabel('graus'); grid on; ylim([min(sd_u1)-5 max(sd_u1)+5])
subplot(3,2,3); plot(sd_t2,sd_u2,'b'); title('DC=45 | Amp=45 | #2'); ylabel('graus'); grid on; ylim([min(sd_u2)-5 max(sd_u2)+5])
subplot(3,2,5); plot(sd_t3,sd_u3,'b'); title('DC=45 | Amp=45 | #3'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(sd_u3)-5 max(sd_u3)+5])
subplot(3,2,2); plot(sd_t4,sd_u4,'r'); title('DC=60 | Amp=60 | #1'); ylabel('graus'); grid on; ylim([min(sd_u4)-5 max(sd_u4)+5])
subplot(3,2,4); plot(sd_t5,sd_u5,'r'); title('DC=60 | Amp=60 | #2'); ylabel('graus'); grid on; ylim([min(sd_u5)-5 max(sd_u5)+5])
subplot(3,2,6); plot(sd_t6,sd_u6,'r'); title('DC=60 | Amp=60 | #3'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(sd_u6)-5 max(sd_u6)+5])

% --- 2. APRBS ---
figure('Name','APRBS (2-4 s)','Position',[100 50 1400 800])
subplot(3,2,1); plot(ap_t1,ap_u1,'b'); title('DC=45 | Amp=45 | #1'); ylabel('graus'); grid on; ylim([min(ap_u1)-5 max(ap_u1)+5])
subplot(3,2,3); plot(ap_t2,ap_u2,'b'); title('DC=45 | Amp=45 | #2'); ylabel('graus'); grid on; ylim([min(ap_u2)-5 max(ap_u2)+5])
subplot(3,2,5); plot(ap_t3,ap_u3,'b'); title('DC=45 | Amp=45 | #3'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(ap_u3)-5 max(ap_u3)+5])
subplot(3,2,2); plot(ap_t4,ap_u4,'r'); title('DC=60 | Amp=60 | #1'); ylabel('graus'); grid on; ylim([min(ap_u4)-5 max(ap_u4)+5])
subplot(3,2,4); plot(ap_t5,ap_u5,'r'); title('DC=60 | Amp=60 | #2'); ylabel('graus'); grid on; ylim([min(ap_u5)-5 max(ap_u5)+5])
subplot(3,2,6); plot(ap_t6,ap_u6,'r'); title('DC=60 | Amp=60 | #3'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(ap_u6)-5 max(ap_u6)+5])

% --- 3. CHIRP ---
figure('Name','Chirp (Swept-Sine)','Position',[150 50 1400 800])
subplot(3,2,1); plot(ss_t1,ss_u1,'b'); title('DC=45 | Range: 20-70'); ylabel('graus'); grid on; ylim([min(ss_u1)-5 max(ss_u1)+5])
subplot(3,2,3); plot(ss_t2,ss_u2,'b'); title('DC=45 | Range: 10-80'); ylabel('graus'); grid on; ylim([min(ss_u2)-5 max(ss_u2)+5])
subplot(3,2,5); plot(ss_t3,ss_u3,'b'); title('DC=45 | Range: 0-90'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(ss_u3)-5 max(ss_u3)+5])
subplot(3,2,2); plot(ss_t4,ss_u4,'r'); title('DC=60 | Range: 20-100'); ylabel('graus'); grid on; ylim([min(ss_u4)-5 max(ss_u4)+5])
subplot(3,2,4); plot(ss_t5,ss_u5,'r'); title('DC=60 | Range: 10-110'); ylabel('graus'); grid on; ylim([min(ss_u5)-5 max(ss_u5)+5])
subplot(3,2,6); plot(ss_t6,ss_u6,'r'); title('DC=60 | Range: 0-120'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(ss_u6)-5 max(ss_u6)+5])

% --- 4. MULTI-SENO ---
figure('Name','Multi-Seno','Position',[200 50 1400 800])
subplot(3,2,1); plot(ms_t1,ms_u1,'b'); title('DC=45 | Freq=0.30 Hz'); ylabel('graus'); grid on; ylim([min(ms_u1)-5 max(ms_u1)+5])
subplot(3,2,3); plot(ms_t2,ms_u2,'b'); title('DC=45 | Freq=0.40 Hz'); ylabel('graus'); grid on; ylim([min(ms_u2)-5 max(ms_u2)+5])
subplot(3,2,5); plot(ms_t3,ms_u3,'b'); title('DC=45 | Freq=0.50 Hz'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(ms_u3)-5 max(ms_u3)+5])
subplot(3,2,2); plot(ms_t4,ms_u4,'r'); title('DC=60 | Freq=0.30 Hz'); ylabel('graus'); grid on; ylim([min(ms_u4)-5 max(ms_u4)+5])
subplot(3,2,4); plot(ms_t5,ms_u5,'r'); title('DC=60 | Freq=0.40 Hz'); ylabel('graus'); grid on; ylim([min(ms_u5)-5 max(ms_u5)+5])
subplot(3,2,6); plot(ms_t6,ms_u6,'r'); title('DC=60 | Freq=0.50 Hz'); xlabel('Tempo [s]'); ylabel('graus'); grid on; ylim([min(ms_u6)-5 max(ms_u6)+5])

disp('=============================================================')
disp(' SUCESSO! 24 sinais gerados baseados na Tabela Final.        ')
disp(' 4 figuras abertas com os respectivos 6 graficos cada.       ')
disp('=============================================================')
