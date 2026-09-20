% =========================================================================
% Script de Identificação de Sistemas e Sintonia de PID (Gain Scheduling)
% para o Aeropêndulo usando System Identification Toolbox.
% =========================================================================

clc; clear; close all;

% Configurações
pasta_experimentos = 'experimentos\MALHA-ABERTA\';
arquivos = dir(fullfile(pasta_experimentos, '*.csv'));

% Arrays para armazenar resultados do Gain Scheduling
pwm_list = [];
kp_list = [];
ki_list = [];
kd_list = [];
tf_list = [];

fprintf('Iniciando identificação em %d arquivos...\n\n', length(arquivos));

for i = 1:length(arquivos)
    nome_arq = arquivos(i).name;
    caminho = fullfile(pasta_experimentos, nome_arq);
    
    % Extrai o valor do PWM do nome do arquivo (ex: malha_aberta_-65_0917.csv)
    % Usando expressão regular para capturar números positivos ou negativos
    tokens = regexp(nome_arq, 'malha_aberta_(-?\d+)_', 'tokens');
    if isempty(tokens)
        continue; % Pula se o arquivo não tiver o padrão esperado
    end
    pwm_val = str2double(tokens{1}{1});
    
    % Lê o CSV
    opts = detectImportOptions(caminho);
    dados = readtable(caminho, opts);
    
    % As colunas são: tempo_ms, angulo_deg, u_pct, referencia
    t = dados.tempo_ms / 1000.0; % Converte para segundos
    y = dados.angulo_deg;
    u = dados.u_pct;
    
    % O passo foi aplicado aos 5.0 segundos. 
    % Vamos cortar os primeiros 4.5 segundos para garantir que pegamos
    % apenas a dinâmica limpa a partir do degrau (sem a espera inteira).
    idx_start = find(t >= 4.9, 1, 'first');
    t_cut = t(idx_start:end) - t(idx_start); % Zera o tempo inicial
    y_cut = y(idx_start:end);
    u_cut = u(idx_start:end);
    
    % Remove o offset (condições iniciais) para focar na variação (delta)
    y_delta = y_cut - y_cut(1);
    u_delta = u_cut - u_cut(1);
    
    Ts = 0.01; % Tempo de amostragem de 10ms (100Hz)
    
    % Cria o objeto de dados iddata
    % Como a entrada é um degrau, o MATLAB se beneficia se a entrada tiver as
    % variações corretas. Se u_delta for 0 em todo o array (porque o pulo
    % ocorreu exatamente no t=0 cortado), vamos forçar um degrau perfeito.
    % Na prática, o vetor u_cut já tem a transição.
    data = iddata(y_delta, u_delta, Ts, 'Name', sprintf('PWM %d', pwm_val));
    
    % Estimação de Função de Transferência Contínua (tfest)
    % Para um aeropêndulo, um modelo de 2 polos e 0 zeros ou 1 polo 0 zeros costuma ir bem.
    np = 2; % Número de polos
    nz = 0; % Número de zeros
    try
        sys_tf = tfest(data, np, nz);
    catch
        fprintf('Falha ao identificar o modelo para PWM = %d\n', pwm_val);
        continue;
    end
    
    % Sintoniza um PID robusto para este modelo específico usando pidtune
    % O pidtune procura o balanço ideal de performance e robustez
    [C_pi, info] = pidtune(sys_tf, 'PID'); 
    
    % Salva na lista
    pwm_list(end+1) = pwm_val;
    kp_list(end+1) = C_pi.Kp;
    ki_list(end+1) = C_pi.Ki;
    kd_list(end+1) = C_pi.Kd;
    tf_list{end+1} = sys_tf;
    
    fprintf('PWM %3d%%: Kp=%.3f, Ki=%.3f, Kd=%.3f (Polos: [%.2f, %.2f])\n', ...
        pwm_val, C_pi.Kp, C_pi.Ki, C_pi.Kd, pole(sys_tf));
end

%% Apresentação do Gain Scheduling
% Ordena os resultados com base no valor do PWM
[pwm_list_sorted, sort_idx] = sort(pwm_list);
kp_sorted = kp_list(sort_idx);
ki_sorted = ki_list(sort_idx);
kd_sorted = kd_list(sort_idx);

fprintf('\n=== TABELA DE GAIN SCHEDULING (PWM) ===\n');
fprintf(' PWM (%%) |   Kp    |   Ki    |   Kd    \n');
fprintf('-----------------------------------------\n');
for i = 1:length(pwm_list_sorted)
    fprintf(' %6d | %7.3f | %7.3f | %7.3f \n', ...
        pwm_list_sorted(i), kp_sorted(i), ki_sorted(i), kd_sorted(i));
end

%% Plot das Superfícies de Ganho (Kp, Ki, Kd vs PWM)
figure('Name', 'Gain Scheduling', 'NumberTitle', 'off');
subplot(3,1,1);
plot(pwm_list_sorted, kp_sorted, '-o', 'LineWidth', 2);
title('Ganho Proporcional (Kp) vs PWM');
grid on;

subplot(3,1,2);
plot(pwm_list_sorted, ki_sorted, '-o', 'LineWidth', 2);
title('Ganho Integral (Ki) vs PWM');
grid on;

subplot(3,1,3);
plot(pwm_list_sorted, kd_sorted, '-o', 'LineWidth', 2);
title('Ganho Derivativo (Kd) vs PWM');
xlabel('Ponto de Operação (PWM %)');
grid on;

fprintf('\nScript finalizado. O gráfico de Gain Scheduling foi gerado!\n');
