% =========================================================================
% 2nd Order System Simulação - Discretized Plant + Ideal PID
% =========================================================================
clear; clc; close all;

% =========================================================================
% 1. Parameters
% =========================================================================
Ts   = 0.01;
Tsim = 20;
N    = Tsim / Ts;

Kp = 0.5793;
Ki = 0.6647;
Kd = 0.20;

% =========================================================================
% 2. Load & Discretize Plant (ZOH)
% =========================================================================
conteudo = load('tf1.mat');
nomes = fieldnames(conteudo);
sys_p_c = conteudo.(nomes{1});

sys_p_d = c2d(sys_p_c, Ts, 'zoh');
[num_p_d, den_p_d] = tfdata(sys_p_d, 'v');

b1 = num_p_d(2);
b2 = num_p_d(3);
a1 = den_p_d(2);
a2 = den_p_d(3);

fprintf('Plant:\n');
fprintf('y[k] = %.6f*y[k-1] + %.6f*y[k-2] + %.6f*u[k-1] + %.6f*u[k-2]\n\n', -a1, -a2, b1, b2);

% =========================================================================
% 3. PID Discretization (component-wise, no filter)
%
%   P  ->  u_p[k] = Kp * e[k]
%   I  ->  u_i[k] = u_i[k-1] + Ki*(Ts/2)*(e[k] + e[k-1])   [Tustin]
%   D  ->  u_d[k] = -(Kd/Ts) * (y[k] - y[k-1])              [derivador na medicao]
%
%   u[k] = u_p[k] + u_i[k] + u_d[k]
% =========================================================================
fprintf('PID gains: Kp=%.4f  Ki=%.4f  Kd=%.4f\n\n', Kp, Ki, Kd);

% =========================================================================
% 4. Simulação Variables
% =========================================================================
time = (0:N-1) * Ts;

y   = zeros(1, N);
u   = zeros(1, N);
e   = zeros(1, N);
u_i = zeros(1, N);
r   = zeros(1, N);

r(time > 0) = 60;   % degrau desde t=0, alinhado com o Arduino

% =========================================================================
% 5. Simulação Loop
% =========================================================================
for k = 3:N
    % Plant
    y(k) = -a1*y(k-1) - a2*y(k-2) + b1*u(k-1) + b2*u(k-2);

    % Error
    e(k) = r(k) - y(k);

    % PID components
    u_p    = Kp * e(k);
    u_i(k) = u_i(k-1) + Ki * (Ts/2) * (e(k) + e(k-1));   % Tustin
    u_d    = -(Kd/Ts) * (y(k) - y(k-1));                  % derivador na medicao

    u(k) = u_p + u_i(k) + u_d;
end

% =========================================================================
% 6. Plot base (simulacao)
% =========================================================================
figure('Name', 'Simulacao vs Real  |  PID', 'Position', [100, 100, 900, 700]);

subplot(3,1,1);
plot(time, r, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Referência'); hold on;
plot(time, y, 'b-',  'LineWidth', 1.5, 'DisplayName', 'Simulação');
title(sprintf('Step Response  |  Kp=%.2f  Ki=%.2f  Kd=%.2f', Kp, Ki, Kd));
xlabel('Time (s)'); ylabel('Output y[k]');
legend('Location', 'best');
grid on;

subplot(3,1,2);
plot(time, e, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Simulação'); hold on;
yline(0, 'k--');
title('Tracking Error');
xlabel('Time (s)'); ylabel('Error (deg)');
legend('Location', 'best');
grid on;

subplot(3,1,3);
stairs(time, u, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Simulação'); hold on;
title('Control Effort');
xlabel('Time (s)'); ylabel('Control u[k]');
legend('Location', 'best');
grid on;

sgtitle('Simulacao vs Real  |  Discretized Closed-Loop PID', ...
        'FontSize', 14, 'FontWeight', 'bold');
% =========================================================================
% 6. Carregamento de Dados Reais
% =========================================================================
arquivo_csv = 'pi-60-1.csv';
has_csv = isfile(arquivo_csv);

if has_csv
    dados = readtable(arquivo_csv);
    t_real   = dados.tempo_ms / 1000.0;
    t_real   = t_real - t_real(1); % Alinhamento de tempo
    ang_real = dados.angulo_deg;
    u_real   = dados.u_pct;
    r_real   = 40 * ones(size(t_real));
    e_real   = r_real - ang_real;
else
    fprintf('Arquivo CSV nao encontrado: %s\n', arquivo_csv);
end

% =========================================================================
% 7. Visualizacao Otimizada (Com Erro Modelo vs Real)
% =========================================================================
fig = figure('Name', 'Simulacao vs Real  |  PID', 'Position', [100, 100, 900, 850]);

% Painel 1: Saída
ax1 = subplot(4,1,1);
plot(time, r, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Referência'); hold on;
plot(time, y, 'b-',  'LineWidth', 1.5, 'DisplayName', 'Simulação');
if has_csv
    plot(t_real, ang_real, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Real'); 
end
title(sprintf('Step Response  |  Kp=%.2f  Ki=%.2f  Kd=%.2f', Kp, Ki, Kd));
ylabel('Output y[k]');
legend('Location', 'best'); grid on;

% Painel 2: Erro de Rastreamento (Referência - Saída)
ax2 = subplot(4,1,2);
yline(0, 'k--', 'HandleVisibility', 'off'); hold on;
plot(time, e, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Simulação');
if has_csv
    plot(t_real, e_real, 'r-', 'LineWidth', 1.0, 'DisplayName', 'Real'); 
end
title('Tracking Error (Ref - Output)');
ylabel('Error (deg)');
legend('Location', 'best'); grid on;

% Painel 3: Esforço de Controle
ax3 = subplot(4,1,3);
stairs(time, u, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Simulação'); hold on;
if has_csv
    stairs(t_real, u_real, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Real'); 
end
title('Control Effort');
ylabel('Control u[k]');
legend('Location', 'best'); grid on;

% Painel 4: Erro de Modelagem (Real - Simulação)
ax4 = subplot(4,1,4);
yline(0, 'k--', 'HandleVisibility', 'off'); hold on;
if has_csv
    % Interpola a saída simulada (y) para os instantes de tempo reais (t_real)
    y_sim_interp = interp1(time, y, t_real, 'linear', 'extrap');
    
    % Calcula o erro entre a planta física e o modelo
    erro_modelo_real = ang_real - y_sim_interp;
    
    plot(t_real, erro_modelo_real, 'm-', 'LineWidth', 1.2, 'DisplayName', 'Real - Simulação'); 
end
title('Modeling Error (Real vs Simulação)');
xlabel('Time (s)'); ylabel('Diff (deg)');
legend('Location', 'best'); grid on;

sgtitle('Simulacao vs Real  |  Discretized Closed-Loop PID', 'FontSize', 14, 'FontWeight', 'bold');

% Vincula o zoom no eixo X para os 4 subplots
linkaxes([ax1, ax2, ax3, ax4], 'x');
xlim(ax1, [0, Tsim]);
% =========================================================================
% 8. Metricas Comparativas
% =========================================================================
if has_csv
    idx_ss_sim = time > 0.7 * Tsim;
    idx_ss_real = find(t_real > 0.7 * t_real(end), 1);
    
    fprintf('=== Comparacao em regime permanente ===\n');
    fprintf('Simulacao  ->  media: %.2f deg   std: %.4f deg\n', mean(y(idx_ss_sim)), std(y(idx_ss_sim)));
    fprintf('Real       ->  media: %.2f deg   std: %.4f deg\n', mean(ang_real(idx_ss_real:end)), std(ang_real(idx_ss_real:end)));
    
    overshoot_real = (max(ang_real) - r_real(1)) / r_real(1) * 100;
    fprintf('Overshoot real: %.2f%%\n', overshoot_real);
    2%%%%
    band = 0.02 * r_real(1);
    settled = find(abs(ang_real - r_real(1)) > band, 1, 'last');
    if ~isempty(settled)
        fprintf('Tempo de acomodacao real (2%%): %.2f s\n', t_real(settled));
    end
end