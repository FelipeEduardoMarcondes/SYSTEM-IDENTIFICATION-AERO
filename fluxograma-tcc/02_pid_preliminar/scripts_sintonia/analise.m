% 1. Planta (substitua pelos seus coeficientes reais)
s = tf('s');
conteudo = load('tf1.mat');
nomes_variaveis = fieldnames(conteudo);
G = conteudo.(nomes_variaveis{1});

% 2. Requisitos
Mp = 15;   % Sobressinal maximo (%)
ts = 5;    % Tempo de acomodacao (s)

% 3. Conversao Tempo -> Frequencia
zeta = -log(Mp/100) / sqrt(pi^2 + log(Mp/100)^2);
wn   = 4 / (zeta * ts);  % Criterio de 2%

% Aproximacoes mais precisas
PM = atan(2*zeta / sqrt(-2*zeta^2 + sqrt(1 + 4*zeta^4))) * (180/pi);
wc = wn * sqrt(sqrt(1 + 4*zeta^4) - 2*zeta^2);

fprintf('zeta = %.4f | wn = %.4f rad/s | PM = %.2f graus | wc = %.4f rad/s\n', ...
        zeta, wn, PM, wc);

% 4. Sintonia PID
disp('Controlador PID calculado:');
opts = pidtuneOptions('PhaseMargin', PM);
[C, info] = pidtune(G, 'pid', wc, opts);

disp('Controlador PID calculado:');
disp(C);

% 5. Validacao
T = feedback(C * G, 1);
figure;
step(T);
title('Resposta ao Degrau com PID');
grid on;

info_step = stepinfo(T);
disp(info_step);

% kd final 0.2