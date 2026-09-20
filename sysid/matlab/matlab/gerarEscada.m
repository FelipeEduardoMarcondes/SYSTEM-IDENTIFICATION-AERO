function [u, t] = gerarEscada(Ts, zeropad, max_u, step_u, Tduracao)
% GERARESCADA Gera um sinal em forma de escada (staircase) para atrito estático.
%
%   Entradas:
%     Ts       - periodo de amostragem [s]
%     zeropad  - vetor de zeros usado no inicio/fim do sinal
%     max_u    - amplitude máxima do degrau (vai ate max_u e -max_u)
%     step_u   - incremento de amplitude por degrau
%     Tduracao - tempo que cada degrau é mantido [s]

    % Define as amplitudes dos degraus
    passos_subida = 0:step_u:max_u;
    passos_descida = (max_u-step_u):-step_u:0;
    
    % Concatena a sequencia: 0 -> max -> 0
    amplitudes = [passos_subida, passos_descida];
    
    N_duracao = round(Tduracao / Ts);
    
    u_din = [];
    for i = 1:length(amplitudes)
        u_din = [u_din, amplitudes(i) * ones(1, N_duracao)];
    end
    
    % Constroi o sinal final mantendo as margens de zeropad
    u = [zeropad zeropad u_din zeropad zeropad];
    N = length(u);
    t = ((1:N)-1)*Ts;

    % figure
    % plot(t, u, 'LineWidth', 1.5)
    % title('Sequencia de Micro-Degraus (Escada para Atrito)')
    % xlabel('Tempo [s]')
    % ylabel('Amplitude (Sinal de Controle)')
    % grid on

    dados = [t(:) u(:)];
    writematrix(dados, 'dados_escada.csv');
end
