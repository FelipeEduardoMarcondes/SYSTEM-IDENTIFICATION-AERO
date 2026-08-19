function [u, t] = gerarQuedaLivre(Ts, zeropad, amp_max, tempo_segurar, tempo_queda)
% GERARQUEDALIVRE Gera um ensaio de queda livre (coast-down)
%
%   Leva o sinal até uma amplitude alta (amp_max) para levantar o pêndulo,
%   aguarda estabilizar e desliga o motor subitamente (0) para observar o 
%   amortecimento puramente por atrito e gravidade.

    % Fase 1: Sobe para amp_max (degrau) e segura para estabilizar
    passos_segurar = round(tempo_segurar / Ts);
    u_cima = amp_max * ones(1, passos_segurar);
    
    % Fase 2: Desliga o motor (0) e deixa cair livremente
    passos_queda = round(tempo_queda / Ts);
    u_baixo = zeros(1, passos_queda);
    
    % Constroi a sequencia
    u_din = [u_cima, u_baixo];
    
    u = [zeropad u_din zeropad];
    N = length(u);
    t = ((1:N)-1)*Ts;
    
    dados = [t(:) u(:)];
    writematrix(dados, 'dados_queda_livre.csv');

end
