function [u3, t3] = gerarSweptSine(Ts, zeropad, valorDC, Fmax, A, T0)
% GERARSWEPTSINE Gera o sinal swept sine (chirp)
%
%   [u3, t3] = gerarSweptSine(Ts, zeropad, valorDC, Fmax, A, T0)
%
%   Entradas:
%     Ts       - periodo de amostragem [s]
%     zeropad  - vetor de zeros usado no inicio/fim do sinal
%     valorDC  - valor medio somado ao sinal
%     Fmax     - frequencia maxima do chirp [Hz]
%     A        - amplitude do sinal
%     T0       - periodo base [s]
%
%   Saidas:
%     u3, t3   - sinal gerado e vetor de tempo correspondente
%
%   Ao final, salva u3 e t3 em 'dados_swept_sine.mat'

f0 = 1/T0;         % frequencia base
k1 = 1;            % [k1*f0, k2*f0] menor e maior frequencia
k2 = Fmax/f0;      % k2 > k1, numeros naturais
disp(k1*f0)        % freq inicial
disp(k2*f0)        % freq final
a = pi*(k2-k1)*f0^2;
b = 2*pi*k1*f0;

t_chirp = 0:Ts:T0;
u_chirp = A*sin((a*t_chirp+b).*t_chirp);

u3 = [zeropad zeropad ([zeropad zeropad u_chirp zeropad zeropad] + valorDC) zeropad zeropad];
N3 = length(u3);
t3 = ((1:N3)-1)*Ts;

figure
plot(t3, u3)
title('Swept sine')
xlabel('Tempo [s]')
ylabel('Amplitude')

ut = timetable(u3', 'SampleRate', 1/Ts);
figure
pspectrum(ut)
xlim([0 10])




dados = [t3(:) u3(:)];
writematrix(dados, 'dados_swept_sine.csv');


end
