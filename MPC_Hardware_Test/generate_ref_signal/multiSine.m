function [u, t] = multiSine(fsample, fmax, T, A, ninp)

Ts=1/fsample;                    % sample frequency and sample period
Ndata1=T/Ts;                     % length of the signal
t=(0:Ndata1-1)'*Ts;              % time axis
Nsines = Ndata1 - Ndata1*...
    (fsample - fmax)/fsample;    % number of sines
f = (0:Ndata1-1)*fsample/Ndata1; % multisine frequencies
LinesPlot = 1:floor(Ndata1/2);   % lines to be plotted

% multisine 2: with a random phase
U = [];  % freq. value
u = [];  % time value
Um = []; % power vs frequency
leg = {}; % for plotting
for i = 1:ninp
    U = [U zeros(Ndata1,1)];             % choose random phases
    U(2:Nsines+1,i) = exp(1i*2*pi*rand(Nsines,1));
    aux = 2*real(ifft(U(:,i)));
    u = [u aux/max(abs(aux))*A;];    
    leg = [leg {['Input ' num2str(i)]}];
end


