Emulador de planta: roda o MODELO FISICO do 1/4-drone dentro do Arduino
(theta_ddot = c1*sin(theta) + c2*u*|u| + c3*theta_dot, RK4, Ts=0.05s).

Mesmo modelo do FakeSerialArduino do hil_pid_gain_scheduling.py.
Sem IMU e sem motor: use para validar o laco serial antes da bancada real.
