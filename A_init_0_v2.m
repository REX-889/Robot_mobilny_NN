1%% c
clc;
clear all;

%% parametry


m     = 24;                 % masa platformy [kg]
b     = 0.15;              % połowa rozstawu kół [m]
r     = 0.0845;              % promień kół [m]
xi    = 0.1;               % tłumienie w łożyskach kół [N·m·s]
Ik    = 1.014*(10^(-5));                % bezwładność koła [kg·m^2] (brak konkretnej wartości – C symboliczne)

% Parametry silników
In    = 4.22e-3 * 1e-4;    % bezwładność wirnika silnika [kg·m^2]
xi_m  = 1.85 *1e-8;               % tłumienie w łożyskach silnika [N·m·s]
ng    = 1/2;              % przełożenie przekładni redukcyjnej [-]
R     = 3.78;              % rezystancja uzwojeń silnika [Ω]
km    = 0.855;           % stała maszynowa silnika [N·m/A]

B=2*b;

Ic = m*(B^2+B^2)/12; % moment bezwładności platformy [kg·m²]

kpr = 5;
kir = 25;
kpl = 5;
kil = 25;
kcl = kil + 50;
kcr = kir + 50;
T_Fr = 1/kpr;
T_Fl = 1/kpl;

%% warunki poczatkowe
% Zakresy losowania:
x_gsr_min = -5.0;   x_gsr_max = 5.0;    % [m]
y_gsr_min = -5.0;   y_gsr_max = 5.0;    % [m]
theta_gsr_min = -pi; theta_gsr_max = pi; % [rad]
x_0_min = -5.0;   x_0_max = 5.0;    % [m]
y_0_min = -5.0;   y_0_max = 5.0;    % [m]
theta_0_min = -pi; theta_0_max = pi; % [rad]
v_rand_0 = 0; v_rand_max = 0.1; % [m/s]
omega_rand_min = 0; omega_rand_max = 0.05; % 
% Losowanie pozycji i orientacji
x_gsr = x_gsr_min + (x_gsr_max - x_gsr_min) * rand();
y_gsr = y_gsr_min + (y_gsr_max - y_gsr_min) * rand();
theta_gsr = theta_gsr_min + (theta_gsr_max - theta_gsr_min) * rand();
x0 = x_0_min + (x_0_max - x_0_min) * rand();
y0 = y_0_min + (y_0_max - y_0_min) * rand();
theta0 = theta_0_min + (theta_0_max - theta_0_min) * rand();
v_rand = v_rand_0 + (v_rand_max - v_rand_0) * rand();
omega_gsr = omega_rand_min + (omega_rand_max - omega_rand_min) * rand();

%% warunki poczatkowe
% Zakresy losowania:
x_gsr_min = -5.0;   x_gsr_max = 5.0;    % [m]
y_gsr_min = -5.0;   y_gsr_max = 5.0;    % [m]
theta_gsr_min = -pi; theta_gsr_max = pi; % [rad]
x_0_min = -5.0;   x_0_max = 5.0;    % [m]
y_0_min = -5.0;   y_0_max = 5.0;    % [m]
theta_0_min = -pi; theta_0_max = pi; % [rad]
v_rand_0 = 0; v_rand_max = 0.1; % [m/s]
% Losowanie pozycji i orientacji
x_gsr = x_gsr_min + (x_gsr_max - x_gsr_min) * rand();
y_gsr = y_gsr_min + (y_gsr_max - y_gsr_min) * rand();
theta_gsr = theta_gsr_min + (theta_gsr_max - theta_gsr_min) * rand();
x0 = x_0_min + (x_0_max - x_0_min) * rand();
y0 = y_0_min + (y_0_max - y_0_min) * rand();
theta0 = theta_0_min + (theta_0_max - theta_0_min) * rand();
v_rand = v_rand_0 + (v_rand_max - v_rand_0) * rand();
