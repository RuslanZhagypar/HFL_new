clc; clear; close all;

%% ======================== Simulation Parameters ========================= %%
N = 3;
n = 10;
m = 2;
I = 2;

L = 1;
G = 1;
eta = 1;
A2 = 1;
C = 112/5;

b2 = 0:0.1:1;  % Range for b2
b3 = 0:0.1:1;  % Range for b3

% Create a grid of coordinates
[B2, B3] = meshgrid(b2, b3);

%% ======================== Compute Delta_F and Delta_G ========================= %%
% Compute F1 and F2
F1 = (5*C*eta^2*L^2) * (((N-1)/(n-1)) * (((1)/(m^2)) * ((n-N)/N) + 1) * m^2 * I^2 ...
    + (1 - ((N-1)/(n-1))) * (1 - m^2 * ((((1)/(m^2)) * ((n-N)/N) + 1) - 1) * (N/(n-N))) * I^2);
F2 = (5*C*eta^2*L^2) * (((N-1)/(n-1)) * m^2 * I^2 + (1 - ((N-1)/(n-1))) * I^2);
Delta_F = F2 - F1; % Positive improvement

% Compute G1 and G2
G1 = (2*C*A2^2*L^2) * ((B3 + B2) .* (((1)/(m^2)) * ((n-N)/N) + 1) * m^2 * I^2 ...
    + B3 .* (1 - m^2 * ((((1)/(m^2)) * ((n-N)/N) + 1) - 1) * (N/(n-N))) * I^2);
G2 = (2*C*A2^2*L^2) * ((B3 + B2) * m^2 * I^2 + B3 * I^2);
Delta_G = G2 - G1; % Should be negative (improvement)

ZeroPlane = zeros(size(B2)); % Reference plane at z = 0

%% ======================== Plot the 3D Surface ========================= %%
figure;
hold on;

% Plot Zero Plane (Reference)
surf(b2, b3, ZeroPlane, 'FaceColor', 'black', 'EdgeColor', 'none', 'FaceAlpha', 0.3);

% Plot Improvement of Upper Bound (Delta_F + Delta_G)
surf(b2, b3, Delta_F + Delta_G, 'FaceColor', 'green', 'FaceAlpha', 0.3);

% Labels and Title
xlabel('b2');
ylabel('b3');
zlabel('Improvement of the upper bound');
title('3D Surface Plot of Delta_f');
grid on;
