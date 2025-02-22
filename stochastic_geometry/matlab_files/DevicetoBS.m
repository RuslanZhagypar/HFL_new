clc; clear; close all;

%% ======================== Simulation Parameters ========================= %%
radiusCluster = 1000;    % Radius of cluster disk (for daughter points)
H_BS = 0;               % Base Station height
N_devs = 50;            % Number of devices

% Channel Parameters
m = 1;                  % Nakagami-m parameter
alpha = 2;              % Path loss exponent
P_d = 0.25;             % Device transmission power
n0Squared = 4.14e-6;    % Noise power

rho = 0.01;
mu = 1;

% SINR Thresholds
taudB = (-30:2:10)';  
tau = 10.^(taudB ./ 10);

nbit = 900000;          % Number of simulation iterations
nsir = zeros(length(tau), 1); % Coverage probability counter
valid_n = 0;            % Counter for valid simulations

%% ===================== Monte Carlo Simulation ========================= %%
for n = 1:nbit
    fprintf("Simulation Iteration: %d\n", n);

    %% --------------- Device Generation ---------------- %%
    t = 2 * pi * rand(N_devs, 1);
    r = radiusCluster * sqrt(rand(N_devs, 1));
    xxDevices = r .* cos(t);
    yyDevices = r .* sin(t);

    % Select a random main transmitting device
    indexMainDev = randi([1, N_devs]);
    TxXDev = xxDevices(indexMainDev);  
    TxYDev = yyDevices(indexMainDev);

    % Remove the main transmitting device from interference set
    xxDevices(indexMainDev) = [];
    yyDevices(indexMainDev) = [];
    interfDevX = xxDevices;
    interfDevY = yyDevices;

    %% --------------- Establishing Main Link ---------------- %%
    dist = sqrt((0 - TxXDev).^2 + (0 - TxYDev).^2 + H_BS^2);
    r_0 = dist;
    h_0_up = gamrnd(m, 1/m, 1);

    %% ================== Uplink Interference Calculation =================== %%
    r_i_up = sqrt((0 - interfDevX).^2 + (0 - interfDevY).^2 + H_BS^2);

    % Apply power constraints
    r_i_up_filtered = r_i_up;

    % Compute Interference
    h_i_up = gamrnd(m, 1/m, length(r_i_up_filtered), 1);
    I_up = sum(rho .* h_i_up .* r_i_up_filtered.^(alpha * (mu - 1)));

    % Compute Uplink SINR
    SIR_up = h_0_up * rho * r_0^(-alpha) * r_0^(mu * alpha) / (I_up + n0Squared);
    SIRdB_up = 10 * log10(SIR_up);

    % Compute Coverage Probability
    for t = 1:length(tau)
        if (SIRdB_up >= taudB(t))
            nsir(t) = nsir(t) + 1;
        end
    end
    valid_n = valid_n + 1;
end

%% ===================== Coverage Probability Plot ========================= %%
Pc_simul = nsir ./ valid_n;

figure;
hold on;
plot(taudB, Pc_simul, 'x', 'DisplayName', 'Simulation', 'LineWidth', 1.5, 'MarkerSize', 8, 'Color', 'red');
xlabel('SINR Threshold, \theta', 'FontSize', 12);
ylabel('Coverage Probability', 'FontSize', 12);
grid on;
legend show;
