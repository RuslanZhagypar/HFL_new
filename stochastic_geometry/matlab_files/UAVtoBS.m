clc; clear; close all;

%% ======================== Simulation Parameters ========================= %%
radiusCluster = 500;      % Radius of cluster disk (for daughter points)
H_UAV = 120;             % UAV altitude
N = 9;                   % Number of UAVs

% Channel Parameters
mL = 4;                  % LOS Nakagami-m parameter
mN = 1;                  % NLOS Nakagami-m parameter
alphaL = 2.0;            % LOS Path loss exponent
alphaN = 3.5;            % NLOS Path loss exponent
P_u = 1.5;               % UAV transmission power
n0Squared = 4.14e-6;     % Noise power

% Resource Blocks (RBs)
RBs = 5;
activeProb =  floor((N+1)/RBs - 1);  % Probability of UAVs being active

% LOS Probability Model
a = 9.61;
b = 0.16;

% SINR Thresholds
taudB = (-20:2:10)';  
tau = 10.^(taudB ./ 10);

diskArea = pi * radiusCluster^2;
nbit = 10000;            % Number of simulation iterations
nsir = zeros(length(tau), 1); % Coverage probability counter
valid_n = 0;             % Counter for valid simulations

%% ===================== Monte Carlo Simulation ========================= %%
for n = 1:nbit
    fprintf("Simulation Iteration: %d\n", n);

    %% --------------- UAV Generation ---------------- %%
    t = 2 * pi * rand(N, 1);
    r = radiusCluster * sqrt(rand(N, 1));
    xxUAV = r .* cos(t);
    yyUAV = r .* sin(t);

    % Chosen UAV for communication
    xx_chosenUAV = 30;
    yy_chosenUAV = 282;

    xxUAV = [xxUAV; xx_chosenUAV];
    yyUAV = [yyUAV; yy_chosenUAV];

    %% --------------- Device Generation ---------------- %%
    xx_chosenUE = 0;
    yy_chosenUE = 0;  

    %% --------------- Establishing Main & Interfering Links ---------------- %%
    TxXDev = xx_chosenUE;  % Main transmitting device
    TxYDev = yy_chosenUE;

    TxXUAV = xxUAV(end);  % Main transmitting UAV
    TxYUAV = yyUAV(end);

    % Compute distances
    dists_down = pdist2([xxUAV, yyUAV], [TxXDev, TxYDev]);
    Oa = atand(H_UAV ./ dists_down);
    PLOS = 1 ./ (1 + a .* exp(-b .* (Oa - a)));

    pa = binornd(1, PLOS);
    ial = find(pa == 1);
    ian = find(pa == 0);

    % Compute channel gains
    if any(ial(:) == N+1)  
        ial = ial(ial ~= N+1);  
        dist = sqrt((TxXUAV - TxXDev)^2 + (TxYUAV - TxYDev)^2 + H_UAV^2);
        r_0 = dist^(-alphaL);
        h_0_down = gamrnd(mL, 1 / mL, 1);
    elseif any(ian(:) == N+1)
        ian = ian(ian ~= N+1);
        dist = sqrt((TxXUAV - TxXDev)^2 + (TxYUAV - TxYDev)^2 + H_UAV^2);
        r_0 = dist^(-alphaN);
        h_0_down = gamrnd(mN, 1 / mN, 1);
    end

    %% ================== Downlink Interference Calculation =================== %%
    xxUAV(end) = []; % Remove the main transmitting UAV
    yyUAV(end) = [];

    % Randomly activate interfering UAVs
    availableUAVs = 1:N; % Devices excluding the current UAV
    indices = randsample(availableUAVs, activeProb);

    ial = intersect(indices, ial);
    ian = intersect(indices, ian);

    % LOS Interference
    h_i_down_LOS = gamrnd(mL, 1 / mL, length(ial), 1);
    r_i_down_LOS = sqrt((TxXDev - xxUAV(ial)).^2 + (TxYDev - yyUAV(ial)).^2 + H_UAV^2).^(-alphaL);
    I_down_LOS = sum(P_u .* h_i_down_LOS .* r_i_down_LOS);

    % NLOS Interference
    h_i_down_NLOS = gamrnd(mN, 1 / mN, length(ian), 1);
    r_i_down_NLOS = sqrt((TxXDev - xxUAV(ian)).^2 + (TxYDev - yyUAV(ian)).^2 + H_UAV^2).^(-alphaN);
    I_down_NLOS = sum(P_u .* h_i_down_NLOS .* r_i_down_NLOS);

    % Compute Downlink SINR
    SIR_down = h_0_down * P_u * r_0 / (I_down_LOS + I_down_NLOS + n0Squared);
    SIRdB_down = 10 * log10(SIR_down);

    % Compute Coverage Probability
    for t = 1:length(tau)
        if (SIRdB_down >= taudB(t))
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

plot(taudB,[0.467356, 0.467336, 0.467243, 0.466861, 0.465538, 0.461778, 0.453137, 0.436953, 0.411626, 0.377417, 0.335911, 0.286558, 0.221268, 0.134082, 0.051153, 0.00944879])
