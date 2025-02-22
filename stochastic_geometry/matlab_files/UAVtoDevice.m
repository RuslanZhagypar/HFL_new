clc; clear; close all;

%% ======================== Simulation Parameters ========================= %%
radiusCluster = 500;      % Radius of cluster disk (for daughter points)
H_UAV = 120;             % UAV altitude
N = 9;                   % Number of UAVs
N_devs = 49;             % Number of devices

% Channel Parameters
mL = 4;                  % LOS Nakagami-m parameter
mN = 1;                  % NLOS Nakagami-m parameter
alphaL = 2.0;            % LOS Path loss exponent
alphaN = 3.5;            % NLOS Path loss exponent
P_d = 0.75;              % Device transmission power
P_u = 1.5;               % UAV transmission power
n0Squared = 4.14e-6;     % Noise power

% Resource Blocks (RBs)
RB_uav = 5;
RB_devs = 10;

% Activation probabilities
activeProb_uav = (N / RB_uav) / N;  % Probability of UAVs being active
activeProb_devs = (N_devs / RB_devs) / N_devs;

% LOS Probability Model
a = 9.61;
b = 0.16;

% SINR Thresholds
taudB = (-20:2:10)';  
tau = 10.^(taudB ./ 10);

diskArea = pi * radiusCluster^2;
nbit = 10000;            % Number of simulation iterations
nsir = zeros(length(tau), 1); % Coverage probability counter

valid_n = 0;

%% ===================== Monte Carlo Simulation ========================= %%
for n = 1:nbit
    fprintf("Simulation Iteration: %d\n", n);

    %% --------------- UAV Generation ---------------- %%
    t = 2 * pi * rand(N, 1);
    r = radiusCluster * sqrt(rand(N, 1));
    xxUAV = r .* cos(t);
    yyUAV = r .* sin(t);
    zzUAV = H_UAV + 0 * t;

    % Chosen UAV for communication y_0
    xx_chosenUAV = 30;
    yy_chosenUAV = 282;

    xxUAV = [xxUAV; xx_chosenUAV];
    yyUAV = [yyUAV; yy_chosenUAV];

    %% --------------- Device Generation ---------------- %%
    t = 2 * pi * rand(N_devs, 1);
    r = radiusCluster * sqrt(rand(N_devs, 1));
    xxDevices = r .* cos(t);
    yyDevices = r .* sin(t);

    % Remove one random device (to simulate a typical user)
    inx = randi([1 N]);  
    xxDevices(inx) = [];
    yyDevices(inx) = [];

    % Chosen UE for communication x_0
    xx_chosenUE = -162;
    yy_chosenUE = 362;

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
        h_0_up = gamrnd(mL, 1 / mL, 1);
    elseif any(ian(:) == N+1)
        ian = ian(ian ~= N+1);
        dist = sqrt((TxXUAV - TxXDev)^2 + (TxYUAV - TxYDev)^2 + H_UAV^2);
        r_0 = dist^(-alphaN);
        h_0_down = gamrnd(mN, 1 / mN, 1);
        h_0_up = gamrnd(mN, 1 / mN, 1);
    end

    %% ================== Downlink Interference Calculation =================== %%
    randomArray = rand(N-1,1);
    indices = find(randomArray<=activeProb_uav);
    % indices = randi([1, N]);  
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

    %% ================== Uplink Interference Calculation =================== %%
    dists_up = pdist2([xxDevices, yyDevices], [TxXUAV, TxYUAV]);
    Oa2 = atand(H_UAV ./ dists_up);
    PLOS2 = 1 ./ (1 + a .* exp(-b .* (Oa2 - a)));

    pa2 = binornd(1, PLOS2);
    ial2 = find(pa2 == 1);
    ian2 = find(pa2 == 0);

    indices2 = find(rand(N_devs, 1) <= activeProb_devs);
    if length(indices2) > 4
        indices2 = indices2(1:4);
    end
    ial2 = intersect(indices2, ial2);
    ian2 = intersect(indices2, ian2);

    % LOS Interference
    h_i_up_LOS = gamrnd(mL, 1 / mL, length(ial2), 1);
    r_i_up_LOS = sqrt((TxXUAV - xxDevices(ial2)).^2 + (TxYUAV - yyDevices(ial2)).^2 + H_UAV^2).^(-alphaL);
    I_up_LOS = sum(P_d .* h_i_up_LOS .* r_i_up_LOS);

    % NLOS Interference
    h_i_up_NLOS = gamrnd(mN, 1 / mN, length(ian2), 1);
    r_i_up_NLOS = sqrt((TxXUAV - xxDevices(ian2)).^2 + (TxYUAV - yyDevices(ian2)).^2 + H_UAV^2).^(-alphaN);
    I_up_NLOS = sum(P_d .* h_i_up_NLOS .* r_i_up_NLOS);

    % Compute Uplink SINR
    SIR_up = h_0_up * P_d * r_0 / (I_up_LOS + I_up_NLOS + n0Squared);
    SIRdB_up = 10 * log10(SIR_up);

    % Compute Coverage Probability
    for t = 1:length(tau)
        if (SIRdB_down >= taudB(t) && SIRdB_up >= taudB(t))
            nsir(t) = nsir(t) + 1;
        end
    end
end

Pc_simul = nsir ./ nbit;
figure;
plot(taudB, Pc_simul, 'x', 'LineWidth', 1.5, 'MarkerSize', 8);
xlabel('SINR Threshold, \theta');
ylabel('Coverage Probability');
grid on;
