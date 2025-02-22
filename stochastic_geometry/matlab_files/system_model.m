clc; clear; close all;

%% ======================== Simulation Parameters ========================= %%
Nuav = 10;           % Number of UAVs
radiusCluster = 500; % Cluster radius

% Device Coordinates (xxDevs, yyDevs)
xxDevs = [-175.74, -99.90, 201.77, -373.08, 52.44, 222.05, -142.56, 364.76, 45.06, 215.25, ...
          -45.89, 344.55, 229.27, -436.25, 213.84, 168.35, 42.18, 93.86, -194.73, 312.96, ...
           439.06, -258.93, -2.29, 389.92, -315.23, 229.22, -39.15, -315.27, 245.62, 259.99, ...
          -237.59, -162.16, 27.71, 44.36, 218.40, -441.33, 102.95, 40.41, 301.98, -220.10, ...
          -312.88, 235.40, 89.83, -37.83, -9.85, -53.31, -66.88, -359.74, 205.03, 178.07];

yyDevs = [460.37, 47.34, -366.64, -154.24, -94.14, -4.28, 249.65, 234.74, -298.70, 293.86, ...
          -294.00, -99.75, 113.38, 150.39, -331.78, 218.57, -438.28, -463.76, -382.66, 340.50, ...
           110.68, -386.33, 309.07, -28.42, -67.10, 89.38, -117.38, -324.00, -188.61, 420.84, ...
          -197.52, 362.27, -331.76, 66.42, -102.96, 6.07, -178.58, 319.38, 239.79, -423.40, ...
          -169.41, -232.90, -1.62, 235.94, -212.84, 240.40, -444.84, 20.20, -223.86, 289.61];

% UAV Coordinates (xxUAV, yyUAV)
xxUAV = [-28.85, -336.62, 171.35, -64.93, 65.05, -357.88, 95.26, 123.45, 302.13, 355.54];
yyUAV = [-217.58, -184.91, 171.28, 48.69, -406.94, -33.12, 282.14, -162.26, -87.01, -15.38];

%% ===================== Device-to-UAV Assignment ========================= %%
% Find the closest UAV for each device
[k, MinDist] = dsearchn([xxUAV', yyUAV'], [xxDevs', yyDevs']);

% Create a cell array to store devices assigned to each UAV
myCell = cell(1, Nuav); 

% Assign devices to UAVs
for i = 1:Nuav
    myCell{i} = find(k == i);
end

%% ===================== Plotting the System Model ========================= %%
figure;
hold on;

% Plot Devices
scatter(xxDevs, yyDevs, 'filled', 'LineWidth', 1, 'DisplayName', 'Devices');

% Plot UAVs
scatter(xxUAV, yyUAV, 80, 'x', 'LineWidth', 2, 'DisplayName', 'UAVs');

% Plot Cluster Center (0,0)
scatter(0, 0, 80, '^', 'LineWidth', 2, 'DisplayName', 'Cluster Center');

% Draw Cluster Boundary
circle(0, 0, radiusCluster, 'black');

% Maintain Aspect Ratio
pbaspect([1 1 1]);

% Add Voronoi Diagram for UAV coverage areas (without adding multiple entries to legend)
[vx, vy] = voronoi(xxUAV, yyUAV);
voronoiPlot = plot(vx, vy, 'Color', 'blue'); % Store plot handle

% Set Plot Limits
xlim([-radiusCluster - 50, radiusCluster + 50]);
ylim([-radiusCluster - 50, radiusCluster + 50]);

% Axis Labels
xlabel('x (m)');
ylabel('y (m)');

% Add Legend (Only for key elements, not every Voronoi segment)
legend({'Devices', 'UAVs', 'Cluster Center', 'Voronoi Diagram'});

% Show the Figure
shg;
