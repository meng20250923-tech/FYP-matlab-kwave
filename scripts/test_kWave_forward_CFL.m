clear all; close all; clc;

repoRoot = fileparts(fileparts(mfilename('fullpath')));

addpath(genpath(fullfile(repoRoot, 'CoronaeNet', 'FourierOperator2D')));
addpath(genpath(fullfile(repoRoot, 'KWaveOperator2D')));

% Add the local k-Wave toolbox if it is bundled under Test3/external.
kwavePath = fullfile(repoRoot, 'external', 'k-wave', 'k-wave-toolbox-version-1.4.1');
if exist(kwavePath, 'dir')
    addpath(genpath(kwavePath));
end

if exist('kWaveGrid', 'file') ~= 2
    error(['k-Wave toolbox is not on the MATLAB path. ', ...
           'Add it manually or place it at: ', kwavePath]);
end

load(fullfile(repoRoot, 'CoronaeNet', 'measurements', 'ellipsePhantom.mat'));

MLangle = 45;

baseSetting.Nx = size(p0, 1);
baseSetting.Ny = size(p0, 2);
baseSetting.dx = 1e-4;
baseSetting.dy = 1e-4;
baseSetting.soundSpeed = 1500;
baseSetting.mediumDensity = 1000;
baseSetting.computation.theta_max = MLangle / 180 * pi;
baseSetting.computation.interpolationMethodF = 'cubic';

domDiam = sqrt((baseSetting.Nx * baseSetting.dx)^2 + ...
               (baseSetting.Ny * baseSetting.dy)^2);

cflValues = [1, 1/2, 1/4];
data_kwave = cell(numel(cflValues), 1);
settings = cell(numel(cflValues), 1);

for i = 1:numel(cflValues)
    cfl = cflValues(i);

    setting = baseSetting;
    setting.CFL = cfl;
    setting.dt = cfl * setting.dx / setting.soundSpeed;
    setting.Nt = ceil(domDiam / (setting.soundSpeed * setting.dt));
    setting.t_array = (0:setting.Nt-1) * setting.dt;

    fprintf('Running k-Wave forward with CFL = %.4g, dt = %.6g, Nt = %d\n', ...
        cfl, setting.dt, setting.Nt);

    data_kwave{i} = kSpaceForwardKWave2D(p0, setting);
    settings{i} = setting;

    fprintf('  data size: %d x %d, max abs: %.6e\n', ...
        size(data_kwave{i}, 1), size(data_kwave{i}, 2), max(abs(data_kwave{i}(:))));
end

outDir = fullfile(repoRoot, 'results', 'forward_CFL');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

save(fullfile(outDir, 'forward_CFL_data.mat'), ...
    'p0', 'baseSetting', 'settings', 'cflValues', 'data_kwave');

%% Plot sensor data images
figure('Name', 'k-Wave forward data for different CFL values');
for i = 1:numel(cflValues)
    subplot(1, numel(cflValues), i);
    imagesc(data_kwave{i});
    axis image;
    colorbar;
    title(sprintf('CFL = %.2g', cflValues(i)));
    xlabel('time index');
    ylabel('sensor index');
end
saveas(gcf, fullfile(outDir, 'forward_CFL_sensor_data.png'));
savefig(gcf, fullfile(outDir, 'forward_CFL_sensor_data.fig'));

%% Plot comparable time traces using physical time axis
figure('Name', 'Middle sensor trace for different CFL values');
midSensor = round(baseSetting.Ny / 2);
hold on;
for i = 1:numel(cflValues)
    t_us = settings{i}.t_array * 1e6;
    plot(t_us, data_kwave{i}(midSensor, :), 'LineWidth', 1.5);
end
grid on;
legend('CFL = 1', 'CFL = 1/2', 'CFL = 1/4');
title('Middle sensor trace');
xlabel('time [\mus]');
ylabel('pressure');
saveas(gcf, fullfile(outDir, 'forward_CFL_middle_trace.png'));
savefig(gcf, fullfile(outDir, 'forward_CFL_middle_trace.fig'));

%% Plot relative differences after interpolating to CFL=1 time grid
ref = data_kwave{1};
t_ref = settings{1}.t_array;
relDiff = zeros(numel(cflValues), 1);

figure('Name', 'Difference to CFL = 1 after time interpolation');
for i = 1:numel(cflValues)
    candidate = data_kwave{i};
    t_candidate = settings{i}.t_array;

    candidate_on_ref = zeros(size(ref));
    for s = 1:size(ref, 1)
        candidate_on_ref(s, :) = interp1(t_candidate, candidate(s, :), ...
            t_ref, 'linear', 0);
    end

    diffData = candidate_on_ref - ref;
    relDiff(i) = norm(diffData(:)) / max(norm(ref(:)), eps);

    subplot(1, numel(cflValues), i);
    imagesc(diffData);
    axis image;
    colorbar;
    title(sprintf('CFL %.2g - CFL 1, rel %.2e', cflValues(i), relDiff(i)));
    xlabel('time index');
    ylabel('sensor index');
end
saveas(gcf, fullfile(outDir, 'forward_CFL_difference_to_CFL1.png'));
savefig(gcf, fullfile(outDir, 'forward_CFL_difference_to_CFL1.fig'));

save(fullfile(outDir, 'forward_CFL_metrics.mat'), 'relDiff', 'cflValues');

fprintf('\nRelative differences to CFL = 1 after interpolation:\n');
for i = 1:numel(cflValues)
    fprintf('  CFL = %.4g: %.6e\n', cflValues(i), relDiff(i));
end

fprintf('\nSaved CFL test results to: %s\n', outDir);
