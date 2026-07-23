% Compare Fourier-PAT and k-Wave PAT forward/adjoint operators.
%
% This script is based on CoronaeNet/FourierOperator2D/
% example_LimitedAnlge_Fourier_Operatros.m, with additional calls to the
% isolated k-Wave forward and adjoint operators.
%
% It should run with MATLAB + k-Wave toolbox.

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

rng('default');

%% Load phantom
load(fullfile(repoRoot, 'CoronaeNet', 'measurements', 'ellipsePhantom.mat'));

%% Settings
MLangle = 45;

setting.Nx = size(p0, 1);
setting.Ny = size(p0, 2);
setting.dx = 1e-4;
setting.dy = 1e-4;
setting.soundSpeed = 1500;
setting.mediumDensity = 1000;
setting.CFL = 1;

domDiam = sqrt((setting.Nx * setting.dx)^2 + ...
               (setting.Ny * setting.dy)^2);

setting.dt = setting.CFL * setting.dx / setting.soundSpeed;
setting.Nt = ceil(domDiam / (setting.soundSpeed * setting.dt));
setting.t_array = (0:setting.Nt-1) * setting.dt;

setting.computation.theta_max = MLangle / 180 * pi;
setting.computation.interpolationMethodF = 'cubic';
setting.computation.interpolationMethodA = 'cubic';
setting.computation.interpolationMethodI = 'cubic';

%% Fourier forward / adjoint / inverse
fprintf('Running Fourier forward...\n');
data_fft = kSpaceForwardMirrorFFT2D(p0, setting);

fprintf('Running Fourier adjoint...\n');
adj_fft = kSpaceAdjointMirrorFFT2D(data_fft, setting);

fprintf('Running Fourier inverse...\n');
inv_fft = kSpaceInverseMirrorFFT2D(data_fft, setting);

%% k-Wave forward / adjoint
fprintf('Running k-Wave forward...\n');
data_kwave = kSpaceForwardKWave2D(p0, setting);

fprintf('Running k-Wave adjoint...\n');
adj_kwave = kSpaceAdjointKWave2D(data_kwave, setting);

%% Metrics
data_rel_l2 = norm(data_kwave(:) - data_fft(:)) / max(norm(data_fft(:)), eps);
adj_rel_l2 = norm(adj_kwave(:) - adj_fft(:)) / max(norm(adj_fft(:)), eps);

data_corr = sum(data_kwave(:) .* data_fft(:)) / ...
    max(norm(data_kwave(:)) * norm(data_fft(:)), eps);

adj_corr = sum(adj_kwave(:) .* adj_fft(:)) / ...
    max(norm(adj_kwave(:)) * norm(adj_fft(:)), eps);

fprintf('\nSizes:\n');
fprintf('  p0:          %d x %d\n', size(p0, 1), size(p0, 2));
fprintf('  data_fft:    %d x %d\n', size(data_fft, 1), size(data_fft, 2));
fprintf('  data_kwave:  %d x %d\n', size(data_kwave, 1), size(data_kwave, 2));
fprintf('  adj_fft:     %d x %d\n', size(adj_fft, 1), size(adj_fft, 2));
fprintf('  adj_kwave:   %d x %d\n', size(adj_kwave, 1), size(adj_kwave, 2));

fprintf('\nMax abs values:\n');
fprintf('  data_fft:    %.6e\n', max(abs(data_fft(:))));
fprintf('  data_kwave:  %.6e\n', max(abs(data_kwave(:))));
fprintf('  adj_fft:     %.6e\n', max(abs(adj_fft(:))));
fprintf('  adj_kwave:   %.6e\n', max(abs(adj_kwave(:))));

fprintf('\nComparisons:\n');
fprintf('  forward rel L2, k-Wave vs Fourier: %.6e\n', data_rel_l2);
fprintf('  forward corr,   k-Wave vs Fourier: %.6e\n', data_corr);
fprintf('  adjoint rel L2, k-Wave vs Fourier: %.6e\n', adj_rel_l2);
fprintf('  adjoint corr,   k-Wave vs Fourier: %.6e\n', adj_corr);

%% Save data
outDir = fullfile(repoRoot, 'results', 'compare_Fourier_kWave_PAT');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

save(fullfile(outDir, 'compare_Fourier_kWave_PAT_data.mat'), ...
    'p0', 'setting', ...
    'data_fft', 'data_kwave', ...
    'adj_fft', 'adj_kwave', 'inv_fft', ...
    'data_rel_l2', 'adj_rel_l2', 'data_corr', 'adj_corr');

%% Plot forward data
vmax_data = max([max(abs(data_fft(:))), max(abs(data_kwave(:))), eps]);

figure('Name', 'Forward data comparison');
subplot(1, 3, 1);
imagesc(data_fft, [-vmax_data, vmax_data]);
axis image;
colorbar;
title('Fourier forward data');
xlabel('time index');
ylabel('sensor index');

subplot(1, 3, 2);
imagesc(data_kwave, [-vmax_data, vmax_data]);
axis image;
colorbar;
title('k-Wave forward data');
xlabel('time index');
ylabel('sensor index');

subplot(1, 3, 3);
imagesc(data_kwave - data_fft);
axis image;
colorbar;
title('k-Wave - Fourier');
xlabel('time index');
ylabel('sensor index');

saveas(gcf, fullfile(outDir, 'forward_data_comparison.png'));
savefig(gcf, fullfile(outDir, 'forward_data_comparison.fig'));

%% Plot middle sensor trace
figure('Name', 'Forward middle sensor trace');
midSensor = round(setting.Ny / 2);
plot(setting.t_array * 1e6, data_fft(midSensor, :), 'LineWidth', 1.5);
hold on;
plot(setting.t_array * 1e6, data_kwave(midSensor, :), 'LineWidth', 1.5);
grid on;
legend('Fourier', 'k-Wave');
title('Middle sensor trace');
xlabel('time [\mus]');
ylabel('pressure');

saveas(gcf, fullfile(outDir, 'forward_middle_sensor_trace.png'));
savefig(gcf, fullfile(outDir, 'forward_middle_sensor_trace.fig'));

%% Plot adjoint images
vmax_adj = max([max(abs(adj_fft(:))), max(abs(adj_kwave(:))), eps]);

figure('Name', 'Adjoint image comparison');
subplot(2, 3, 1);
imagesc(p0);
axis image;
colorbar;
title('p0');

subplot(2, 3, 2);
imagesc(adj_fft);
axis image;
colorbar;
title('Fourier adjoint');

subplot(2, 3, 3);
imagesc(adj_kwave);
axis image;
colorbar;
title('k-Wave adjoint');

subplot(2, 3, 4);
imagesc(inv_fft);
axis image;
colorbar;
title('Fourier inverse');

subplot(2, 3, 5);
imagesc(adj_fft, [-vmax_adj, vmax_adj]);
axis image;
colorbar;
title('Fourier adj fixed scale');

subplot(2, 3, 6);
imagesc(adj_kwave, [-vmax_adj, vmax_adj]);
axis image;
colorbar;
title('k-Wave adj fixed scale');

saveas(gcf, fullfile(outDir, 'adjoint_image_comparison.png'));
savefig(gcf, fullfile(outDir, 'adjoint_image_comparison.fig'));

%% Plot middle row traces
figure('Name', 'Adjoint middle row trace');
midRow = round(setting.Nx / 2);
plot(p0(midRow, :), 'k', 'LineWidth', 1.5);
hold on;
plot(adj_fft(midRow, :), 'LineWidth', 1.5);
plot(adj_kwave(midRow, :), 'LineWidth', 1.5);
grid on;
legend('p0', 'Fourier adjoint', 'k-Wave adjoint');
title('Middle row trace');
xlabel('y index');

saveas(gcf, fullfile(outDir, 'adjoint_middle_row_trace.png'));
savefig(gcf, fullfile(outDir, 'adjoint_middle_row_trace.fig'));

fprintf('\nSaved comparison results to: %s\n', outDir);