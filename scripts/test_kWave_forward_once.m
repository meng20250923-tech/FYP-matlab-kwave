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

setting.Nx = size(p0, 1);
setting.Ny = size(p0, 2);
setting.dx = 1e-4;
setting.dy = 1e-4;
setting.soundSpeed = 1500;
setting.mediumDensity = 1000;

domDiam = sqrt((setting.Nx * setting.dx)^2 + (setting.Ny * setting.dy)^2);
setting.dt = setting.dx / setting.soundSpeed;
setting.Nt = ceil(domDiam / (setting.soundSpeed * setting.dt));
setting.t_array = (0:setting.Nt-1) * setting.dt;

setting.computation.theta_max = MLangle / 180 * pi;
setting.computation.interpolationMethodF = 'cubic';

data_fft = kSpaceForwardMirrorFFT2D(p0, setting);
data_kwave = kSpaceForwardKWave2D(p0, setting);

fprintf('FFT data size:    %d x %d\n', size(data_fft, 1), size(data_fft, 2));
fprintf('k-Wave data size: %d x %d\n', size(data_kwave, 1), size(data_kwave, 2));

figure;
subplot(1, 3, 1);
imagesc(data_fft);
axis image;
colorbar;
title('Fourier data');

subplot(1, 3, 2);
imagesc(data_kwave);
axis image;
colorbar;
title('k-Wave data');

subplot(1, 3, 3);
imagesc(data_kwave - data_fft);
axis image;
colorbar;
title('k-Wave - Fourier');

figure;
mid_sensor = round(size(data_fft, 1) / 2);
plot(data_fft(mid_sensor, :), 'LineWidth', 1.5);
hold on;
plot(data_kwave(mid_sensor, :), 'LineWidth', 1.5);
grid on;
legend('Fourier', 'k-Wave');
title('Middle sensor trace');
xlabel('time index');
ylabel('pressure');

outDir = fullfile(repoRoot, 'results', 'forward_once');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

save(fullfile(outDir, 'forward_once_data.mat'), ...
    'p0', 'setting', 'data_fft', 'data_kwave');

figs = findall(0, 'Type', 'figure');
for i = 1:numel(figs)
    saveas(figs(i), fullfile(outDir, sprintf('forward_once_figure_%02d.png', i)));
    savefig(figs(i), fullfile(outDir, sprintf('forward_once_figure_%02d.fig', i)));
end

fprintf('Saved results to: %s\n', outDir);
