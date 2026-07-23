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

domDiam = sqrt((setting.Nx * setting.dx)^2 + ...
               (setting.Ny * setting.dy)^2);

setting.dt = setting.dx / setting.soundSpeed;
setting.Nt = ceil(domDiam / (setting.soundSpeed * setting.dt));
setting.t_array = (0:setting.Nt-1) * setting.dt;

setting.computation.theta_max = MLangle / 180 * pi;
setting.computation.interpolationMethodF = 'cubic';
setting.computation.interpolationMethodA = 'cubic';

data_fft = kSpaceForwardMirrorFFT2D(p0, setting);
adj_fft = kSpaceAdjointMirrorFFT2D(data_fft, setting);

data_kwave = kSpaceForwardKWave2D(p0, setting);
adj_kwave = kSpaceAdjointKWave2D(data_kwave, setting);

fprintf('data_fft size:    %d x %d\n', size(data_fft, 1), size(data_fft, 2));
fprintf('data_kwave size:  %d x %d\n', size(data_kwave, 1), size(data_kwave, 2));
fprintf('adj_fft size:     %d x %d\n', size(adj_fft, 1), size(adj_fft, 2));
fprintf('adj_kwave size:   %d x %d\n', size(adj_kwave, 1), size(adj_kwave, 2));
fprintf('max abs adj_fft:   %.6e\n', max(abs(adj_fft(:))));
fprintf('max abs adj_kwave: %.6e\n', max(abs(adj_kwave(:))));

outDir = fullfile(repoRoot, 'results', 'adjoint_once');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

save(fullfile(outDir, 'adjoint_once_data.mat'), ...
    'p0', 'setting', 'data_fft', 'data_kwave', 'adj_fft', 'adj_kwave');

vmax_p = max(abs(p0(:)));
vmax_adj = max([max(abs(adj_fft(:))), max(abs(adj_kwave(:))), eps]);

figure('Name', 'Adjoint comparison');
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
imagesc(p0, [-vmax_p vmax_p]);
axis image;
colorbar;
title('p0 fixed scale');

subplot(2, 3, 5);
imagesc(adj_fft, [-vmax_adj vmax_adj]);
axis image;
colorbar;
title('Fourier adj fixed scale');

subplot(2, 3, 6);
imagesc(adj_kwave, [-vmax_adj vmax_adj]);
axis image;
colorbar;
title('k-Wave adj fixed scale');

saveas(gcf, fullfile(outDir, 'adjoint_once_images.png'));
savefig(gcf, fullfile(outDir, 'adjoint_once_images.fig'));

figure('Name', 'Adjoint row trace');
row = round(setting.Nx / 2);
plot(p0(row, :), 'k', 'LineWidth', 1.5);
hold on;
plot(adj_fft(row, :), 'LineWidth', 1.5);
plot(adj_kwave(row, :), 'LineWidth', 1.5);
grid on;
legend('p0', 'Fourier adjoint', 'k-Wave adjoint');
title('Middle row trace');
xlabel('y index');

saveas(gcf, fullfile(outDir, 'adjoint_once_middle_row_trace.png'));
savefig(gcf, fullfile(outDir, 'adjoint_once_middle_row_trace.fig'));

fprintf('Saved adjoint test results to: %s\n', outDir);
