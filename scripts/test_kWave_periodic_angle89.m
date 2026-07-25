clear all; close all; clc;
rng('default')

cd('/home/mao/Projects/FYP-matlab-kwave')

addpath(genpath('/home/mao/Projects/FYP/Test2/external/k-wave/k-wave-toolbox-version-1.4.1'))
addpath('CoronaeNet/FourierOperator2D')
addpath('KWaveOperator2D')

load('CoronaeNet/measurements/ellipsePhantom.mat')

MLangle = 89;

setting.Nx = size(p0,1);
setting.Ny = size(p0,2);
setting.dx = 1e-4;
setting.dy = 1e-4;
setting.soundSpeed = 1500;
setting.mediumDensity = 1000;

domDiam = sqrt((setting.Nx*setting.dx)^2 + (setting.Ny*setting.dy)^2);
setting.dt = setting.dx / setting.soundSpeed;
setting.Nt = ceil(domDiam / (setting.soundSpeed * setting.dt));
setting.t_array = (0:setting.Nt-1) * setting.dt;

setting.computation.theta_max = MLangle/180*pi;
setting.computation.interpolationMethodF = 'cubic';
setting.computation.interpolationMethodA = 'cubic';
setting.computation.interpolationMethodI = 'cubic';

setting.kwaveBoundary = 'periodic';

fprintf('Running FFT operators with angle %d...\n', MLangle)
data_fft = setting.soundSpeed * kSpaceForwardMirrorFFT2D(p0, setting);
adj_fft = kSpaceAdjointMirrorFFT2D((1/setting.soundSpeed) * data_fft, setting);
inv_fft = kSpaceInverseMirrorFFT2D((1/setting.soundSpeed) * data_fft, setting);

fprintf('Running periodic k-Wave operators...\n')
data_kw = kSpaceForwardKWave2D(p0, setting);
adj_kw = kSpaceAdjointKWave2D(data_kw, setting);
inv_kw = kSpaceInverseKWave2D(data_kw, setting);

fprintf('\nSizes:\n')
fprintf('  data_fft: %d x %d\n', size(data_fft,1), size(data_fft,2))
fprintf('  data_kw:  %d x %d\n', size(data_kw,1), size(data_kw,2))
fprintf('  adj_fft:  %d x %d\n', size(adj_fft,1), size(adj_fft,2))
fprintf('  adj_kw:   %d x %d\n', size(adj_kw,1), size(adj_kw,2))
fprintf('  inv_fft:  %d x %d\n', size(inv_fft,1), size(inv_fft,2))
fprintf('  inv_kw:   %d x %d\n', size(inv_kw,1), size(inv_kw,2))

fprintf('\nCorrelations:\n')
forward_corr = local_corr(data_fft(:), data_kw(:));
adjoint_corr = local_corr(adj_fft(:), adj_kw(:));
inverse_corr = local_corr(inv_fft(:), inv_kw(:));

fprintf('  forward corr: %e\n', forward_corr)
fprintf('  adjoint corr: %e\n', adjoint_corr)
fprintf('  inverse corr: %e\n', inverse_corr)

figure
subplot(2,3,1); imagesc(data_fft'); axis image; colorbar; title('FFT data angle 89')
subplot(2,3,2); imagesc(adj_fft); axis image; colorbar; title('FFT adj angle 89')
subplot(2,3,3); imagesc(inv_fft); axis image; colorbar; title('FFT inv angle 89')
subplot(2,3,4); imagesc(data_kw'); axis image; colorbar; title('kWave periodic data')
subplot(2,3,5); imagesc(adj_kw); axis image; colorbar; title('kWave periodic adj')
subplot(2,3,6); imagesc(inv_kw); axis image; colorbar; title('kWave periodic inv')

function r = local_corr(x, y)
x = double(x(:));
y = double(y(:));

x = x - mean(x);
y = y - mean(y);

denom = norm(x) * norm(y);
if denom == 0
    r = NaN;
else
    r = (x' * y) / denom;
end
end