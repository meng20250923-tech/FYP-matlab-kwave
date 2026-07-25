% An example to demonstrate limited-angle PAT Fourier operatros on 2D 
% ellipse phantom with line sensors geometry.
%
% Add k-wave into the path for running this script.
%
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke


clear all; close all; clc;

% Reset rand, randn, randi to default seed
rng('default')

addpath KWaveOperator2D/

% define path
path = 'CoronaeNet/';

% load phantom
load([path,'measurements/ellipsePhantom.mat']) 

% define maximum limited angle
MLangle = 45; % 45 degree


%% setting parameters
setting.Nx = size(p0,1);  % number of grid points in the x (row) direction
setting.Ny = size(p0,2);  % number of grid points in the y (column) direction
setting.dx = 1e-4; % grid point spacing in the x direction [m]
setting.dy = 1e-4; % grid point spacing in the x direction [m]
c = 1500; % speed of sound

% set up time steps manually, matching the dx discretisation 1:1
domDiam =  sqrt((setting.Nx*setting.dx)^2 + (setting.Ny*setting.dy)^2); % domain diameter
setting.dt = setting.dx/c;
setting.Nt = ceil(domDiam/(c*setting.dt));
setting.t_array = (0:setting.Nt-1)*setting.dt;
setting.soundSpeed = c;

% limited angle
setting.computation.theta_max = MLangle/180*pi;


%% Forward PAT via FFT: compute PAT data 
setting.computation.interpolationMethodF = 'cubic'; % fwd: {'trig','nearest','linear','cubic'}
data = c*kSpaceForwardMirrorFFT2D(p0,setting);


%% Adjoint PAT via FFT:
setting.computation.interpolationMethodA = 'cubic'; % adj: {'nearest','linear','cubic'}
adj = kSpaceAdjointMirrorFFT2D(1/c*data,setting); %note undoing the "c" scaling is part of the adjoint computation in (omega,kS) non-normalised frequency, normalised frequency omega would be omega/c


%% Inverse PAT via FFT:
setting.computation.interpolationMethodI = 'cubic'; % inv: {'nearest','linear','cubic'}
inv = kSpaceInverseMirrorFFT2D(1/c*data,setting); %note undoing the "c" scaling is part of the inverse computation in non-normalised frequency omega


%% display and compare
figure
subplot(2,2,1);imagesc(p0);axis image;colorbar;title('p0')
subplot(2,2,2);imagesc(data');axis image;colorbar;title('FFT: data')
subplot(2,2,3);imagesc(adj);axis image;colorbar;title('FFT: adj')
subplot(2,2,4);imagesc(inv);axis image;colorbar;title('FFT: inv')



%% Forward PAT via kWave: compute PAT data setting.computation.interpolationMethodF = 'cubic'; % fwd: {'trig','nearest','linear','cubic'}
dataKW = kSpaceForwardKWave2D(p0,setting);


%% Adjoint PAT via kWave:
adjKW = kSpaceAdjointKWave2D(dataKW,setting);
%% Mixed operator: Forward FFT data into kWave adjoint
fwdFFTadjKW = kSpaceAdjointKWave2D(data,setting); %note we are NOT undoing the scaling if used with kWave

% to be implemented
%% Inverse PAT via kWave:
invKW = kSpaceInverseKWave2D(dataKW,setting);
%% Mixed operator: Forward FFT data into kWave inverse
fwdFFTinvKW = kSpaceInverseKWave2D(data,setting);

%% display and compare
figure
subplot(2,3,1);imagesc(p0);axis image;colorbar;title('kW: p0')
subplot(2,3,2);imagesc(dataKW');axis image;colorbar;title('kW: data')
subplot(2,3,3);imagesc(adjKW);axis image;colorbar;title('kW: adj')
subplot(2,3,4);imagesc(fwdFFTadjKW);axis image;colorbar;title('FFT: fwd, kW: adj')
subplot(2,3,5);imagesc(invKW);axis image;colorbar;title('KW: inv')
subplot(2,3,6);imagesc(fwdFFTinvKW);axis image;colorbar;title('FFT: fwd, kW: inv')
