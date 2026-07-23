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
data = kSpaceForwardMirrorFFT2D(p0,setting);


%% Adjoint PAT via FFT:
setting.computation.interpolationMethodA = 'cubic'; % adj: {'nearest','linear','cubic'}
adj = kSpaceAdjointMirrorFFT2D(data,setting);


%% Inverse PAT via FFT:
setting.computation.interpolationMethodI = 'cubic'; % inv: {'nearest','linear','cubic'}
inv = kSpaceInverseMirrorFFT2D(data,setting);


%% display and compare
subplot(2,2,1);imagesc(p0);axis image;colorbar;title('p0')
subplot(2,2,2);imagesc(data');axis image;colorbar;title('data')
subplot(2,2,3);imagesc(adj);axis image;colorbar;title('adj')
subplot(2,2,4);imagesc(inv);axis image;colorbar;title('inv')



