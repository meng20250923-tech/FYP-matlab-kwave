function data = kSpaceForwardKWave2D(p0, setting)
% kSpaceForwardKWave2D - k-Wave forward PAT operator.
%
% This function uses the same input/output convention as
% kSpaceForwardMirrorFFT2D:
%
%   data = kSpaceForwardKWave2D(p0, setting)
%
% Input:
%   p0       initial pressure image, size Nx x Ny
%   setting  structure with fields:
%            Nx, Ny, dx, dy, Nt, dt, soundSpeed
%
% Output:
%   data     PAT sensor data, size Ny x Nt
%
% Geometry:
%   To match the mirrored Fourier setup, p0 is embedded as [flipud(p0); p0],
%   giving a 2*Nx by Ny k-Wave grid. The sensor line is placed at the mirror
%   interface.

% Basic checks
if size(p0, 1) ~= setting.Nx || size(p0, 2) ~= setting.Ny
    error('p0 size must be setting.Nx x setting.Ny.');
end

if ~isfield(setting, 'mediumDensity')
    setting.mediumDensity = 1000;
end

% Mirrored initial pressure, matching kSpaceForwardMirrorFFT2D.
p0_mirror = [flipud(p0); p0];

Nx_full = 2 * setting.Nx;
Ny = setting.Ny;

% k-Wave grid
kgrid = kWaveGrid(Nx_full, setting.dx, Ny, setting.dy);
kgrid.setTime(setting.Nt, setting.dt);

% Medium
medium.sound_speed = setting.soundSpeed * ones(Nx_full, Ny);
medium.density = setting.mediumDensity * ones(Nx_full, Ny);

% Initial pressure source
source.p0 = p0_mirror;

% Sensor mask: line sensor at the mirror interface.
% Python index n corresponds to MATLAB index setting.Nx + 1.
sensor.mask = zeros(Nx_full, Ny);
sensor.mask(setting.Nx + 1, :) = 1;

% k-Wave options.
% PML is outside the computational grid, so the physical grid size remains
% the same as the mirrored Fourier grid.
if ~isfield(setting, 'kwaveBoundary')
    setting.kwaveBoundary = 'pml';
end

switch setting.kwaveBoundary
    case 'pml'
        input_args = { ...
            'PMLInside', false, ...
            'PlotSim', false, ...
            'DataCast', 'single' ...
        };

        if isfield(setting, 'pmlSize')
            input_args = [input_args, {'PMLSize', setting.pmlSize}];
        end

    case 'periodic'
        input_args = { ...
            'PMLInside', false, ...
            'PMLSize', 0, ...
            'PMLAlpha', 0, ...
            'PlotSim', false, ...
            'DataCast', 'single' ...
        };

    otherwise
        error('Unknown setting.kwaveBoundary. Use ''pml'' or ''periodic''.');
end

% Run forward simulation.
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

% k-Wave returns Nsensors x Nt. Here Nsensors = Ny, so this matches the
% Fourier forward output convention: Ny x Nt.
data = double(sensor_data);

end
