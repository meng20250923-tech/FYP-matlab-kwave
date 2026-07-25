function inv = kSpaceInverseKWave2D(data, setting)
% kSpaceInverseKWave2D - k-Wave inverse PAT operator using time reversal.
%
% This function uses the same input/output convention as
% kSpaceInverseMirrorFFT2D:
%
%   inv = kSpaceInverseKWave2D(data, setting)
%
% Input:
%   data     PAT sensor data, size Ny x Nt
%   setting  structure with fields Nx, Ny, dx, dy, Nt, dt, soundSpeed
%
% Output:
%   inv      time-reversal reconstruction, size Nx x Ny
%
% Geometry:
%   The reconstruction is performed on the mirrored 2*Nx x Ny grid, matching
%   the forward k-Wave operator. The final image is reduced back to Nx x Ny.

if ~isfield(setting, 'mediumDensity')
    setting.mediumDensity = 1000;
end

if size(data, 1) ~= setting.Ny
    error('data must have size Ny x Nt.');
end

if size(data, 2) ~= setting.Nt
    error('data time dimension must equal setting.Nt.');
end

Nx_full = 2 * setting.Nx;
Ny = setting.Ny;

% k-Wave grid
kgrid = kWaveGrid(Nx_full, setting.dx, Ny, setting.dy);
kgrid.setTime(setting.Nt, setting.dt);

% Medium
medium.sound_speed = setting.soundSpeed * ones(Nx_full, Ny);
medium.density = setting.mediumDensity * ones(Nx_full, Ny);

% Sensor mask at the mirror interface, matching the forward operator.
sensor.mask = zeros(Nx_full, Ny);
sensor.mask(setting.Nx + 1, :) = 1;

% k-Wave time reversal. k-Wave expects the boundary data as Nsensors x Nt.
sensor.time_reversal_boundary_data = data;

% No initial pressure source is used for time reversal.
source.p0 = zeros(Nx_full, Ny);

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

p_full = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
p_full = double(p_full);

% Reduce mirrored reconstruction back to Nx x Ny.
upper = p_full(1:setting.Nx, :);
lower = p_full(setting.Nx+1:end, :);
inv = 0.5 * (lower + flipud(upper));

end
