function adj = kSpaceAdjointKWave2D(data, setting)
% kSpaceAdjointKWave2D - k-Wave adjoint PAT operator.
%
% This function uses the same input/output convention as
% kSpaceAdjointMirrorFFT2D:
%
%   adj = kSpaceAdjointKWave2D(data, setting)
%
% Input:
%   data     PAT sensor data, size Ny x Nt
%   setting  structure with fields:
%            Nx, Ny, dx, dy, Nt, dt, soundSpeed, mediumDensity
%
% Output:
%   adj      adjoint image, size Nx x Ny
%
% The source construction follows the additive-source k-Wave adjoint
% convention used in the MATLAB example:
%   r = time-reversed data
%   source.p = [r, 0] + [0, r], with the final column folded back
%
% Scaling follows the supervisor's checked adjoint code:
%   source scaling = rho * c * dx / (4 * dt)
%   output scaling = 1 / (rho * c^2)

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

% Sensor/source mask at the mirror interface.
sensor_mask = zeros(Nx_full, Ny);
sensor_mask(setting.Nx + 1, :) = 1;

% Construct additive adjoint source from time-reversed data.
data_rev = fliplr(data);
zero_col = zeros(size(data_rev, 1), 1);

adj_source = [data_rev, zero_col] + [zero_col, data_rev];
adj_source(:, end-1) = adj_source(:, end-1) + adj_source(:, end);
adj_source = adj_source(:, 1:end-1);

% Apply source scaling from supervisor's checked adjoint code.
source_scaling = setting.mediumDensity * setting.soundSpeed * ...
    kgrid.dx / (4 * kgrid.dt);
adj_source = adj_source * source_scaling;

source.p = adj_source;
source.p_mask = sensor_mask;
source.p_mode = 'additive';

% Record the final pressure field.
sensor.mask = sensor_mask;
sensor.record = {'p_final'};

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

sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

% k-Wave returns p_final as the full mirrored grid.
p_full = double(sensor_data.p_final);

% Apply output scaling from supervisor's checked adjoint code.
output_scaling = 1 / (setting.mediumDensity * setting.soundSpeed.^2);
p_full = p_full * output_scaling;

% Reduce mirrored full-grid image back to Nx x Ny.
% The forward embedding is [flipud(p0); p0], so the corresponding transpose
% combines the lower half with the flipped upper half.
upper = p_full(1:setting.Nx, :);
lower = p_full(setting.Nx+1:end, :);
adj = lower + flipud(upper);

end
