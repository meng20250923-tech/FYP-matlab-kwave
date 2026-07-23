function p = kSpaceAdjointMirrorFFT2D(f,setting)
% kSpaceAdjointMirrorFFT2D.m - recovers PAT image from PAT data via
% limited-angle Adjoint Fourier PAT operator in flat sensor geometry
%
% Note: The core structure of this code is based on the Fourier inversion in k-wave toolbox kspaceLineRecon.m
%
%
% Inputs:
%     f          PAT data
%     setting    a structure containing parameters
%                'Nx' - number of voxels in x direction
%                'Ny' - number of voxels in y direction
%                'soundSpeed' - sound speed in the medium in [m/s]
%                'dx', 'dy', dz' - voxel size in different directions
%                'Nt' - number of time steps
%                'dt' - temporal resolution of the forward solver
%                'computation.interpolationMethodA' - adjoint interpolation method
%                'omputation.theta_max' - maximum limited angle 
%
% Outputs:
%      p  - reconstructed image
%
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke


% skip warning massages
warning off

% interpolation method
interpolationMethod = setting.computation.interpolationMethodA;

% maximal angle of incidence (0, pi/2)
theta_max = setting.computation.theta_max;

% mirror the time domain data about t = 0 to allow the cosine transform to be computed using an FFT (p_ty)
fmask = f';
fmask = [flipdim(fmask, 1); fmask]/sqrt(2); % rescale to keep the norm

% speed of sound
c = setting.soundSpeed;

% extract the size of mirrored input data
[dataNt, dataNy] = size(fmask);

% computer kgrid for data
datakgrid = kWaveGrid(dataNt, c.*setting.dt, dataNy, setting.dy);

% from the grid for kx, create a computational grid for w using the
% relation dx = dt*c; this represents the initial sampling of p(w, ky)
w = c .* datakgrid.kx;

% compute the FFT of the input data p(t, y) to yield p(w, ky)
p_wky = fftshift(fftn(ifftshift(fmask)))./sqrt(numel(fmask));
p_wky(abs(w) < abs(c .* datakgrid.ky)) = 0;

% compute regularised factor
%sf(abs(datakgrid.ky) > ky_max) = 0;
ky_max = abs((w/c)*sin(theta_max)); % via angle
mask_beta = (abs(datakgrid.ky) <= ky_max);

% adjoint no factor required as shown in paper
sf = c*ones(size(ky_max));
sf(~mask_beta) = 0;

% scale with the factor
p_wky_sf = p_wky.*sf;

% construct kgrid with double size Nx
p0Dkgrid = kWaveGrid(2*setting.Nx,setting.dx,setting.Ny,setting.dy);

% interpolation from regular (kx, ky) grid to regular (kx_new, ky) grid
% calculate the values of kx corresponding to the (ky, w) grid
kx_new = real(sign(w).*sqrt((w./c).^2 - datakgrid.ky.^2));

% backward interpolation (mirrored p0)
p_wky_r = griddata(kx_new,datakgrid.ky,p_wky_sf,p0Dkgrid.kx,p0Dkgrid.ky,interpolationMethod);
p_wky_r(isnan(p_wky_r)) = 0;

% rescale for unitary FFT
p_wky_r = p_wky_r.* sqrt(numel(p_wky_sf)/numel(p_wky_r));

% recover mirrored p0 via inverse Fourier transform 
p_xy = real(fftshift(ifftn(ifftshift(p_wky_r)))).*sqrt(numel(p_wky_r));

% remove the left part of the mirrored data which corresponds to the
% negative part of the mirrored time data
p = p_xy((setting.Nx+1):end,:);

