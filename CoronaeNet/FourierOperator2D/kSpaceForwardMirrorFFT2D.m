function f = kSpaceForwardMirrorFFT2D(p,setting)
% kSpaceForwardMirrorFFT2D.m - returns the PAT data obtained from
% limited-angle Forward Fourier PAT operator in flat sensor geometry
%
% Note: The core structure of this code is based on the Fourier inversion in k-wave toolbox kspaceLineRecon.m
%
%
% Inputs:
%     p          initial pressure (image)
%     setting    a structure containing parameters
%                'Nx' - number of voxels in x direction
%                'Ny' - number of voxels in y direction
%                'soundSpeed' - sound speed in the medium in [m/s]
%                'dx', 'dy', dz' - voxel size in different directions
%                'Nt' - number of time steps
%                'dt' - temporal resolution of the forward solver
%                'computation.interpolationMethodF' - forward interpolation method
%                'omputation.theta_max' - maximum limited angle 
%
% Outputs:
%      f  - the PAT sensor data
%
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke


% interpolation method
interpolationMethod = setting.computation.interpolationMethodF;

% maximal angle of incidence (0, pi/2)
theta_max = setting.computation.theta_max;

% speed of sound
c = setting.soundSpeed;

% computer kgrid for data
datakgrid = kWaveGrid(2*setting.Nt, c.*setting.dt, setting.Ny, setting.dy);

% from the grid for kx, create a computational grid for w using the
% relation dx = dt*c; this represents the initial sampling of p(w, ky)
w = c .* datakgrid.kx;

% mirror the p0 for fft use
p0Mask = [flipdim(p, 1); p];

% construct kgrid with double size Nx
p0Dkgrid = kWaveGrid(2*setting.Nx,setting.dx,setting.Ny,setting.dy);

% compute the unitary FFT of the input data p0 to yield p(kx, ky)
p_kxky = fftshift(fftn(ifftshift(p0Mask)))./sqrt(numel(p0Mask));

% interpolation from regular (kx, ky) grid to regular (kx_new, ky) grid
% calculate the values of kx corresponding to the (ky, w) grid
kx_new = real(sign(w).*sqrt((w./c).^2 - datakgrid.ky.^2));
switch interpolationMethod
    % using 1D trigonometric interpolation (slow!)
    case 'trig'
        Ncol = size(kx_new,2);
        p_wky = [];
        for i = 1:Ncol
            kx_new_col = kx_new(:,i);
            kx_col = p0Dkgrid.kx(:,i);
            p_kxky_col = p_kxky(:,i);          
            % call 1D interpolation function
            P_interp = triginterp(kx_new_col,kx_col,p_kxky_col);
            p_wky = [p_wky,P_interp];
        end    
    % using griddata interpolation options
    case {'nearest','linear','natural','cubic','v4'}
        p_wky = griddata(p0Dkgrid.ky,p0Dkgrid.kx,p_kxky,datakgrid.ky,kx_new, interpolationMethod);
end

% rescale for unitary FFT
p_wky = p_wky.* sqrt(numel(p_kxky)/numel(p_wky));

% set values outside the interpolation range to zero
p_wky(isnan(p_wky)) = 0;

% remove any evanescent parts of the field 
p_wky = p_wky.*(kx_new~=0); % remove the bow-tie from data term 

% keep the dc term
p_wky((datakgrid.ky==0)&(w==0)) = p_kxky((p0Dkgrid.kx==0)&(p0Dkgrid.ky==0));

% calculate the regularised weighting factor 
wf = w ./ (c.^2 * kx_new);         % unregularised weighting factor
wf(isnan(wf)) = 0;                 % removing NaNs (when w_new = ky_new = 0)
wf(datakgrid.ky==0 & w==0) = 1./c; % keep the DC component (singularities)


% compute the regularized factor
ky_max = abs((w/c)*sin(theta_max));
wf(abs(datakgrid.ky) > ky_max) = 0;

% apply the weighting factor
p_wky_wf = wf .* p_wky; 

% compute the inverse FFT of p(kx, ky, w) to yield p(x, y, t)
frec = real(fftshift(ifftn(ifftshift(p_wky_wf)))*sqrt(numel(p_wky_wf)));

% take result for positive times (as it has been designed to be symmetric)
fast_kspace = sqrt(2)*frec(end/2+1:end,:); % rescale to keep norm
f = fast_kspace.';

