"""Differentiable Fourier PAT operators implemented with PyTorch.

The forward, inverse, and reference-adjoint constructions follow the Fourier
PAT formulation of Pan and Betcke (2022) and the BolinPan/CoronaeNet routines.
A differentiable Catmull-Rom interpolant replaces the reference ``griddata``
operation, while the spectral mappings, scaling factors, angular masks, and
domain extraction remain unchanged.

Automatic differentiation provides the transpose of the implemented discrete
forward operator. The constant-weight reference adjoint is retained separately
and must not be interpreted as that numerical transpose.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "unitary_fft",
    "unitary_ifft",
    "create_k_vector",
    "cubic_interp_axis0",
    "k_space_forward_mirror_fft_2d",
    "fpat_forward_2d",
    "fpat_forward_2d_batched",
    "k_space_inverse_mirror_fft_2d",
    "fpat_inverse_2d",
    "k_space_adjoint_mirror_fft_2d",
    "fpat_adjoint_2d",
]


def _complex_dtype(real_dtype):
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def unitary_fft(x, dim=None):
    """Compute an orthonormally scaled multidimensional FFT."""
    return (
        torch.fft.fftn(x, norm="ortho") if dim is None else torch.fft.fftn(x, dim=dim, norm="ortho")
    )


def unitary_ifft(x, dim=None):
    """Compute an orthonormally scaled multidimensional inverse FFT."""
    return (
        torch.fft.ifftn(x, norm="ortho")
        if dim is None
        else torch.fft.ifftn(x, dim=dim, norm="ortho")
    )


def create_k_vector(N, d, *, device=None, dtype=torch.float64):
    """1-D wavenumber vector (rad/m); matches k-Wave's ``kWaveGrid.kx`` convention."""
    if N % 2 == 0:
        nx = torch.arange(-(N // 2), N // 2, device=device, dtype=dtype) / N
    else:
        nx = torch.arange(-((N - 1) // 2), (N - 1) // 2 + 1, device=device, dtype=dtype) / N
    nx[N // 2] = 0.0
    return (2.0 * math.pi / d) * nx


def cubic_interp_axis0(P, idx):
    """Interpolate columns using a differentiable Catmull-Rom kernel.

    The source values are defined on uniform nodes along axis zero.
    Sample positions outside the source domain are returned as zero.

    Args:
        P: Tensor of shape ``(N, Ny)`` containing real or complex values.
        idx: Tensor of shape ``(M, Ny)`` containing continuous node indices.

    Returns:
        An interpolated tensor of shape ``(M, Ny)``.
    """
    n = P.shape[0]
    cdt = P.dtype
    rdt = idx.dtype
    i0 = torch.floor(idx)
    t = (idx - i0).to(rdt)
    i0 = i0.long()

    def g(off):
        return torch.gather(P, 0, (i0 + off).clamp(0, n - 1))

    # Catmull-Rom weights for nodes [-1, 0, 1, 2]; sum to 1, exact at t in {0,1}.
    wm1 = 0.5 * (-t + 2 * t**2 - t**3)
    w0 = 0.5 * (2 - 5 * t**2 + 3 * t**3)
    wp1 = 0.5 * (t + 4 * t**2 - 3 * t**3)
    wp2 = 0.5 * (-(t**2) + t**3)
    out = wm1.to(cdt) * g(-1) + w0.to(cdt) * g(0) + wp1.to(cdt) * g(1) + wp2.to(cdt) * g(2)
    valid = ((idx >= 0) & (idx <= n - 1)).to(cdt)
    return out * valid


def k_space_forward_mirror_fft_2d(p0, theta_max, c, Nt, d, dt):
    """Limited-angle forward fPAT operator (2-D): ``p0 (Nx, Ny) -> g (Ny, Nt)``.

    Faithful to ``kSpaceForwardMirrorFFT2D.m`` with cubic interpolation; differentiable.
    """
    rdt = p0.dtype if p0.is_floating_point() else torch.float64
    cdt = _complex_dtype(rdt)
    Nx, Ny = p0.shape
    dx, dy = d

    p0_m = torch.cat([torch.flip(p0, [0]), p0], dim=0).to(rdt)
    kx = create_k_vector(2 * Nx, dx, device=p0.device, dtype=rdt)
    ky = create_k_vector(Ny, dy, device=p0.device, dtype=rdt)
    w = c * create_k_vector(2 * Nt, c * dt, device=p0.device, dtype=rdt)

    P_kxky = torch.fft.fftshift(unitary_fft(torch.fft.ifftshift(p0_m.to(cdt))))

    W, KY = torch.meshgrid(w, ky, indexing="ij")
    rad = torch.clamp((W / c) ** 2 - KY**2, min=0.0)
    kx_new = torch.sign(W) * torch.sqrt(rad)

    # cubic interpolation from regular kx onto kx_new (per ky column)
    idx = (kx_new - kx[0]) / (kx[1] - kx[0])
    P_wk = cubic_interp_axis0(P_kxky, idx)

    P_wk = P_wk * math.sqrt(P_kxky.numel() / P_wk.numel())
    P_wk = P_wk * (kx_new != 0.0).to(cdt)  # evanescent cutoff

    # carry the exact DC sample (omega=0, ky=0)
    kx_g, ky_g = torch.meshgrid(kx, ky, indexing="ij")
    dc_val = P_kxky[(kx_g == 0) & (ky_g == 0)].sum()
    dc_mask = (W == 0.0) & (KY == 0.0)
    P_wk = torch.where(dc_mask, dc_val, P_wk)

    # real weight wf = w / (c^2 kx_new), DC -> 1/c, with limited-angle + admissible masks
    mask_angle = torch.abs(KY) <= (torch.abs(W) / c) * math.sin(theta_max)
    admissible = torch.abs(W) >= c * torch.abs(KY)
    wf = torch.zeros_like(W)
    nz = (kx_new != 0.0) & mask_angle & admissible
    wf[nz] = W[nz] / (c**2 * kx_new[nz])
    wf[dc_mask] = 1.0 / c
    weight = (wf * mask_angle.to(rdt) * admissible.to(rdt)).to(cdt)
    P_wk = P_wk * weight

    frec = torch.real(torch.fft.fftshift(unitary_ifft(torch.fft.ifftshift(P_wk))))
    gpos = math.sqrt(2.0) * frec[frec.shape[0] // 2 :, :]
    return gpos.transpose(0, 1).contiguous()  # (Ny, Nt)


def fpat_forward_2d(p0, theta_max, c, Nt, d, dt):
    """Differentiable limited-angle forward fPAT operator. ``p0 (Nx, Ny) -> (Ny, Nt)``."""
    return k_space_forward_mirror_fft_2d(p0, theta_max, c, Nt, d, dt)


def k_space_inverse_mirror_fft_2d(data, theta_max, c, N, d, dt):
    """Limited-angle inverse fPAT operator (2-D): ``data (Ny, Nt) -> p0 (Nx, Ny)``.

    Faithful to ``kSpaceInverseMirrorFFT2D.m`` (weight ``c^2 kx/w``, DC=c) with
    cubic interpolation; differentiable. The reference's ``griddata`` over kx_new is
    realised as cubic interpolation in w at ``w = c*sign(kx)*sqrt(kx^2+ky^2)`` — exact
    when the ky grids align, and uniform-source so cubic-convolution applies.
    """
    rdt = data.dtype if data.is_floating_point() else torch.float64
    cdt = _complex_dtype(rdt)
    Nx, Ny = N
    dx, dy = d
    Ny_data, Nt = data.shape
    assert Ny_data == Ny, f"data.shape[0]={Ny_data} but Ny={Ny}"

    dataT = data.transpose(0, 1).to(rdt)  # (Nt, Ny)
    data_m = torch.cat([torch.flip(dataT, [0]), dataT], dim=0) / math.sqrt(2.0)  # (2Nt, Ny)

    w = c * create_k_vector(2 * Nt, c * dt, device=data.device, dtype=rdt)
    ky = create_k_vector(Ny, dy, device=data.device, dtype=rdt)
    kx = create_k_vector(2 * Nx, dx, device=data.device, dtype=rdt)

    P = torch.fft.fftshift(unitary_fft(torch.fft.ifftshift(data_m.to(cdt))))  # (2Nt, Ny)
    W, KY = torch.meshgrid(w, ky, indexing="ij")
    P = P * (torch.abs(W) >= torch.abs(c * KY)).to(cdt)  # evanescent cutoff

    rad = torch.clamp((W / c) ** 2 - KY**2, min=0.0)
    kx_new = torch.sign(W) * torch.sqrt(rad)
    sf = torch.zeros_like(W)
    valid = W != 0
    sf[valid] = (c**2 * kx_new[valid]) / W[valid]
    dc = (W == 0.0) & (KY == 0.0)
    sf[dc] = c
    ky_max = torch.abs((W / c) * math.sin(theta_max))
    sf = sf * (torch.abs(KY) <= ky_max).to(rdt)
    P_scaled = P * sf.to(cdt)

    # interpolate (w, ky) -> (kx, ky) via w_target = c*sign(kx)*sqrt(kx^2 + ky^2)
    kxg, kyg = torch.meshgrid(kx, ky, indexing="ij")
    w_target = c * torch.sign(kxg) * torch.sqrt(kxg**2 + kyg**2)
    idx = (w_target - w[0]) / (w[1] - w[0])
    p_kxky = cubic_interp_axis0(P_scaled, idx)
    p_kxky = p_kxky * math.sqrt(P_scaled.numel() / p_kxky.numel())

    p_xy = torch.real(torch.fft.fftshift(unitary_ifft(torch.fft.ifftshift(p_kxky))))
    return p_xy[Nx:, :].contiguous()  # (Nx, Ny)


def fpat_inverse_2d(data, theta_max, c, N, d, dt):
    """Differentiable limited-angle inverse fPAT operator. ``data (Ny, Nt) -> (Nx, Ny)``."""
    return k_space_inverse_mirror_fft_2d(data, theta_max, c, N, d, dt)


def k_space_adjoint_mirror_fft_2d(data, theta_max, c, N, d, dt):
    """Limited-angle adjoint fPAT operator (2-D): ``data (Ny, Nt) -> p0 (Nx, Ny)``.

    Faithful to MATLAB kSpaceAdjointMirrorFFT2D.m. This follows the same
    pipeline as the inverse operator, but uses the adjoint spectral factor
    sf = c inside the limited-angle cone, instead of the inverse factor
    sf = c^2 * kx_new / w.

    NOTE: this is the MATLAB/paper adjoint reconstruction operator, NOT the
    exact numerical transpose of fpat_forward_2d. For the exact transpose
    of this PyTorch implementation, use autograd.
    """
    rdt = data.dtype if data.is_floating_point() else torch.float64
    cdt = _complex_dtype(rdt)
    Nx, Ny = N
    dx, dy = d
    Ny_data, Nt = data.shape
    assert Ny_data == Ny, f"data.shape[0]={Ny_data} but Ny={Ny}"

    dataT = data.transpose(0, 1).to(rdt)  # (Nt, Ny)
    data_m = torch.cat([torch.flip(dataT, [0]), dataT], dim=0) / math.sqrt(2.0)  # (2Nt, Ny)

    w = c * create_k_vector(2 * Nt, c * dt, device=data.device, dtype=rdt)
    ky = create_k_vector(Ny, dy, device=data.device, dtype=rdt)
    kx = create_k_vector(2 * Nx, dx, device=data.device, dtype=rdt)

    P = torch.fft.fftshift(unitary_fft(torch.fft.ifftshift(data_m.to(cdt))))  # (2Nt, Ny)
    W, KY = torch.meshgrid(w, ky, indexing="ij")
    P = P * (torch.abs(W) >= torch.abs(c * KY)).to(cdt)  # evanescent cutoff

    # MATLAB adjoint scaling:
    #     sf = c * ones(size(ky_max));
    #     sf(~mask_beta) = 0;
    ky_max = torch.abs((W / c) * math.sin(theta_max))
    sf = c * (torch.abs(KY) <= ky_max).to(rdt)
    P_scaled = P * sf.to(cdt)

    # interpolate (w, ky) -> (kx, ky) via w_target = c*sign(kx)*sqrt(kx^2 + ky^2)
    kxg, kyg = torch.meshgrid(kx, ky, indexing="ij")
    w_target = c * torch.sign(kxg) * torch.sqrt(kxg**2 + kyg**2)
    idx = (w_target - w[0]) / (w[1] - w[0])
    p_kxky = cubic_interp_axis0(P_scaled, idx)
    p_kxky = p_kxky * math.sqrt(P_scaled.numel() / p_kxky.numel())

    p_xy = torch.real(torch.fft.fftshift(unitary_ifft(torch.fft.ifftshift(p_kxky))))
    return p_xy[Nx:, :].contiguous()  # (Nx, Ny)


def fpat_adjoint_2d(data, theta_max, c, N, d, dt):
    """Differentiable limited-angle adjoint fPAT operator. ``data (Ny, Nt) -> (Nx, Ny)``."""
    return k_space_adjoint_mirror_fft_2d(data, theta_max, c, N, d, dt)


def _interp_batch(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Catmull-Rom interpolate (B,N,Ny) spectra along N at shared indices."""
    batch, count, _ = values.shape
    lower = torch.floor(indices).long()
    fraction = indices - torch.floor(indices)

    def values_at(offset: int) -> torch.Tensor:
        gather_indices = (lower + offset).clamp(0, count - 1)[None].expand(batch, -1, -1)
        return torch.gather(values, 1, gather_indices)

    wm1 = 0.5 * (-fraction + 2 * fraction**2 - fraction**3)
    w0 = 0.5 * (2 - 5 * fraction**2 + 3 * fraction**3)
    wp1 = 0.5 * (fraction + 4 * fraction**2 - 3 * fraction**3)
    wp2 = 0.5 * (-(fraction**2) + fraction**3)
    output = (
        wm1[None] * values_at(-1)
        + w0[None] * values_at(0)
        + wp1[None] * values_at(1)
        + wp2[None] * values_at(2)
    )
    return output * ((indices >= 0) & (indices <= count - 1))[None]


def fpat_forward_2d_batched(
    p0: torch.Tensor, theta_max: float, c: float, nt: int, d: tuple[float, float], dt: float
) -> torch.Tensor:
    """Map a ``(B,Nx,Ny)`` pressure batch to a ``(B,Ny,Nt)`` data batch."""
    if p0.ndim != 3:
        raise ValueError("p0 must have shape (batch, Nx, Ny)")
    _, nx, ny = p0.shape
    dx, dy = d
    dtype = p0.dtype
    complex_dtype = _complex_dtype(dtype)
    mirrored = torch.cat([torch.flip(p0, [-2]), p0], dim=-2).to(dtype)
    kx = create_k_vector(2 * nx, dx, device=p0.device, dtype=dtype)
    ky = create_k_vector(ny, dy, device=p0.device, dtype=dtype)
    w = c * create_k_vector(2 * nt, c * dt, device=p0.device, dtype=dtype)
    spectrum = torch.fft.fftn(
        torch.fft.ifftshift(mirrored.to(complex_dtype), dim=(-2, -1)), dim=(-2, -1), norm="ortho"
    )
    spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
    W, KY = torch.meshgrid(w, ky, indexing="ij")
    kx_new = torch.sign(W) * torch.sqrt(torch.clamp((W / c) ** 2 - KY**2, min=0.0))
    indices = (kx_new - kx[0]) / (kx[1] - kx[0])
    transformed = _interp_batch(spectrum, indices) * math.sqrt(
        spectrum.shape[-2] * spectrum.shape[-1] / (2 * nt * ny)
    )
    transformed = transformed * (kx_new != 0)[None]
    dc = (W == 0) & (KY == 0)
    dc_x = (kx == 0).nonzero(as_tuple=True)[0][0]
    dc_y = (ky == 0).nonzero(as_tuple=True)[0][0]
    transformed[:, dc] = spectrum[:, dc_x, dc_y][:, None]
    angle = torch.abs(KY) <= (torch.abs(W) / c) * math.sin(float(theta_max))
    admissible = torch.abs(W) >= c * torch.abs(KY)
    weight = torch.zeros_like(W)
    usable = (kx_new != 0) & angle & admissible
    weight[usable] = W[usable] / (c**2 * kx_new[usable])
    weight[dc] = 1.0 / c
    transformed = transformed * weight[None].to(complex_dtype)
    reconstructed = torch.fft.ifftn(
        torch.fft.ifftshift(transformed, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
    )
    reconstructed = torch.real(torch.fft.fftshift(reconstructed, dim=(-2, -1)))
    return math.sqrt(2.0) * reconstructed[:, nt:, :].transpose(-2, -1).contiguous()
