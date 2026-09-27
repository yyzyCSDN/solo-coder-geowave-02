from __future__ import annotations
import numpy as np

# 4th-order centred stencil for the 2nd spatial derivative
_C1 = 4.0 / 3.0
_C2 = -1.0 / 12.0

def causal_ricker(peak_hz: float, dt_s: float) -> np.ndarray:
    """Ricker wavelet, peak delayed by 1.2/f0 so it starts near zero."""
    t0 = 1.2 / peak_hz
    nt = int(round(2.0 * t0 / dt_s)) + 1
    t = np.arange(nt) * dt_s
    p = (np.pi * peak_hz * (t - t0)) ** 2
    return (1.0 - 2.0 * p) * np.exp(-p)

def sponge_taper(nz: int, nx: int, cells: int, free_surface: bool, coeff: float=0.015) -> np.ndarray:
    """Cerjan-style damping factor, 1 inside, decaying inside the sponge frame."""
    damp = np.ones((nz, nx))
    if cells <= 0:
        return damp
    profile = np.exp(-(coeff * (cells - np.arange(cells))) ** 2)
    damp[:, :cells] = np.minimum(damp[:, :cells], profile[None, :])
    damp[:, nx - cells:] = np.minimum(damp[:, nx - cells:], profile[None, ::-1])
    damp[nz - cells:, :] = np.minimum(damp[nz - cells:, :], profile[::-1, None])
    if not free_surface:
        damp[:cells, :] = np.minimum(damp[:cells, :], profile[:, None])
    return damp

class AcousticModeler2D:
    """Constant-density acoustic solver, 2nd order in time / 4th in space.

    Top boundary is a free surface (p = 0) when enabled, remaining edges use a
    sponge absorbing frame. The adjoint gradient is the gradient of
    J = 1/2 * dt * sum(residual**2) with respect to velocity (m/s).
    """

    def __init__(self, nz: int, nx: int, dx_m: float, dt_s: float, sponge_cells: int=15, free_surface: bool=True):
        self.nz, self.nx = nz, nx
        self.dx_m = dx_m
        self.dt_s = dt_s
        self.free_surface = free_surface
        self.damp = sponge_taper(nz, nx, sponge_cells, free_surface)
        self.sponge_cells = sponge_cells

    def _laplacian(self, p: np.ndarray, out: np.ndarray) -> None:
        c = _C2 * (p[4:, 2:-2] + p[:-4, 2:-2] + p[2:-2, 4:] + p[2:-2, :-4]) \
          + _C1 * (p[3:-1, 2:-2] + p[1:-3, 2:-2] + p[2:-2, 3:-1] + p[2:-2, 1:-3]) \
          - 5.0 * p[2:-2, 2:-2]
        out[2:-2, 2:-2] = c / (self.dx_m ** 2)

    def forward(self, velocity: np.ndarray, wavelet: np.ndarray, source: tuple[int, int],
                receivers: tuple[np.ndarray, np.ndarray], record_wavefield: bool=False):
        """Returns (data[nt, nrec], wavefield[nt, nz, nx] or None)."""
        v = np.asarray(velocity, float)
        nt = wavelet.size
        v2dt2 = (v * self.dt_s) ** 2
        dt2 = self.dt_s ** 2
        p_prev = np.zeros((self.nz, self.nx))
        p_curr = np.zeros((self.nz, self.nx))
        lap = np.zeros((self.nz, self.nx))
        data = np.zeros((nt, receivers[0].size))
        wavefield = np.empty((nt, self.nz, self.nx)) if record_wavefield else None
        for n in range(nt):
            if record_wavefield:
                wavefield[n] = p_curr
            data[n] = p_curr[receivers]
            self._laplacian(p_curr, lap)
            p_next = 2.0 * p_curr - p_prev + v2dt2 * lap
            p_next[source] += dt2 * wavelet[n]
            if self.free_surface:
                p_next[0, :] = 0.0
            p_next *= self.damp
            p_curr *= self.damp
            p_prev, p_curr = p_curr, p_next
        return data, wavefield

    def adjoint_gradient(self, velocity: np.ndarray, residual: np.ndarray,
                         receivers: tuple[np.ndarray, np.ndarray], wavefield: np.ndarray) -> np.ndarray:
        """Gradient of the L2 misfit w.r.t. velocity for one shot.

        residual = synthetic - observed at the receivers, wavefield is the stored
        forward field. With q the adjoint field back-propagated from the
        residuals, dJ/dv = 2 * v * integral(q * lap(p)) dt. The adjoint of the
        forward operator (2q - q_prev + v^2 dt^2 lap(q)) is
        (2q - q_next + lap(v^2 dt^2 q)): v^2 and lap do not commute, so the
        velocity product must be applied before the Laplacian.
        """
        v = np.asarray(velocity, float)
        nt = wavefield.shape[0]
        v2dt2 = (v * self.dt_s) ** 2
        dt2 = self.dt_s ** 2
        q_next = np.zeros((self.nz, self.nx))
        q_curr = np.zeros((self.nz, self.nx))
        lap = np.zeros((self.nz, self.nx))
        lap_p = np.zeros((self.nz, self.nx))
        grad = np.zeros((self.nz, self.nx))
        for n in range(nt - 1, 0, -1):
            self._laplacian(v2dt2 * q_curr, lap)
            q_back = 2.0 * q_curr - q_next + lap
            q_back[receivers] += dt2 * residual[n]
            if self.free_surface:
                q_back[0, :] = 0.0
            q_back *= self.damp
            q_curr *= self.damp
            self._laplacian(wavefield[n], lap_p)
            grad += q_curr * lap_p
            q_next, q_curr = q_curr, q_back
        grad *= self.dt_s
        return 2.0 * v * grad

    def gradient_mask(self) -> np.ndarray:
        """False inside the sponge frame and the two top rows (gradient kept 0 there)."""
        mask = np.ones((self.nz, self.nx), bool)
        nb = self.sponge_cells
        if nb > 0:
            mask[:, :nb] = False
            mask[:, self.nx - nb:] = False
            mask[self.nz - nb:, :] = False
            if not self.free_surface:
                mask[:nb, :] = False
        mask[:2, :] = False
        return mask
