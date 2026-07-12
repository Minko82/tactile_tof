"""
tof_sensor.py — THE filter for the 8x8 ToF sensor (proximity + tactile in one).

Import this; the plot_*.py scripts each draw one graph on top of it.

    from tof_sensor import ToFSensor, fit_zone_calibration, load_round, load_raw

Pipeline, per frame, causal / real-time:
    raw 8x8 -> per-zone degree-4 calibration -> per-zone IMM Kalman bank (denoise)
            -> PROXIMITY: fuse the agreeing zones + scalar filter -> one distance
               TACTILE:   zones pushed below rest -> deformation field -> normal/shear

ToFSensor.update(frame, t) -> dict(field_mm, proximity_mm, deformation_mm,
                                   contact, normal, shear, mode, ...)
Calibration is fit OFFLINE once (fit_zone_calibration / ToFSensor.from_data).
"""
import csv, os
import numpy as np

TABLE_Z_MM = -143.30
CENTRAL    = (27, 28, 35, 36)          # central 2x2 zones (boresight)
CAL_DEGREE = 4                         # per-zone poly (close-range accuracy)
Q_LO, Q_HI, P_STAY = 50.0, 150000.0, 0.97     # IMM smooth + agile modes
R_FLOOR    = 4.0                       # min per-zone measurement variance (mm^2)
CONTACT_MM, MAD_K, MIN_CONTACT = 4.0, 4.0, 3  # touch = pushed >=4mm below rest, >=3 zones
ZONE_PITCH_MM = 2.5                    # zone spacing at the surface (for shear in mm)


# --------------------------------------------------------- offline calibration fit
def fit_zone_calibration(frames_list, gt_list, degree=CAL_DEGREE, min_samples=200, std_cut=6.0):
    """Per-zone poly raw->mm vs ground truth. -> coeffs (64,deg+1, nan=untrusted), var (64,)."""
    raw = np.concatenate(frames_list); gt = np.concatenate(gt_list)
    coeffs = np.full((raw.shape[1], degree + 1), np.nan); var = np.full(raw.shape[1], np.nan)
    for j in range(raw.shape[1]):
        v = np.isfinite(raw[:, j])
        if v.sum() < min_samples or np.ptp(raw[v, j]) < 1e-6:
            continue
        c = np.polyfit(raw[v, j], gt[v], degree)
        s = float(np.std(np.polyval(c, raw[v, j]) - gt[v]))
        if s <= std_cut:
            coeffs[j] = c; var[j] = s * s
    return coeffs, var


# --------------------------------------------------------------- per-zone IMM (stream)
class _IMMZone:
    __slots__ = ("r", "x", "P", "mu")

    def __init__(self, r):
        self.r = float(r); self.x = None

    def step(self, z, dt):
        if self.x is None:
            self.x = [np.array([z, 0.0]), np.array([z, 0.0])]
            self.P = [np.eye(2) * 1e3, np.eye(2) * 1e3]; self.mu = np.array([0.5, 0.5])
            return z
        qa = (Q_LO, Q_HI); Pi = np.array([[P_STAY, 1-P_STAY], [1-P_STAY, P_STAY]])
        H = np.array([[1.0, 0.0]]); I = np.eye(2); F = np.array([[1.0, dt], [0.0, 1.0]])
        cbar = Pi.T @ self.mu; xm = [np.zeros(2), np.zeros(2)]; Pm = [np.zeros((2, 2)), np.zeros((2, 2))]
        for j in range(2):
            for i in range(2): xm[j] = xm[j] + (Pi[i, j]*self.mu[i]/cbar[j]) * self.x[i]
            for i in range(2):
                d = self.x[i] - xm[j]; Pm[j] = Pm[j] + (Pi[i, j]*self.mu[i]/cbar[j]) * (self.P[i] + np.outer(d, d))
        L = np.zeros(2)
        for j in range(2):
            Q = qa[j] * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
            xp = F @ xm[j]; Pp = F @ Pm[j] @ F.T + Q
            y = z - (H @ xp)[0]; S = (H @ Pp @ H.T)[0, 0] + self.r; K = (Pp @ H.T).flatten() / S
            self.x[j] = xp + K*y; self.P[j] = (I - np.outer(K, H)) @ Pp
            L[j] = np.exp(-0.5*y*y/S) / np.sqrt(2*np.pi*S) + 1e-300
        self.mu = cbar * L; self.mu = self.mu / self.mu.sum()
        return float(self.mu[0]*self.x[0][0] + self.mu[1]*self.x[1][0])


# ------------------------------------------------------------------ the unified sensor
class ToFSensor:
    """Real-time 8x8 ToF filter -> {proximity distance, tactile field + force}."""

    def __init__(self, coeffs=None, var=None, baseline=None,
                 contact_mm=CONTACT_MM, mad_k=MAD_K, min_contact=MIN_CONTACT):
        self.coeffs = coeffs
        self.var = var if var is not None else np.full(64, R_FLOOR)
        self.trusted = np.isfinite(self.var) if coeffs is not None else np.ones(64, bool)
        self.filters = [_IMMZone(max(self.var[j] if np.isfinite(self.var[j]) else R_FLOOR, R_FLOOR))
                        for j in range(64)]
        self.baseline = None if baseline is None else np.asarray(baseline, float)
        self.contact_mm, self.mad_k, self.min_contact = contact_mm, mad_k, min_contact
        self.prox_filter = _IMMZone(1.0)
        self._t_prev = None; self._last = np.full(64, np.nan); self._c0 = None; self._nc = 0

    @classmethod
    def from_data(cls, frames_list, gt_list, **kw):
        coeffs, var = fit_zone_calibration(frames_list, gt_list)
        return cls(coeffs, var, **kw)

    def _calibrate(self, frame):
        f = np.asarray(frame, float); f[f <= 0] = np.nan
        if self.coeffs is None:
            return f
        out = np.full(64, np.nan)
        for j in np.where(self.trusted & np.isfinite(f))[0]:
            out[j] = np.polyval(self.coeffs[j], f[j])
        return out

    def update(self, frame, t):
        cal = self._calibrate(frame)
        dt = 1e-3 if self._t_prev is None else max(t - self._t_prev, 1e-3)
        self._t_prev = t

        filt = np.empty(64)
        for j in range(64):
            if np.isfinite(cal[j]):
                filt[j] = self.filters[j].step(float(cal[j]), dt); self._last[j] = filt[j]
            else:
                filt[j] = self._last[j]
        valid = np.isfinite(filt) & self.trusted

        fv = filt[valid]; med = np.median(fv) if fv.size else np.nan
        if self.baseline is not None:
            contact = valid & (filt < self.baseline - self.contact_mm)
        else:
            mad = 1.4826 * np.median(np.abs(fv - med)) if fv.size else 0.0
            contact = valid & (filt < med - max(self.mad_k * mad, self.contact_mm))
        ncontact = int(contact.sum()); is_tactile = ncontact >= self.min_contact

        # PROXIMITY: fuse raw-calibrated agreeing zones (spatial denoise) + scalar filter
        prox = prox_std = np.nan
        pz = valid & ~contact & np.isfinite(cal)
        if pz.sum() >= 3:
            w = 1.0 / np.clip(self.var[pz], R_FLOOR, None); d = cal[pz]
            m = np.median(d); keep = np.abs(d - m) <= max(3 * 1.4826 * np.median(np.abs(d - m)), 3.0)
            w, d = w[keep], d[keep]
            prox = self.prox_filter.step(float(np.sum(w * d) / np.sum(w)), dt)
            prox_std = float(np.sqrt(1.0 / np.sum(w)))

        # TACTILE: deformation of contact zones vs the reference (baseline or field median)
        ref = self.baseline if self.baseline is not None else np.full(64, med)
        deform = np.clip(ref - filt, 0.0, None); deform[~valid] = 0.0
        normal = float(deform[contact].sum()) if is_tactile else 0.0
        peak = float(deform[contact].max()) if is_tactile else 0.0
        centroid = np.array([np.nan, np.nan]); shear = np.array([0.0, 0.0])
        if is_tactile:
            g = np.zeros(64); g[contact] = deform[contact]; g = g.reshape(8, 8)
            rr, cc = np.mgrid[0:8, 0:8]; wsum = g.sum()
            centroid = np.array([(g*rr).sum()/wsum, (g*cc).sum()/wsum])
            self._nc += 1
            if self._c0 is None and self._nc >= 3:
                self._c0 = centroid.copy()
            if self._c0 is not None:
                shear = (centroid - self._c0) * ZONE_PITCH_MM
        else:
            self._c0 = None; self._nc = 0

        return dict(field_mm=filt.reshape(8, 8), proximity_mm=prox, proximity_std_mm=prox_std,
                    deformation_mm=(deform * contact).reshape(8, 8), contact=contact.reshape(8, 8),
                    n_contact=ncontact, normal=normal, peak=peak, centroid=centroid,
                    shear=shear, shear_mag=float(np.hypot(*shear)),
                    mode="tactile" if is_tactile else "proximity")


# ---------------------------------------------------------------------- data loaders
def load_round(folder):
    """A3 round -> (t, frames n x 64 [invalid->nan], gt mm above table)."""
    with open(os.path.join(folder, "tof_log.csv")) as f:
        r = csv.reader(f); h = next(r); zc = [i for i, c in enumerate(h) if c.startswith("z")]
        ts, fr = [], []
        for row in r:
            if not row: continue
            ts.append(float(row[0])); fr.append([float(row[i]) for i in zc])
    F = np.array(fr); F[F <= 0] = np.nan
    with open(os.path.join(folder, "robot_log.csv")) as f:
        r = csv.reader(f); next(r); tr = [x for x in r if x]
    t = np.array(ts)
    gt = np.interp(t, [float(x[0]) for x in tr], [float(x[3]) - TABLE_Z_MM for x in tr])
    return t, F, gt


def load_raw(path):
    """A2 static recording -> (t, frames n x 64 [invalid->nan])."""
    ts, fr = [], []
    with open(path) as f:
        r = csv.reader(f); h = next(r); zc = [i for i, c in enumerate(h) if c.startswith("z")]
        for row in r:
            if not row: continue
            ts.append(float(row[0])); fr.append([float(row[i]) for i in zc])
    F = np.array(fr); F[F <= 0] = np.nan
    return np.array(ts), F


if __name__ == "__main__":                      # quick self-test
    import glob
    HERE = os.path.dirname(os.path.abspath(__file__))
    dirs = sorted(glob.glob(os.path.join(HERE, "test_data_A3", "round_*")))
    rounds = [load_round(d) for d in dirs]
    s = ToFSensor.from_data([r[1] for r in rounds[1:]], [r[2] for r in rounds[1:]])
    t, F, gt = rounds[0]
    p = np.array([s.update(F[k], t[k])["proximity_mm"] for k in range(len(t))])
    print(f"self-test: proximity RMSE {np.sqrt(np.nanmean((p-gt)**2)):.2f} mm vs UR5")
