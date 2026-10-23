"""The Control VAR estimator (Section 4), its residual bootstrap that re-estimates the weights
(Section 6), the response of the controls (Proposition 1), the sensitivity bounds (Proposition 2)
and the test of the design (Section 5)."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .utils import as2d, band
from .weights import rank_step, fit_weights, dols
from scipy.linalg import null_space
from .var import var_fit, irf, var_sim, ice_from
from .designtest import design_test
from .diagnostics import lag_diagnostics


class CVAR:
    """Control VAR.

    Parameters
    p : lag length of the VAR in (other policies, policy, cycles). The paper uses 12 for monthly
        data (Section 4, 'Lag length').
    q : current value and q lags of every policy in the weights regression.
    L : leads and lags of the changes in the trend directions of the controls.
    constrained : non-negative weights.
    rank : 'johansen' (trace test at rank_level), 'bic' (Cheng-Phillips type criterion) or an
        integer number of cointegrating relations among the controls.
    rank_lags : lags in differences of the Johansen regression, 'aic' or an integer.
    breaks : None, 'auto' (sup-Wald tests on the cycle net of the policy terms) or a list of
        observation indices of level shifts. With several outcomes, a list with one entry per outcome.
    """

    def __init__(self, p=12, q=12, L=1, constrained=True, rank="johansen", rank_level=0.05,
                 rank_lags="aic", breaks=None, max_breaks=2, break_cv=8.85):
        self.p, self.q, self.L = p, q, L
        self.constrained = constrained
        self.rank, self.rank_level, self.rank_lags = rank, rank_level, rank_lags
        self.breaks, self.max_breaks, self.break_cv = breaks, max_breaks, break_cv

    def fit(self, y, controls, policy, other=None, var_exog=None, names=None, control_names=None,
            dates=None, weights=None, difference=False):
        """y: (T,) or (T, m) treated outcomes in levels. controls: (T, K) array shared by all
        outcomes, or a list of m arrays, one per outcome. policy: (T,) policy of the treated unit.
        other: (T, nr) other policies, ordered before the policy (None if absent).
        var_exog: (T, ne) deterministic regressors of the VAR (e.g. a crisis dummy), held fixed in
        the bootstrap.
        weights: None (estimate them), or imposed weights: a (K,) array for one outcome, or a list
        with one entry per outcome (an array, or None to estimate that outcome's weights).
        difference: False, True, or a list of booleans per outcome. A differenced outcome enters the
        VAR as the change of its cycle and its responses are cumulated (extension for a treated unit
        with a trend of its own; requires imposed weights for that outcome).
        Returns a CVARResult."""
        Y = as2d(y)
        T, m = Y.shape
        Ctr = controls if isinstance(controls, (list, tuple)) else [controls] * m
        Ctr = [as2d(c) for c in Ctr]
        if len(Ctr) != m:
            raise ValueError("one control matrix per outcome")
        W = np.asarray(policy, dtype=float).reshape(T)
        R = as2d(other, T)
        pol = np.column_stack([R, W])
        brk = self.breaks if (m > 1 and isinstance(self.breaks, (list, tuple))) else [self.breaks] * m
        wl = weights if (isinstance(weights, (list, tuple)) and (m > 1 or weights is None or
                                                                  np.ndim(weights[0]) > 0 or weights[0] is None)) \
            else [weights] * m
        dfl = list(difference) if isinstance(difference, (list, tuple)) else [bool(difference)] * m
        units = []
        for j in range(m):
            if wl[j] is not None:
                w = np.asarray(wl[j], dtype=float).reshape(-1)
                units.append(dict(a=0.0, b=w, shift=np.zeros(0), steps=np.zeros((T, 0)), breaks=[],
                                  cycle=Y[:, j] - Ctr[j] @ w, supf_before=np.nan, supf_after=np.nan,
                                  rank=None, D1=None, imposed=True))
                continue
            if dfl[j]:
                raise ValueError("a differenced outcome needs imposed weights")
            mr, D1 = rank_step(Ctr[j], self.rank, self.rank_level, self.rank_lags)
            fw = fit_weights(Y[:, j], Ctr[j], pol, D1, self.q, self.L, self.constrained,
                             brk[j], self.max_breaks, self.break_cv)
            fw.update(rank=mr, D1=D1, imposed=False)
            units.append(fw)
        off = 1 if any(dfl) else 0
        tr = lambda M: _transform(M, dfl, off)
        G = tr(np.column_stack([u["cycle"] for u in units]))
        Z = np.column_stack([R[off:], W[off:], G])
        ve = None if var_exog is None else as2d(var_exog)[off:]
        A, U = var_fit(Z, self.p, ve)
        names = names or [f"y{j}" for j in range(m)]
        cn = control_names or [[f"c{k}" for k in range(c.shape[1])] for c in Ctr]
        if cn and not isinstance(cn[0], (list, tuple)):
            cn = [list(cn)] * m
        return CVARResult(self, Y, Ctr, R, W, pol, units, Z, A, U, ve, names, cn, dates, dfl, off)


def _transform(M, dfl, off):
    """Differenced columns become changes, the others stay in levels; the first `off` rows drop."""
    M = np.asarray(M, dtype=float)
    out = np.empty((M.shape[0] - off, M.shape[1]))
    for j in range(M.shape[1]):
        out[:, j] = np.diff(M[:, j]) if dfl[j] else M[off:, j]
    return out


def _cum(Rb, dfl):
    """Cumulate the responses of the differenced columns (level responses of the cycle)."""
    Rb = np.array(Rb, dtype=float)
    for j, d in enumerate(dfl):
        if d:
            Rb[:, j] = np.cumsum(Rb[:, j])
    return Rb


@dataclass
class CVARResult:
    model: CVAR
    Y: np.ndarray
    controls: list
    R: np.ndarray
    W: np.ndarray
    pol: np.ndarray
    units: list
    Z: np.ndarray
    A: np.ndarray
    U: np.ndarray
    var_exog: np.ndarray
    names: list
    control_names: list
    dates: object = None
    diff: list = None
    off: int = 0
    _nc: tuple = field(default=None, repr=False)
    _cr: tuple = field(default=None, repr=False)

    # ------------------------------------------------------------ basic objects
    @property
    def nR(self):
        return self.R.shape[1]

    @property
    def m(self):
        return self.Y.shape[1]

    @property
    def T(self):
        return self.Y.shape[0]

    def weights(self):
        """Weights, rank and level shifts of each outcome as a DataFrame."""
        rows = []
        for j, u in enumerate(self.units):
            for k, (c, b) in enumerate(zip(self.control_names[j], u["b"])):
                rows.append(dict(outcome=self.names[j], control=c, weight=b))
        return pd.DataFrame(rows)

    def summary(self):
        rows = []
        for j, u in enumerate(self.units):
            bd = u["breaks"]
            if self.dates is not None:
                bd = [str(pd.Timestamp(self.dates[k]).date())[:7] for k in bd]
            rows.append(dict(outcome=self.names[j], coint_relations_among_controls=u["rank"],
                             intercept=u["a"],
                             weights=", ".join(f"{c} {b:.2f}" for c, b in sorted(
                                 zip(self.control_names[j], u["b"]), key=lambda z: -z[1]) if b > 0.005),
                             level_shifts=" ".join(map(str, bd)),
                             shift_size=" ".join(f"{v:.2f}" for v in u["shift"]),
                             supF_before=u["supf_before"], supF_after=u["supf_after"]))
        return pd.DataFrame(rows)

    def cycle(self):
        return self.Z[:, self.nR + 1:]

    # ------------------------------------------------------------ point estimates
    def irf(self, H=36):
        """Response of the cycles (gamma_h), of the policy to its own innovation (psi_h) and the
        impulse causal effects (Theorem 4(a)) to a unit innovation in the policy."""
        p = self.model.p
        Rr = irf(self.A, self.U, p, self.nR, H)
        g = _cum(Rr[:, self.nR + 1:], self.diff)
        psi = Rr[:, self.nR]
        return dict(cycle=g, policy=psi, ice=ice_from(g, psi), all=Rr)

    def _nc_fit(self):
        if self._nc is None:
            o = self.off
            Zv = np.column_stack([self.R[o:], self.W[o:], _transform(self.Y, self.diff, o)])
            A, U = var_fit(Zv, self.model.p, self.var_exog)
            self._nc = (Zv, A, U)
        return self._nc

    def irf_no_controls(self, H=36):
        """Response of the treated outcomes in a recursive VAR without controls in (R, W, Y)."""
        Zv, A, U = self._nc_fit()
        return _cum(irf(A, U, self.model.p, self.nR, H)[:, self.nR + 1:], self.diff)

    # ------------------------------------------------------------ bootstrap
    def _draw(self, idx, H, no_controls=True):
        """One draw of the residual bootstrap that re-estimates the weights (Section 6, steps
        i to iv). idx: residual rows to use (length T - p)."""
        md, p, T = self.model, self.model.p, self.Z.shape[0]
        Zs = var_sim(self.A, self.U, self.Z[:p], p, T, idx, self.var_exog)
        pols = Zs[:, :self.nR + 1]
        Gs = []
        for j, u in enumerate(self.units):
            if u["imposed"]:
                Gs.append(Zs[:, self.nR + 1 + j])
                continue
            ys = u["a"] + self.controls[j] @ u["b"] + u["steps"] @ u["shift"] + Zs[:, self.nR + 1 + j]
            st = u["steps"]
            a2, b2, sh2 = dols(ys, self.controls[j], pols, u["D1"], exog=st if st.size else None,
                               q=md.q, L=md.L, constrained=md.constrained)
            Gs.append(ys - a2 - self.controls[j] @ b2 - (st @ sh2 if st.size else 0.0))
        A2, U2 = var_fit(np.column_stack([pols] + Gs), p, self.var_exog)
        Rr = irf(A2, U2, p, self.nR, H)
        g, psi = _cum(Rr[:, self.nR + 1:], self.diff), Rr[:, self.nR]
        out = dict(cycle=g, policy=psi, ice=ice_from(g, psi))
        if no_controls:
            Zv, Av, Uv = self._nc_fit()
            Zvs = var_sim(Av, Uv, Zv[:p], p, T, idx, self.var_exog)
            A3, U3 = var_fit(Zvs, p, self.var_exog)
            out["no_controls"] = _cum(irf(A3, U3, p, self.nR, H)[:, self.nR + 1:], self.diff)
        return out

    def _jt_fit(self):
        """VAR in J_t = (change in the trend directions of the controls, stationary combinations of
        the controls, policies, cycle), for the joint bootstrap of Theorem thm:joint."""
        if getattr(self, "_jt", None) is None:
            u = self.units[0]
            Y0 = self.controls[0]
            K = Y0.shape[1]
            D1 = u["D1"]
            H = null_space(D1.T) if D1.shape[1] < K else np.zeros((K, 0))
            Dfull = np.hstack([D1, H])
            r = D1.shape[1]
            Xr = Y0 @ Dfull
            G = self.Z[:, self.nR + 1]
            Jm = np.column_stack([np.diff(Xr[:, :r], axis=0), Xr[1:, r:], self.pol[1:], G[1:]])
            veJ = None if self.var_exog is None else self.var_exog[1:]
            A, U = var_fit(Jm, self.model.p, veJ)
            self._jt = (Dfull, np.linalg.inv(Dfull), Xr, r, K, Jm, A, U, veJ)
        return self._jt

    def _joint_from_path(self, Js, H, use_steps=True):
        """Rebuild the controls and the treated outcome from a path of the joint VAR, re-estimate the
        weights and the VAR in (R, W, G), and return the responses."""
        md, p = self.model, self.model.p
        Dfull, Dinv, Xr, r, K, Jm, A, U, veJ = self._jt_fit()
        X1s = np.vstack([Xr[:1, :r], Xr[0, :r] + np.cumsum(Js[:, :r], axis=0)])
        Ss = np.vstack([Xr[:1, r:], Js[:, r:K]])
        Y0s = np.column_stack([X1s, Ss]) @ Dinv
        npol = self.pol.shape[1]
        pols = np.vstack([self.pol[:1], Js[:, K:K + npol]])
        G = self.Z[:, self.nR + 1]
        Gs = np.r_[G[0], Js[:, K + npol]]
        u = self.units[0]
        st = u["steps"] if (use_steps and u["steps"].size) else None
        ys = u["a"] + Y0s @ u["b"] + (st @ u["shift"] if st is not None else 0.0) + Gs
        a2, b2, sh2 = dols(ys, Y0s, pols, u["D1"], exog=st, q=md.q, L=md.L, constrained=md.constrained)
        Gh = ys - a2 - Y0s @ b2 - (st @ sh2 if st is not None else 0.0)
        ve = self.var_exog if use_steps else None
        A2, U2 = var_fit(np.column_stack([pols, Gh]), p, ve)
        Rr = irf(A2, U2, p, self.nR, H)
        g, psi = Rr[:, self.nR + 1:], Rr[:, self.nR]
        return dict(cycle=g, policy=psi, ice=ice_from(g, psi))

    def _draw_joint(self, idx, H):
        """One draw of the joint bootstrap: regenerate the controls with the policies and the cycle,
        rebuild the treated outcome, re-estimate the weights, re-estimate the VAR in (R, W, G)."""
        p = self.model.p
        Dfull, Dinv, Xr, r, K, Jm, A, U, veJ = self._jt_fit()
        Js = var_sim(A, U, Jm[:p], p, Jm.shape[0], idx, veJ)
        return self._joint_from_path(Js, H)

    def _joint_population(self, H, mult=50, seed=None):
        """Bootstrap population value of the responses: the estimator applied to one long path of the
        fitted joint VAR (mult times the sample length). Deterministic regressors and level shifts,
        which are tied to calendar dates, are left out of the long path."""
        p = self.model.p
        Dfull, Dinv, Xr, r, K, Jm, A, U, veJ = self._jt_fit()
        n = mult * Jm.shape[0]
        rng = np.random.default_rng(seed)
        A0 = A[:1 + Jm.shape[1] * p]                 # drop the coefficients on deterministic regressors
        Js = var_sim(A0, U, Jm[:p], p, n, rng.integers(0, U.shape[0], n - p), None)
        return self._joint_from_path(Js, H, use_steps=False)

    def _nc_draw(self, idx, H):
        p, T = self.model.p, self.Z.shape[0]
        Zv, Av, Uv = self._nc_fit()
        Zvs = var_sim(Av, Uv, Zv[:p], p, T, idx, self.var_exog)
        A3, U3 = var_fit(Zvs, p, self.var_exog)
        return _cum(irf(A3, U3, p, self.nR, H)[:, self.nR + 1:], self.diff)

    def bootstrap(self, B=999, H=36, seed=None, no_controls=True, indices=None, scheme="auto",
                  recentre=True, recentre_mult=50):
        """Residual bootstrap with percentile bands.

        scheme:
          'auto'        'joint' for a single outcome with estimated weights in levels, 'conditional'
                        otherwise (imposed weights, differenced branch, several outcomes).
          'joint'       Theorem thm:joint: a VAR in (change in the trend directions of the controls,
                        stationary combinations of the controls, policies, cycle) is resampled, the
                        controls and the treated outcome are rebuilt and the weights re-estimated.
                        The rank, the trend directions and the dates of level shifts are held fixed.
                        With recentre=True (Theorem thm:joint), the draws are shifted by the estimate
                        minus the bootstrap population value of the response, computed by applying the
                        estimator to one path of the fitted joint VAR recentre_mult times as long as
                        the sample (the recentring principle of Hall and Horowitz 1996).
          'conditional' the VAR in (policies, cycle) is resampled with the controls held at their
                        observed values; with estimated weights they are re-estimated in every draw.
                        Valid with known weights (Theorem thm:resid); with estimated weights only
                        when the first-step term J_h is zero (Proposition prop:J).
        indices: optional (B, T - p) residual rows for the conditional scheme (joint bootstrap
        across units in panel.py)."""
        rng = np.random.default_rng(seed)
        est = [not u["imposed"] for u in self.units]
        if scheme == "auto":
            scheme = "joint" if (self.m == 1 and est[0] and self.off == 0) else "conditional"
        n = self.U.shape[0]
        if scheme == "joint":
            if self.m != 1 or not est[0] or self.off != 0:
                raise ValueError("the joint scheme is for a single outcome with estimated weights in levels")
            if indices is not None:
                raise ValueError("indices are for the conditional scheme")
            nj = self._jt_fit()[7].shape[0]
            draws = []
            for _ in range(B):
                d = self._draw_joint(rng.integers(0, nj, nj), H)
                if no_controls:
                    d["no_controls"] = self._nc_draw(rng.integers(0, n, n), H)
                draws.append(d)
        else:
            if indices is None:
                indices = rng.integers(0, n, (B, n))
            draws = [self._draw(ix, H, no_controls) for ix in indices]
        point = self.irf(H)
        pt = dict(cycle=point["cycle"], policy=point["policy"], ice=point["ice"])
        pop = None
        if scheme == "joint" and recentre:
            pop = self._joint_population(H, recentre_mult, None if seed is None else seed + 99991)
            for d in draws:
                for k in ("cycle", "policy", "ice"):
                    d[k] = d[k] + (pt[k] - pop[k])
        if no_controls:
            pt["no_controls"] = self.irf_no_controls(H)
        out = BootstrapResult(pt, {k: np.array([d[k] for d in draws]) for k in draws[0]}, self.names)
        out.scheme = scheme
        out.population = pop
        return out

    # ------------------------------------------------------------ response of the controls
    def _cr_fit(self):
        if self._cr is None:
            S = np.column_stack([self.controls[j] @ u["b"] + u["steps"] @ u["shift"]
                                 for j, u in enumerate(self.units)])
            o = self.off
            Za = np.column_stack([self.R[o:], self.W[o:], _transform(self.Y, self.diff, o),
                                  _transform(S, self.diff, o)])
            A, U = var_fit(Za, self.model.p, self.var_exog)
            self._cr = (Za, A, U)
        return self._cr

    def controls_response(self, H=36, B=499, seed=None, indices=None):
        """Proposition 1: recursive VAR in (R, W, Y, S) with S = b'Y0 the weighted controls.
        Returns the responses of Y (the VAR with the controls added), of S (the common-trend bias)
        and of G = Y - S, with their bootstrap draws (weights held at their estimates)."""
        Za, A, U = self._cr_fit()
        p, m, nR = self.model.p, self.m, self.nR
        Rr = irf(A, U, p, nR, H)
        pt = dict(treated=_cum(Rr[:, nR + 1:nR + 1 + m], self.diff),
                  controls=_cum(Rr[:, nR + 1 + m:], self.diff))
        pt["difference"] = pt["treated"] - pt["controls"]
        rng = np.random.default_rng(seed)
        n = U.shape[0]
        if indices is None:
            indices = rng.integers(0, n, (B, n))
        dr = {k: [] for k in pt}
        for ix in indices:
            A2, U2 = var_fit(var_sim(A, U, Za[:p], p, Za.shape[0], ix, self.var_exog), p, self.var_exog)
            R2 = irf(A2, U2, p, nR, H)
            t, c = _cum(R2[:, nR + 1:nR + 1 + m], self.diff), _cum(R2[:, nR + 1 + m:], self.diff)
            dr["treated"].append(t); dr["controls"].append(c); dr["difference"].append(t - c)
        return BootstrapResult(pt, {k: np.array(v) for k, v in dr.items()}, self.names)

    # ------------------------------------------------------------ test and diagnostics
    def design_test(self, B=999, seed=None, cbar=-7.0, p_sieve="rule", pmax=12):
        """Test of Section 5 for each outcome, with the same controls, policy terms (current value
        and q lags) and level shifts as the weights regression."""
        out = []
        for j, u in enumerate(self.units):
            out.append(design_test(self.Y[:, j], self.controls[j], self.pol, q=self.model.q,
                                   exog=u["steps"] if u["steps"].size else None, cbar=cbar, B=B,
                                   p_sieve=p_sieve, pmax=pmax,
                                   seed=None if seed is None else seed + j))
        return out

    def lag_diagnostics(self, lags=(1, 2, 3, 4, 6, 8, 12), extra=12):
        names = [f"R{i}" for i in range(self.nR)] + ["policy"] + [f"G_{n}" for n in self.names]
        return lag_diagnostics(self.Z, lags, self.var_exog, extra, names)


@dataclass
class BootstrapResult:
    point: dict
    draws: dict
    names: list

    def bands(self, key="cycle", alpha=0.10, method="percentile"):
        return band(self.draws[key], alpha, self.point[key], method)

    def table(self, horizons=(0, 6, 12, 24), alpha=0.10, keys=None, method="percentile"):
        """Long table of point estimates and bands at selected horizons."""
        keys = keys or list(self.point)
        rows = []
        for k in keys:
            pt = np.asarray(self.point[k])
            lo, hi = self.bands(k, alpha, method)
            pt2 = pt.reshape(pt.shape[0], -1)
            lo2, hi2 = lo.reshape(pt2.shape), hi.reshape(pt2.shape)
            for j in range(pt2.shape[1]):
                nm = self.names[j] if pt.ndim > 1 else "policy"
                for h in horizons:
                    rows.append(dict(object=k, outcome=nm, h=h, estimate=pt2[h, j],
                                     lower=lo2[h, j], upper=hi2[h, j]))
        return pd.DataFrame(rows)


def bounds(cr, lams=(0.0, 0.25, 0.5), horizons=(6, 12), alpha=0.10, grid=None):
    """Proposition 2 from a controls_response result: bands for the bounds
    [gamma^G - lam |gamma^S|, gamma^G + lam |gamma^S|], i.e. the alpha/2 quantile of the lower end
    and the 1 - alpha/2 quantile of the upper end over the bootstrap draws, and the smallest lam on
    `grid` at which the band contains the response of the treated outcome in the same VAR."""
    grid = np.round(np.arange(0, 3.001, 0.01), 2) if grid is None else np.asarray(grid)
    G, S, Yd = cr.draws["difference"], cr.draws["controls"], cr.point["treated"]
    rows = []
    for j, nm in enumerate(cr.names):
        for h in horizons:
            g, s = G[:, h, j], np.abs(S[:, h, j])

            def bd(l):
                return np.quantile(g - l * s, alpha / 2), np.quantile(g + l * s, 1 - alpha / 2)

            row = dict(outcome=nm, h=h, treated_response=Yd[h, j],
                       difference=cr.point["difference"][h, j], controls=cr.point["controls"][h, j])
            for l in lams:
                row[f"lam{l:g}_lower"], row[f"lam{l:g}_upper"] = bd(l)
            target = Yd[h, j]
            hit = [l for l in grid if bd(l)[0] <= target <= bd(l)[1]]
            row["breakdown_lambda"] = hit[0] if hit else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
