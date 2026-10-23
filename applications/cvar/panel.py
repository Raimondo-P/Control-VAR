"""Several treated units that share controls: a joint bootstrap that draws the same time periods
of the residuals for every unit, so that the bands of the average reflect the dependence across
units (replaces the independent-draws average)."""
import numpy as np

from .model import BootstrapResult


def _check(results):
    n = {r.U.shape[0] for r in results}
    if len(n) != 1 or len({r.model.p for r in results}) != 1:
        raise ValueError("joint bootstrap requires the same sample and lag length for every unit")
    return n.pop()


def joint_bootstrap(results, B=999, H=36, seed=None, no_controls=True, labels=None):
    """Returns (per-unit BootstrapResults, BootstrapResult of the average across units). Each draw
    uses the same residual rows (time periods) for all units, for the CVAR and for the VAR
    without controls. Units with several outcomes contribute each outcome to the average."""
    n = _check(results)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    per = [r.bootstrap(B, H, no_controls=no_controls, indices=idx, scheme="conditional") for r in results]
    keys = list(per[0].point)
    pt = {k: np.mean([np.asarray(b.point[k]).reshape(H + 1, -1).mean(1) for b in per], 0)[:, None]
          for k in keys}
    dr = {k: np.mean([b.draws[k].reshape(B, H + 1, -1).mean(2) for b in per], 0)[:, :, None]
          for k in keys}
    return per, BootstrapResult(pt, dr, ["Average"])


def joint_controls_response(results, B=499, H=36, seed=None):
    """Proposition 1 for every unit with common residual draws, and the average across units."""
    n = _check(results)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    per = [r.controls_response(H, B, indices=idx) for r in results]
    keys = list(per[0].point)
    pt = {k: np.mean([b.point[k].mean(1) for b in per], 0)[:, None] for k in keys}
    dr = {k: np.mean([b.draws[k].mean(2) for b in per], 0)[:, :, None] for k in keys}
    return per, BootstrapResult(pt, dr, ["Average"])
