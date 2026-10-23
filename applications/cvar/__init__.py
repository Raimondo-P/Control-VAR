"""cvar: the Control VAR of Pala, 'Control VAR: a counterfactual based approach to inference in
macroeconomics'. Estimation, residual bootstrap that re-estimates the weights, response of the
controls, sensitivity bounds, and the test of the design with its sieve bootstrap."""
from .model import CVAR, CVARResult, BootstrapResult, bounds
from .designtest import design_test, DesignTestResult, ERS_CV_CONSTANT, po_statistic
from .panel import joint_bootstrap, joint_controls_response
from .diagnostics import lag_diagnostics
from .weights import rank_step, dols, fit_weights, supf_mean_break
from .var import var_fit, irf, var_sim, ice_from
from .utils import lag_rule, lagmat, band

__version__ = "0.1.0"
