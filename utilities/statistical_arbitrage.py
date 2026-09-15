"""
Statistical Arbitrage utilities based on Avellaneda and Lee (2008):
"Statistical Arbitrage in the U.S. Equities Market"

This module implements:
1. PCA-based factor model for stock returns
2. Ornstein-Uhlenbeck parameter estimation for residuals
3. S-score computation for mean-reversion trading signals
"""

from typing import Any

import numpy as np
import pandas as pd
from utilities.principal_component_analysis import principal_component_analysis


def compute_volume_adjusted_returns(
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    trailing_window: int = 60,
) -> pd.DataFrame:
    """Compute volume-adjusted ("trading time") returns.

    The idea is that price moves on low-volume days carry more information
    than moves on high-volume days. The adjusted return is:

        R_bar_{i,t} = R_{i,t} * <delta_V_i> / V_{i,t}

    where <delta_V_i> is the trailing average daily volume and V_{i,t} is
    the actual volume on day t. This amplifies low-volume moves and dampens
    high-volume moves.

    Args:
        returns: Daily returns (T x N).
        volume: Daily trading volume (T x N), must share index and columns with returns.
        trailing_window: Number of days for the trailing average volume.

    Returns:
        Volume-adjusted returns (T x N). Rows with insufficient volume history are NaN.
    """
    # Align volume to returns index and columns
    common_cols = returns.columns.intersection(volume.columns)
    common_idx = returns.index.intersection(volume.index)
    vol = volume.loc[common_idx, common_cols]
    ret = returns.loc[common_idx, common_cols]

    # Trailing average volume
    avg_volume = vol.rolling(window=trailing_window, min_periods=trailing_window).mean() 

    # Volume adjustment ratio: <δV> / V_t
    # Clip volume to avoid division by zero or extreme ratios
    vol_clipped = vol.clip(lower=1)
    adjustment = (avg_volume / vol_clipped)  

    # Cap extreme adjustments (e.g., when volume drops to near zero)
    adjustment = adjustment.clip(upper=10.0)

    return ret * adjustment


def estimate_factor_model(
    returns: pd.DataFrame,
    n_factors: int = 15,
) -> dict[str, Any]:
    """Estimate a PCA-based factor model for stock returns.

    The model decomposes returns as:
        R_i,t = alpha_i + sum_j(beta_i,j * F_j,t) + epsilon_i,t

    where F_j are the principal component factors extracted from the correlation matrix.

    Args:
        returns: Returns matrix (T x N), T = time periods, N = assets.
        n_factors: Number of principal component factors to use, default 15.

    Returns:
        Dictionary containing eigenvalues, eigenvectors, factors, betas, alphas,
        residuals, explained_variance, and n_factors.

    Raises:
        TypeError: If returns is not a DataFrame.
        ValueError: If returns is empty, has insufficient data, or invalid parameters.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"returns must be a DataFrame, got {type(returns)}")

    if returns.empty:
        raise ValueError("returns DataFrame is empty")

    if returns.isna().all().any():
        nan_cols = returns.columns[returns.isna().all()].tolist()
        raise ValueError(f"Columns {nan_cols} contain only NaN values")

    T, N = returns.shape

    if T < N:
        raise ValueError(
            f"Insufficient data: {T} observations for {N} assets. "
            f"Need at least N observations for PCA."
        )

    if n_factors > N:
        raise ValueError(
            f"n_factors ({n_factors}) cannot exceed number of assets ({N})"
        )

    if n_factors < 1:
        raise ValueError(f"n_factors must be at least 1, got {n_factors}")

    # Compute correlation matrix
    corr_matrix = returns.corr().values

    # Eigendecomposition
    eigenvalues, eigenvectors = principal_component_analysis(corr_matrix) 

    # Select top n_factors
    eigenvalues_selected = eigenvalues[:n_factors] 
    eigenvectors_selected = eigenvectors[:,:n_factors]

    # Factor returns
    std_vector = returns.std(axis=0)
    std_vector = std_vector.replace(0, np.nan)
  
    returns_std = returns / std_vector
    returns_std = returns_std.fillna(0)


    factors = returns_std.values @ eigenvectors_selected  # !!! COMPLETE AS APPROPRIATE !!!
    factors_df = pd.DataFrame(
        factors, index=returns.index, columns=[f"PC{i + 1}" for i in range(n_factors)]
    )

    # For each asset, regress returns on factors to get betas and alpha
    residuals_df, betas_df, alphas_series = estimate_ou_window_residuals(
        returns,
        factors_df,
    ).values()

    # Explained variance ratio
    explained_variance = eigenvalues_selected/sum(eigenvalues)  

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "factors": factors_df,
        "betas": betas_df,
        "alphas": alphas_series,
        "residuals": residuals_df,
        "explained_variance": explained_variance,
        "n_factors": n_factors,
    }


def estimate_ou_window_residuals(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    ou_window: int | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run OLS on the O-U estimation window and return cumulative residuals.

    Args:
        returns: Returns from the full estimation window (T x N).
        factors: PCA factor returns from the same full window (T x K).
        ou_window: Number of days in the O-U estimation sub-window. If None, use the full window.

    Returns:
        Dictionary with residuals, betas, and alphas for the O-U sub-window.
    """
    if ou_window is None:
        ou_window = len(returns)
    returns_ou = returns.iloc[-ou_window:]
    factors_ou = factors.iloc[-ou_window:]
    T_ou, N = returns_ou.shape
    n_factors = factors_ou.shape[1]

    betas = np.zeros((N, n_factors))
    alphas = np.zeros(N)
    residuals = np.zeros((T_ou, N))

    y = returns_ou.values
    X = np.column_stack([np.ones(T_ou), factors_ou.values])

    # Solve OLS for all assets simultaneously
    sol, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-15)

    # First row of the solution is the intercept (alpha) for each asset
    alphas = sol[0, :]

    # Remaining rows are the factor loadings (betas), transposed to shape (N, n_factors)
    betas = sol[1:, :].T

    # Pointwise residuals: y (T x N) - X (T x K+1) @ sol (K+1 x N)
    residuals = y - X @ sol

    residuals_df = pd.DataFrame(
        residuals, index=returns_ou.index, columns=returns_ou.columns
    )
    betas_df = pd.DataFrame(
        betas, index=returns.columns, columns=[f"PC{i + 1}" for i in range(n_factors)]
    )
    alphas_series = pd.Series(alphas, index=returns.columns)

    return {
        "residuals": residuals_df,
        "betas": betas_df,
        "alphas": alphas_series,
    }


def estimate_ou_parameters(
    residuals: pd.Series,
    dt: float = 1 / 252,
) -> dict[str, float]:
    """Estimate Ornstein-Uhlenbeck process parameters from cumulative residuals.

    The O-U process is: dX_t = kappa * (m - X_t) * dt + sigma * dW_t

    Estimated via discrete AR(1): X_{t+1} = a + b * X_t + epsilon_t
    where b = exp(-kappa*dt), a = m*(1-b).

    The original parameters are recovered as:
        kappa     = -log(b) / dt
        m         = a / (1 - b)
        sigma_eq  = sqrt(Var(epsilon) / (1 - b^2))

    Args:
        residuals: Cumulative residual time series X_t.
        dt: Time step in years, default 1/252 (daily).

    Returns:
        Dictionary with kappa, m, sigma, sigma_eq, half_life, a, b, var_epsilon.
    """
    X = residuals.values
    T = len(X)

    # AR(1) regression of X_{t+1} on X_t with intercept
    X_lag  = X[:-1]
    X_lead = X[1:]
    A = np.column_stack([np.ones(T - 1), X_lag])
    coeffs, *_ = np.linalg.lstsq(A, X_lead, rcond=None)

    a = coeffs[0]                          # AR(1) intercept
    b = coeffs[1]                          # AR(1) autoregressive coefficient
    epsilon     = X_lead - (a + b * X_lag) # AR(1) residuals
    var_epsilon = np.var(epsilon, ddof=2)  # residual variance

    # Guard: b must be in (0, 1) for a valid mean-reverting process.
    # b <= 0 implies negative autocorrelation (not O-U);
    # b >= 1 implies unit root or explosive process.
    # In both cases we return NaN for all derived parameters so the asset
    # is safely excluded by the kappa > 8.4 filter downstream.
    if b <= 0 or b >= 1:
        return {
            "kappa": np.nan, "m": np.nan, "sigma": np.nan,
            "sigma_eq": np.nan, "half_life": np.nan,
            "a": a, "b": b, "var_epsilon": var_epsilon,
        }

    kappa     = -np.log(b) / dt                                        # mean-reversion speed
    m         = a / (1 - b)                                            # long-run equilibrium level
    sigma     = np.sqrt(max(var_epsilon * 2 * kappa / (1 - b**2), 0)) # instantaneous volatility
    sigma_eq  = np.sqrt(max(var_epsilon / (1 - b**2), 0))             # equilibrium volatility
    half_life = np.log(2) / kappa                                      # mean-reversion half-life

    return {
        "kappa":       kappa,
        "m":           m,
        "sigma":       sigma,
        "sigma_eq":    sigma_eq,
        "half_life":   half_life,
        "a":           a,
        "b":           b,
        "var_epsilon": var_epsilon,
    }


def estimate_all_ou_parameters(
    cumulative_residuals: pd.DataFrame,
    dt: float = 1 / 252,
    center_ou_means: bool = True,
) -> dict[str, dict[str, float]]:
    """Estimate O-U parameters for all assets.

    Args:
        cumulative_residuals: Cumulative residuals (T x N).
        dt: Time step in years.
        center_ou_means: Whether to center the equilibrium means by subtracting the
            cross-sectional average (eq. 18 in the paper). This removes model bias
            and is consistent with market-neutrality.

    Returns:
        O-U parameters for each asset.
    """
    ou_params = {
        asset: estimate_ou_parameters(cumulative_residuals[asset].dropna(), dt=dt)
        for asset in cumulative_residuals.columns
    }

    if center_ou_means:
        # Collect only finite means to avoid NaN contaminating the average
        valid_means = [p["m"] for p in ou_params.values() if np.isfinite(p["m"])]
        if valid_means:
            mean_m = np.mean(valid_means)
            for asset in ou_params:
                if np.isfinite(ou_params[asset]["m"]):
                    ou_params[asset]["m"] -= mean_m

    return ou_params


def compute_s_score(
    cumulative_residuals: pd.DataFrame,
    ou_params: dict[str, dict[str, float]],
    modified: bool = False,
) -> pd.DataFrame:
    """Compute the s-score for each asset based on its deviation from equilibrium.

    The s-score measures how many equilibrium standard deviations the current
    residual is away from its mean.

    Standard:  s_i     = (X_i - m_i) / sigma_eq,i                    (eq. 15)
    Modified:  s_mod,i = s_i - alpha_i / (kappa_i * sigma_eq,i)      (eq. 17)

    Args:
        cumulative_residuals: T x N DataFrame of cumulative idiosyncratic residuals.
        ou_params: Dictionary containing O-U parameters for each asset.
        modified: Whether to use the drift-adjusted s-score (Section 4.2 of the paper).

    Returns:
        DataFrame of s-scores for assets satisfying the mean-reversion speed filter.
    """
    # Build parameter Series for all assets
    kappa    = pd.Series({a: p["kappa"]    for a, p in ou_params.items()})
    m        = pd.Series({a: p["m"]        for a, p in ou_params.items()})
    sigma_eq = pd.Series({a: p["sigma_eq"] for a, p in ou_params.items()})

    # Retain only assets with fast mean-reversion (half-life < ~1 month)
    valid_assets = kappa[kappa > 8.4].index

    if valid_assets.empty:
        print("Warning: no assets satisfy kappa > 8.4")
        return pd.DataFrame(index=cumulative_residuals.index)

    # Standard s-score: s = (X - m) / sigma_eq
    s_scores = (
        cumulative_residuals[valid_assets] - m[valid_assets]
    ) / sigma_eq[valid_assets]

    if modified:
        # Modified s-score includes the annualised drift alpha_i (eq. 17)
        alpha = pd.Series({a: ou_params[a]["alphas"] for a in valid_assets})
        s_scores = s_scores - alpha / (kappa[valid_assets] * sigma_eq[valid_assets])

    return s_scores


def update_positions(
    current_positions: dict[str, float],
    cur_s_scores: pd.Series,
    valid_assets: list[str],
    s_bo: float = 1.25,
    s_so: float = 1.25,
    s_bc: float = 0.50,
    s_sc: float = 0.50,
) -> dict[str, float]:
    
    new_positions: dict[str, float] = {}

    # Force-close positions for assets that dropped out of the universe entirely
    for asset in current_positions:
        if asset not in cur_s_scores.index:
            new_positions[asset] = 0.0

    for asset in cur_s_scores.index:
        s    = cur_s_scores[asset]
        prev = current_positions.get(asset, 0.0)

        # Force-close if asset no longer passes the kappa filter
        if asset not in valid_assets:
            new_positions[asset] = 0.0

        elif prev == 1.0:   # currently long
            new_positions[asset] = 0.0 if s > -s_bc else 1.0

        elif prev == -1.0:  # currently short
            new_positions[asset] = 0.0 if s < s_sc  else -1.0

        else:               # flat: check for new entry signals
            if s < -s_bo:
                new_positions[asset] = 1.0
            elif s > s_so:
                new_positions[asset] = -1.0
            else:
                new_positions[asset] = 0.0

    return new_positions



def compute_portfolio_weights(
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute portfolio weights from trading signals.

    Args:
        positions: Position signals for all assets (+1, -1, 0).

    Returns:
        Portfolio weights.
    """
    weights = positions.copy()
    active_signals_count = positions.abs().sum(axis=1)

    weights = positions.div(active_signals_count.replace(0, np.nan), axis=0).fillna(0.0)


    return weights


def compute_strategy_statistics(
    cumulative_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute performance statistics for the strategy.

    Args:
        cumulative_returns: Cumulative portfolio returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of trading periods per year.

    Returns:
        Performance statistics dict.
    """
    returns = cumulative_returns.pct_change().dropna()

    # Annualized return
    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0] - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Annualized volatility
    annualized_vol = returns.std() * np.sqrt(periods_per_year)

    # Sharpe ratio
    sharpe = (
        (annualized_return - risk_free_rate) / annualized_vol
        if annualized_vol > 0
        else 0
    )

    # Maximum drawdown
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
