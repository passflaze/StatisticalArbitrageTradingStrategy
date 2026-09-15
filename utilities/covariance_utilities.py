from typing import Any

import numpy as np
import pandas as pd


def _validate_covariance_matrix(
    covariance: np.ndarray,
    name: str,
    require_positive_definite: bool = False,
    positive_definite_message: str | None = None,
) -> np.ndarray:
    """
    Validate a covariance matrix used by the portfolio utilities.

    Parameters:
        covariance (np.ndarray): Covariance matrix to validate.
        name (str): Parameter name used in error messages.
        require_positive_definite (bool): Whether to enforce positive definiteness.
        positive_definite_message (str | None): Optional custom error message for the
            positive-definite check.

    Returns:
        np.ndarray: The validated covariance matrix.
    """
    if not isinstance(covariance, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(covariance)}")

    if covariance.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got shape {covariance.shape}")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be square, got shape {covariance.shape}")

    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} contains NaN or Inf values")

    if require_positive_definite:
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            if positive_definite_message is None:
                positive_definite_message = f"{name} must be positive definite"
            raise ValueError(positive_definite_message) from exc

    return covariance


def prepare_rolling_estimation_window(
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    lookback: int,
    min_coverage: float = 0.95,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build a trailing estimation window for covariance-based analysis.

    Parameters:
        returns (pd.DataFrame): Asset return history indexed by date.
        rebalance_date (pd.Timestamp): Last date included in the window.
        lookback (int): Target number of rows in the trailing window.
        min_coverage (float): Minimum fraction of non-missing observations required
            for an asset to remain in the window.
        return_diagnostics (bool): Whether to also return filtering diagnostics.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: The filtered window, and
            optionally diagnostics describing the retained and dropped assets.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError(
            f"min_coverage must be in the interval (0, 1], got {min_coverage}"
        )

    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    trailing_window = (
        returns.loc[:rebalance_date]
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .iloc[-lookback:]
    )
    coverage = trailing_window.notna().mean(axis=0)
    eligible_assets = coverage[coverage >= min_coverage].index
    filtered_window = trailing_window.loc[:, eligible_assets].fillna(0.0)

    if not return_diagnostics:
        return filtered_window

    diagnostics = {
        "row_count": trailing_window.shape[0],
        "asset_count_before_filter": trailing_window.shape[1],
        "asset_count_after_filter": filtered_window.shape[1],
        "coverage": coverage.sort_values(ascending=False),
        "retained_assets": list(eligible_assets),
        "dropped_assets": list(coverage[coverage < min_coverage].index),
    }
    return filtered_window, diagnostics
