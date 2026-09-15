import pandas as pd


def _align_portfolios_and_returns(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align a portfolio schedule with the return matrix on the common investable universe.

    Parameters:
        portfolios (pd.DataFrame): Portfolio weights indexed by rebalance date.
        returns (pd.DataFrame): Asset returns indexed by trading date.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The aligned portfolio and return matrices.
    """
    if portfolios.empty or portfolios.shape[1] == 0:
        raise ValueError("portfolios must contain at least one asset")

    overlapping_assets = portfolios.columns.intersection(returns.columns)
    if overlapping_assets.empty:
        raise ValueError("portfolios and returns must share at least one asset")

    aligned_portfolios = (
        portfolios.loc[:, overlapping_assets]
        .reindex(returns.index)
        .ffill()
        .shift()
        .dropna(axis=0, how="all")
    )
    if aligned_portfolios.empty:
        raise ValueError(
            "portfolios and returns must overlap on at least one investable date"
        )

    aligned_returns = returns.loc[aligned_portfolios.index, overlapping_assets]

    return aligned_portfolios.fillna(0.0), aligned_returns


def portfolio_returns(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: float = 0.0,
) -> pd.Series:
    """
    Compute the time series of realized portfolio returns from a rebalance schedule.

    Parameters:
        portfolios (pd.DataFrame): DataFrame where each column represents the weights of a
            portfolio over time (dates as index).
        returns (pd.DataFrame): DataFrame of asset returns (dates as index, assets as columns).
        transaction_costs (float): All-in transaction cost rate applied to one-way turnover
            (e.g. 0.001 = 10 bps). Defaults to 0.0.

    Returns:
        pd.Series: Series of realized portfolio returns over time.
    """
    aligned_portfolios, aligned_returns = _align_portfolios_and_returns(
        portfolios=portfolios,
        returns=returns,
    )

    gross_returns = aligned_portfolios.multiply(aligned_returns).sum(axis=1)

    if transaction_costs != 0.0:
        weight_changes = aligned_portfolios.diff().abs().sum(axis=1)
        weight_changes.iloc[0] = aligned_portfolios.iloc[0].abs().sum()

        gross_returns = gross_returns - (weight_changes * transaction_costs)

    return gross_returns


def backtest(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: float = 0.0,
) -> pd.Series:
    """
    Compute cumulative portfolio performance from a rebalance schedule.

    Parameters:
        portfolios (pd.DataFrame): DataFrame where each column represents the weights of a
            portfolio over time (dates as index).
        returns (pd.DataFrame): DataFrame of asset returns (dates as index, assets as columns).
        transaction_costs (float): All-in transaction cost rate applied to one-way turnover
            (e.g. 0.001 = 10 bps). Defaults to 0.0.

    Returns:
        pd.Series: Series representing the cumulative returns of the portfolios over time.
    """
    return (
        portfolio_returns(
            portfolios=portfolios,
            returns=returns,
            transaction_costs=transaction_costs,
        )
        .add(1)
        .cumprod()
    )
