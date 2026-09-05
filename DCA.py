import numpy as np
import pandas as pd


def create_dca_dataframe(
    start_date: str = "2026-01",
    num_months: int = 120,
    monthly_investment: float = 500.0,
    num_simulations: int = 1000,
    annual_mean_return: float = 0.08,
    annual_volatility: float = 0.18,
    initial_lump_sum: float = 0.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Creates a DataFrame with a YYYYMM index representing months and sets up
    the foundation for a Monte Carlo Dollar Cost Averaging (DCA) simulation.

    Parameters:
    -----------
    start_date : str
        Start year and month in format 'YYYY-MM' or 'YYYY-MM-DD' (e.g. '2026-01').
    num_months : int
        Total number of monthly periods (e.g. 120 for 10 years).
    monthly_investment : float
        Fixed dollar amount invested each month in DCA.
    num_simulations : int
        Number of Monte Carlo paths to simulate.
    annual_mean_return : float
        Expected annualized nominal market return (e.g. 0.08 for 8%).
    annual_volatility : float
        Annualized volatility / standard deviation (e.g. 0.18 for 18%).
    initial_lump_sum : float
        Optional starting capital at month 0.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by 'YYYYMM' string format (e.g. '202601', '202602', ...).
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Create monthly period index formatted as YYYYMM
    periods = pd.period_range(start=start_date, periods=num_months, freq="M")
    yyyymm_index = periods.strftime("%Y%m")

    # 2. Base timeline DataFrame
    df = pd.DataFrame(index=yyyymm_index)
    df.index.name = "period"

    # Monthly parameters using Geometric Brownian Motion (GBM)
    dt = 1 / 12  # 1 month in years
    monthly_mu = (annual_mean_return - 0.5 * annual_volatility**2) * dt
    monthly_sigma = annual_volatility * np.sqrt(dt)

    # 3. Simulate monthly returns for each Monte Carlo iteration: shape (num_months, num_simulations)
    # Using log-normal returns: R_t = exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
    z = np.random.normal(0, 1, size=(num_months, num_simulations))
    monthly_returns = np.exp(monthly_mu + monthly_sigma * z)

    # 4. Generate asset price paths starting at an arbitrary base price (e.g., $100)
    initial_price = 100.0
    price_paths = np.zeros((num_months, num_simulations))
    price_paths[0, :] = initial_price * monthly_returns[0, :]
    for t in range(1, num_months):
        price_paths[t, :] = price_paths[t - 1, :] * monthly_returns[t, :]

    # 5. Track cumulative contributions
    cumulative_contributions = np.arange(1, num_months + 1) * monthly_investment + initial_lump_sum
    df["cumulative_invested"] = cumulative_contributions

    # 6. Simulate DCA portfolio value across all simulations
    # Units purchased each month = monthly_investment / price_at_month_t
    units_bought = monthly_investment / price_paths  # shape: (num_months, num_simulations)
    cumulative_units = np.cumsum(units_bought, axis=0)
    portfolio_values = cumulative_units * price_paths  # Portfolio market value

    # Add summary statistics across Monte Carlo paths
    df["dca_median_value"] = np.median(portfolio_values, axis=1)
    df["dca_p10_value"] = np.percentile(portfolio_values, 10, axis=1)
    df["dca_p90_value"] = np.percentile(portfolio_values, 90, axis=1)
    df["dca_mean_value"] = np.mean(portfolio_values, axis=1)

    return df


if __name__ == "__main__":
    # Generate DataFrame starting in June 2026 for 10 years (120 months)
    df_dca = create_dca_dataframe(
        start_date="2026-06",
        num_months=120,
        monthly_investment=500.0,
        num_simulations=5000,
        annual_mean_return=0.08,
        annual_volatility=0.18,
    )

    print("--- DataFrame Preview (First 5 Periods) ---")
    print(df_dca.head())

    print("\n--- DataFrame Summary (Final 5 Periods) ---")
    print(df_dca.tail())
