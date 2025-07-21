import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# === Utility Functions ===
def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(i) for i in col if i]) for col in df.columns.values]
    return df

def check_columns(df, required, name):
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns in {name} data: {missing}")
        sys.exit(1)

def check_data(df, name, min_len=10):
    if df.empty or len(df) < min_len:
        print(f"Not enough data for {name}. Got {len(df)} rows.")
        sys.exit(1)

# === Get Oil Prices (WTI Crude) ===
oil = yf.download("CL=F", period="3mo", interval="1d")
oil = flatten_columns(oil)
if oil.empty:
    print("No oil data downloaded.")
    sys.exit(1)

oil_close_col = next((col for col in oil.columns if col.startswith('Close')), None)
if oil_close_col is None:
    print("No 'Close' column found in oil data.")
    sys.exit(1)

check_columns(oil, [oil_close_col], 'oil')
oil['returns'] = oil[oil_close_col].pct_change()
oil = oil.dropna(subset=[oil_close_col, 'returns'])

# === Get USD Strength (DXY Index) ===
usd = yf.download("DX-Y.NYB", period="3mo", interval="1d")
usd = flatten_columns(usd)
if usd.empty:
    print("No USD data downloaded.")
    sys.exit(1)

usd_close_col = next((col for col in usd.columns if col.startswith('Close')), None)
if usd_close_col is None:
    print("No 'Close' column found in usd data.")
    sys.exit(1)

check_columns(usd, [usd_close_col], 'usd')
usd = usd.dropna(subset=[usd_close_col])

# === Validate Data Lengths ===
check_data(oil, 'oil')
check_data(usd, 'usd')

# === Compute Drift and EMA Volatility ===
mu = oil['returns'].mean()
ema_sigma = oil['returns'].ewm(span=20).std().clip(lower=0.005)  # EMA volatility with floor
S0 = oil[oil_close_col].iloc[-1]

#USD strength index
usd_latest = usd[usd_close_col].iloc[-1]

if np.isnan(mu) or np.isnan(S0):
    print("Simulation parameters contain NaN. Check data quality.")
    sys.exit(1)

# === Simulation Parameters ===
T = 30  # days
N = 1000
dt = 1

# === USD Effect ===
usd_impact = -0.0001 * (usd_latest - 100)

# === Prepare Simulation ===
price_paths = np.zeros((N, T))

# Use the latest EMA sigma value to start
initial_sigma = ema_sigma.iloc[-1]

for i in range(N):
    prices = [S0]
    for t in range(1, T):
        # Dynamically simulate sigma (stochastic smoothing)
        if t < len(ema_sigma):
            sigma_t = ema_sigma.iloc[-t]  # Use recent past values (reverse order)
        else:
            sigma_t = initial_sigma  # fallback to most recent
        
        Z = np.random.normal()
        adj_mu = mu + usd_impact
        St = prices[-1] * np.exp((adj_mu - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * Z)
        prices.append(St)

    # Pad/truncate prices to T days
    if len(prices) > T:
        prices = prices[:T]
    elif len(prices) < T:
        prices += [prices[-1]] * (T - len(prices))

    price_paths[i] = prices

# === Plot the Simulation ===
plt.figure(figsize=(10, 5))
for i in range(50):
    plt.plot(price_paths[i], alpha=0.3)
plt.title("Simulated Oil Prices (EMA Volatility)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.grid()
plt.show()
