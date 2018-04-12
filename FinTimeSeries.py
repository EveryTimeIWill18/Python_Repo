from FinPy import *

# --- load in data dictionary
data = load_datasets("E:\\", "^OVX.csv", "^VIX.csv", "^SP500TR.csv", "48_Industry_Portfolios.CSV")

"""
df_ovx = (data.get("^OVX.csv")
              .drop(['High', 'Open', 'Volume', 'Low', 'Close'], axis=1)
              .assign(Lagged=lambda x: x['Adj Close'].shift(periods=-1))
              .assign(Shifted=lambda x: x['Adj Close'].shift())
              .assign(Change=lambda x: x['Adj Close'].div(x.Shifted))
              .assign(Return=lambda x: x['Adj Close'].div(x.Shifted).sub(1).mul(100)))
"""
# --- ovx DataFrame
df_ovx = (data.get("^OVX.csv")
           .drop(['High', 'Open', 'Volume', 'Low', 'Close'], axis=1)
           .assign(Diff=lambda x: x['Adj Close'].diff())
           .assign(Pct_Change=lambda x: x['Adj Close'].pct_change().mul(100))
          )

# --- vix DataFrame
df_vix = (data.get("^VIX.csv")
           .drop(['High', 'Open', 'Volume', 'Low', 'Close'], axis=1)
           .assign(Diff=lambda x: x['Adj Close'].diff())
           .assign(Pct_Change=lambda x: x['Adj Close'].pct_change().mul(100))
          )

# --- sp500 DataFrame
df_sp500 = (data.get("^SP500TR.csv")
             .drop(['High', 'Open', 'Volume', 'Low', 'Close'], axis=1)
             .assign(Diff=lambda x: x['Adj Close'].diff())
             .assign(Pct_Change=lambda x: x['Adj Close'].pct_change().mul(100))
            )

df_ff49 = (data.get("48_Industry_Portfolios.CSV"))

ff49_cols = list(df_ff49.columns)

