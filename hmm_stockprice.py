import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import yfinance as yf

# Download historical stock price data
stock_data = yf.download("SPY", start="2010-01-01", end="2020-01-01")
stock_data['Returns'] = stock_data['Adj Close'].pct_change().dropna()

# Prepare data for HMM
X = stock_data['Returns'].dropna().values.reshape(-1, 1)

# Define and train the HMM
model = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000)
model.fit(X)

# Predict hidden states
hidden_states = model.predict(X)

# Add hidden states to the DataFrame
stock_data = stock_data.iloc[1:]  # Adjust for the first NaN return
stock_data['Hidden State'] = hidden_states


# Predict future stock prices using the HMM
def predict_prices(model, X, n_days):
    hidden_states = model.predict(X)
    state_means = model.means_

    predictions = []
    last_price = stock_data['Adj Close'].iloc[-1]

    for _ in range(n_days):
        current_state = hidden_states[-1]
        predicted_return = np.random.normal(state_means[current_state][0], np.sqrt(model.covars_[current_state][0][0]))
        next_price = last_price * (1 + predicted_return)
        predictions.append(next_price)
        last_price = next_price
        hidden_states = np.append(hidden_states, current_state)

    return predictions


# Predict stock prices for the next 30 days
predicted_prices = predict_prices(model, X, 50)

# Plot the actual and predicted stock prices
plt.figure(figsize=(15, 8))

# Plot actual stock prices
plt.plot(stock_data.index, stock_data['Adj Close'], label='Actual Prices')

# Plot predicted stock prices
future_dates = pd.date_range(start=stock_data.index[-1], periods=50, freq='B')
plt.plot(future_dates, predicted_prices, label='Predicted Prices', linestyle='--')

plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
