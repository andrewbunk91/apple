import yfinance as yf

# Retrieve historical data, swap AAPL with whatever stock you want
stock_data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')

# Save the data to a CSV file within work folder, rename 'whatever_here'
stock_data.to_csv('apple_stock_data_2023.csv')
