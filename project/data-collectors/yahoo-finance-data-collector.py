import yfinance as yf
import os

tsla = yf.Ticker("TSLA")
# get stock info
stock_info = tsla.info
# get historical market data
stock_data = tsla.history(period="2y")
dirpath = "../data/raw/price-movements"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

price_change_column = stock_data["Close"] - stock_data["Open"]

stock_data["Price_Change"] = price_change_column


stock_data_csv = "tesla-stock-movement.csv"
stock_data.to_csv(dirpath + '/' + stock_data_csv)
