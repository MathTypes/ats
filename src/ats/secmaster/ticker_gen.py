import yfinance as yf

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2023, 4, 1)
data = yf.download(tickers, start=start, end=end)
data.csv("symbol.csv")
