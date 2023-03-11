import pandas_market_calendars as mcal

# Create a calendar
nyse = mcal.get_calendar('NYSE')

# Show available calendars
print(mcal.get_calendar_names())
early = nyse.schedule(start_date='2012-07-01', end_date='2012-07-10')
print(early)
print(mcal.date_range(early, frequency='1D'))
print(mcal.date_range(early, frequency='1H'))
