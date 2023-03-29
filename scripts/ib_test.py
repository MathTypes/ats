from ib_insync import *
from ibapi import wrapper
from ibapi.common import TickerId, BarData
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.utils import iswrapper

class DownloadApp(EClient, wrapper.EWrapper):
    def __init__(self):
        EClient.__init__(self, wrapper=self)
        wrapper.EWrapper.__init__(self)

#app = DownloadApp()
#app.connect("127.0.0.1", args.port, clientId=1)
ib.connect('127.0.0.1', 4001, clientId = 1)

instruments = [
    {
        "ticker": "CL", "exchange": "NYMEX" 
    }, {
        "ticker": "ES", "exchange": "GLOBEX"
    }, {
        "ticker": "EUR", "exchange": "GLOBEX"
    }, {
        "ticker": "GC", "exchange": "NYMEX"
    }, {
        "ticker": "HE", "exchange": "GLOBEX"
    }, {
        "ticker": "HG", "exchange": "NYMEX"
    }, {
        "ticker": "JPY", "exchange": "GLOBEX"
    }, {
        "ticker": "LE", "exchange": "GLOBEX"
    }, {
        "ticker": "NG", "exchange": "NYMEX"
    }, {
        "ticker": "NQ", "exchange": "GLOBEX"
    }, {
        "ticker": "ZC", "exchange": "ECBOT"
    }, {
        "ticker": "ZF", "exchange": "ECBOT"
    }, {
        "ticker": "ZN", "exchange": "ECBOT"
    }, {
        "ticker": "ZS", "exchange": "ECBOT"
    },
]

for instrument in instruments:
    print( instrument['ticker'], instrument['exchange'] )
    contract = ContFuture(instrument['ticker'], instrument['exchange'])

    bars = ib.reqHistoricalData(
        contract, endDateTime='', durationStr='6 M',
        barSizeSetting='1 day', whatToShow='TRADES', useRTH=False)

    # convert to pandas dataframe:
    df = util.df(bars)    
    df.to_csv( instrument['ticker'] + '.csv')
print("Done")

ib.disconnect()