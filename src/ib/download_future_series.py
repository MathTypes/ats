#!/usr/bin/env python

# Usage: python3 download_hist.py AAPL -p=4001 --start_date="20230304"
#             --end_date="20230309" --duration="5 D" --size="1 day"
#             --exchange=SMART -t MIDPOINT
# Note that there seems to be a bug with loading TRADES.
#
import os
import sys
import argparse
import logging
import math
import time

from datetime import datetime, timedelta
import pytz

# MAX: necessary imports for multi-threading
from threading import Thread
from queue import Queue

from typing import List, Optional
from collections import defaultdict
from dateutil.parser import parse

import numpy as np
import pandas as pd

from ibapi import wrapper
from ibapi.common import TickerId, BarData
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.utils import iswrapper

ContractList = List[Contract]
BarDataList = List[BarData]
OptionalDate = Optional[datetime]


def make_download_path(base_directory, security_type, size, contract: Contract) -> str:
    """Make path for saving csv files.
    Files to be stored in base_directory/<security_type>/<size>/<symbol>/
    """
    path = os.path.sep.join(
        [
            base_directory,
            security_type,
            size.replace(" ", "_"),
            contract.localSymbol if contract.localSymbol else contract.symbol,
        ]
    )
    return path


class DownloadApp(EClient, wrapper.EWrapper):
    def __init__(self, contracts: ContractList, start_date, end_date, duration, base_directory, size, data_type, security_type):
        logging.info(f"DownloadApp: start_date:{start_date}, end_date:{end_date}")
        EClient.__init__(self, wrapper=self)
        wrapper.EWrapper.__init__(self)
        self.request_id = math.floor(time.time())
        self.started = False
        self.start_date = start_date
        self.end_date = end_date
        self.next_valid_order_id = None
        self.contracts = contracts
        self.requests = {}
        self.bar_data = defaultdict(list)
        self.security_type = security_type
        self.pending_ends = set()
        #self.args = args
        self.current = end_date
        self.data_type = data_type
        self.base_directory = base_directory
        self.size = size
        self.duration = duration
        self.useRTH = 0
        # MAX: message queue for inter thread communication
        self.queue = Queue()

    # MAX: function to send the termination signal
    def send_done(self, code):
        print(f"Sending code {code}")
        self.queue.put(code)

    # MAX: function to wait for the termination signal
    def wait_done(self):
        print("Waiting for thread to finish ...")
        code = self.queue.get()
        print(f"Received code {code}")
        self.queue.task_done()
        return code

    def next_request_id(self, contract: Contract) -> int:
        self.request_id += 1
        self.requests[self.request_id] = contract
        return self.request_id

    def historicalDataRequest(self, contract: Contract) -> None:
        cid = self.next_request_id(contract)
        self.pending_ends.add(cid)
        logging.error(f"send historic data request:{contract}, {self}")
        self.reqHistoricalData(
            cid,  # tickerId, used to identify incoming data
            contract,
            self.current.strftime("%Y%m%d-21:59:59"),  # always go to midnight
            self.duration,  # amount of time to go back
            self.size,  # bar size
            self.data_type,  # historical data type
            self.useRTH,  # useRTH (regular trading hours)
            1,  # format the date in yyyyMMdd HH:mm:ss
            False,  # keep up to date after snapshot
            [],  # chart options
        )

    def save_data(self, contract: Contract, bars: BarDataList) -> None:
        #logging.error(f"save_data, contract:{contract}, bars:{bars}")
        data = [
            # MAX: IBAPI 10.15 does not provide bar.average anymore
            # MAX: IBAPI 10.15 has an attribute bar.wap (weighted average)
            [
                b.date,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.barCount
                # , b.wap
            ]
            for b in bars
        ]
        # MAX: IBAPI 10.15 does not provide bar.average anymore
        # MAX: IBAPI 10.15 has an attribute bar.wap (weighted average)
        df = pd.DataFrame(
            data,
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "barCount",
                # "wap"
            ],
        )
        if self.daily_files():
            path = "%s.csv" % make_download_path(self.base_directory, self.security_type, self.size, contract)
        else:
            # since we fetched data until midnight, store data in
            # date file to which it belongs
            #last = (self.current - timedelta(days=1)).strftime("%Y%m%d")
            last = (self.current).strftime("%Y%m%d")
            path = os.path.sep.join(
                [make_download_path(self.base_directory, self.security_type, self.size, contract), "%s.csv" % last,]
            )
        df.to_csv(path, index=False)

    def daily_files(self):
        return SIZES.index(self.size.split()[1]) >= 5

    @iswrapper
    def headTimestamp(self, reqId: int, headTimestamp: str) -> None:
        logging.error(f"headtimestamp:{reqId}")
        contract = self.requests.get(reqId)
        if "-" in headTimestamp:
            ts = datetime.strptime(headTimestamp, "%Y%m%d-%H:%M:%S")
        else:
            ts = datetime.strptime(headTimestamp, "%Y%m%d %H:%M:%S")
        logging.info("Head Timestamp for %s is %s, start_date:%s", contract, ts, self.start_date)
        #if ts > self.start_date or self.args.max_days:
        if ts > self.start_date:
            logging.warning("Overriding start date, setting to %s", ts)
            self.start_date = ts  # TODO make this per contract
        if ts > self.end_date:
            logging.warning("Data for %s is not available before %s", contract, ts)
            # MAX: send termination signal
            self.send_done(-1)
            return
        # if we are getting daily data or longer, we'll grab the entire amount at once
        if self.daily_files():
            days = (self.end_date - self.start_date).days
            logging.info(
                f"days:{days}, start_date:{self.start_date}, end_date:{self.end_date}"
            )
            if days < 365:
                self.duration = "%d D" % days
            else:
                self.duration = "%d Y" % np.ceil(days / 365)
            # when getting daily data, look at regular trading hours only
            # to get accurate daily closing prices
            self.useRTH = 0
            # round up current time to midnight for even days
            self.current = self.current.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        self.historicalDataRequest(contract)

    @iswrapper
    def historicalData(self, reqId: int, bar) -> None:
        self.bar_data[reqId].append(bar)

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        super().historicalDataEnd(reqId, start, end)
        self.pending_ends.remove(reqId)
        if len(self.pending_ends) == 0:
            print(f"All requests for {self.current} complete.")
            for rid, bars in self.bar_data.items():
                self.save_data(self.requests[rid], bars)
            if "/" in start:
                start_vec = start.split()                
                parsed_datetime = datetime.strptime(start_vec[0] + " " + start_vec[1], "%Y%m%d  %H:%M:%S")
                parsed_tz = pytz.timezone(start_vec[2])
                self.current = parsed_datetime.astimezone(parsed_tz)
            else:
                self.current = datetime.strptime(start, "%Y%m%d  %H:%M:%S")
            logging.info(f"current:{self.current} - start:{self.start_date}")
            if self.current.date() <= self.start_date.date():
                # MAX: send termination signal
                self.send_done(0)
            else:
                for contract in self.contracts:
                    self.historicalDataRequest(contract)

    @iswrapper
    def connectAck(self):
        logging.error("Connected")

    @iswrapper
    def nextValidId(self, order_id: int):
        super().nextValidId(order_id)

        self.next_valid_order_id = order_id
        logging.info(f"nextValidId: {order_id}")
        # we can start now
        self.start()

    def start(self):
        if self.started:
            return

        self.started = True
        for contract in self.contracts:
            self.reqHeadTimeStamp(
                self.next_request_id(contract), contract, self.data_type, 0, 1
            )

    @iswrapper
    # MAX: IBAPI 10.15 defines an additional parameter: advancedOrderRejectJson
    # def error(self, req_id: TickerId, error_code: int, error: str, advancedOrderRejectJson: str):
    def error(
        self, req_id: TickerId, error_code: int, error: str, advancedOrderRejectJson=""
    ):
        logging.error("Error. Id: %s Code %s Msg: %s", req_id, error_code, error)
        super().error(req_id, error_code, error)
        if req_id < 0:
            logging.error("Error. Id: %s Code %s Msg: %s", req_id, error_code, error)
        else:
            logging.error("Error. Id: %s Code %s Msg: %s", req_id, error_code, error)
            # we will always exit on error since data will need to be validated
            self.done = True
            self.send_done(error_code)


def make_contract(
    symbol: str,
    sec_type: str,
    currency: str,
    exchange: str,
    localsymbol: str,
    last_trade_date: str,
    include_expired: bool,
) -> Contract:
    contract = Contract()
    # contract.symbol = "EUR"
    # contract.secType = "CASH"
    # contract.currency = "USD"
    # contract.exchange = "IDEALPRO"

    # contract.conId = 495512569
    contract.symbol = symbol
    # contract.localSymbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    logging.info(f"include_expired:{include_expired}")
    if include_expired:
        contract.includeExpired = True
    if localsymbol:
        # contract.tradingClass = localsymbol
        contract.localSymbol = localsymbol
    contract.currency = "USD"
    if last_trade_date:
        contract.lastTradeDateOrContractMonth = last_trade_date
    logging.info(f"symbol:{symbol}")
    logging.info(f"exchange:{exchange}")
    logging.info(f"tradingClass:{localsymbol}")
    logging.info(f"lastTradeDate:{last_trade_date}")
    logging.info(f"contract:{contract}")
    return contract


class ValidationException(Exception):
    pass


def _validate_in(value: str, name: str, valid: List[str]) -> None:
    if value not in valid:
        raise ValidationException(f"{value} not a valid {name} unit: {','.join(valid)}")


def _validate(value: str, name: str, valid: List[str]) -> None:
    tokens = value.split()
    if len(tokens) != 2:
        raise ValidationException("{name} should be in the form <digit> <{name}>")
    _validate_in(tokens[1], name, valid)
    try:
        int(tokens[0])
    except ValueError as ve:
        raise ValidationException(f"{name} dimenion not a valid number: {ve}")


SIZES = ["secs", "min", "mins", "hour", "hours", "day", "week", "month"]
DURATIONS = ["S", "D", "W", "M", "Y"]


def validate_duration(duration: str) -> None:
    _validate(duration, "duration", DURATIONS)


def validate_size(size: str) -> None:
    _validate(size, "size", SIZES)


def validate_data_type(data_type: str) -> None:
    _validate_in(
        data_type,
        "data_type",
        [
            "TRADES",
            "MIDPOINT",
            "BID",
            "ASK",
            "BID_ASK",
            "ADJUSTED_LAST",
            "HISTORICAL_VOLATILITY",
            "OPTION_IMPLIED_VOLATILITY",
            "REBATE_RATE",
            "FEE_RATE",
            "YIELD_BID",
            "YIELD_ASK",
            "YIELD_BID_ASK",
            "YIELD_LAST",
        ],
    )

def get_exchange(symbol):
    if symbol in ["ES", "NQ", "RTY"]:
        return "CME"
    if symbol in ["CL"]:
        return "NYMEX"
    if symbol in ["GC", "SI", "HG"]:
        return "COMEX"
    return ""


def get_local_symbol(symbol, cur_date):
    month_str = ""
    last_trade_date = cur_date
    if symbol in ["ES", "NQ", "RTY"]:
        if cur_date.month < 3:
            month_str = "H"
            last_trade_date = last_trade_date.replace(month=3, day=1)
        elif cur_date.month < 6:
            month_str = "M"
            last_trade_date = last_trade_date.replace(month=6, day=1)
        elif cur_date.month < 9:
            month_str = "U"
            last_trade_date = last_trade_date.replace(month=9, day=1)
        else:
            month_str = "Z"
            last_trade_date = last_trade_date.replace(month=12, day=1)
    year_str = str(cur_date.year % 10)
    logging.info(f'month_str:{month_str}, last_trade:{last_trade_date}')
    return symbol + month_str + year_str, last_trade_date.strftime("%Y%m")

def get_next_local_symbol(symbol, cur_date):
    month_str = ""
    last_trade_date = cur_date
    if symbol in ["ES", "NQ", "RTY"]:
        if cur_date.month < 3:
            month_str = "M"
            last_trade_date = last_trade_date.replace(month=3, day=1)
            year_str = str((cur_date.year) % 10)
        elif cur_date.month < 6:
            month_str = "U"
            last_trade_date = last_trade_date.replace(month=6, day=1)
            year_str = str((cur_date.year) % 10)
        elif cur_date.month < 9:
            month_str = "Z"
            last_trade_date = last_trade_date.replace(month=9, day=1)
            year_str = str((cur_date.year) % 10)
        else:
            month_str = "H"
            last_trade_date = last_trade_date.replace(month=12, day=1)
            year_str = str((cur_date.year + 1) % 10)
    logging.info(f'month_str:{month_str}, last_trade:{last_trade_date}')
    return symbol + month_str + year_str, last_trade_date.strftime("%Y%m")

# borrowed from https://stackoverflow.com/a/13565185
# as noted there, the calendar module has a function of its own
def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + timedelta(days=4)  # this will never fail
    return next_month - timedelta(days=next_month.day)

def monthlist(begin,end):
    result = []
    while True:
        if begin.month == 12:
            next_month = begin.replace(year=begin.year+1,month=1, day=1)
        else:
            next_month = begin.replace(month=begin.month+1, day=1)
        if next_month > end:
            break
        result.append ([begin, last_day_of_month(begin)])
        begin = next_month
    result.append ([begin, end])
    return result


if __name__ == "__main__":
    now = datetime.now()

    class DateAction(argparse.Action):
        """Parses date strings."""

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            value: str,
            option_string: str = None,
        ):
            """Parse the date."""
            setattr(namespace, self.dest, parse(value))

    argp = argparse.ArgumentParser()
    argp.add_argument("symbol", type=str)
    argp.add_argument(
        "-d", "--debug", action="store_true", help="turn on debug logging"
    )
    argp.add_argument(
        "--max_days", action="store_true", help="turn on debug logging"
    )
    argp.add_argument("--logfile", help="log to file")
    argp.add_argument(
        "-p", "--port", type=int, default=7496, help="local port for TWS connection"
    )
    argp.add_argument(
        "--security-type", type=str, default="STK", help="security type for symbols"
    )
    argp.add_argument("--duration", type=str, default="1 D", help="bar duration")
    argp.add_argument("--size", type=str, default="1 min", help="bar size")
    argp.add_argument(
        "-t", "--data-type", type=str, default="TRADES", help="bar data type"
    )
    argp.add_argument(
        "--base-directory",
        type=str,
        default="data",
        help="base directory to write bar files",
    )
    argp.add_argument(
        "--start_date",
        help="First day for bars",
        default=now - timedelta(days=2),
        action=DateAction,
    )
    argp.add_argument(
        "--end_date", help="Last day for bars", default=now, action=DateAction,
    )
    args = argp.parse_args()

    logargs = dict(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.debug:
        logargs["level"] = logging.DEBUG
    else:
        logargs["level"] = logging.INFO

    if args.logfile:
        logargs["filemode"] = "a"
        logargs["filename"] = args.logfile

    logging.basicConfig(**logargs)

    try:
        validate_duration(args.duration)
        validate_size(args.size)
        args.data_type = args.data_type.upper()
        validate_data_type(args.data_type)
    except ValidationException as ve:
        print(ve)
        sys.exit(1)

    logging.debug(f"args={args}")
    for begin, end in monthlist(args.start_date, args.end_date):
        import shlex, subprocess
        command_line = f"python3 ib/download_fut.py {args.symbol} --start_date={begin.strftime('%Y%m%d')} --end_date={end.strftime('%Y%m%d')} --port={args.port} --duration='{args.duration}' --base-directory='{args.base_directory}' --security-type={args.security_type} --size='{args.size}' --data-type={args.data_type}"
        logging.info(f"command_line:{command_line}")
        new_args = shlex.split(command_line)
        #p = subprocess.Popen(new_args) # Success!
        with subprocess.Popen(new_args, stdout=subprocess.PIPE) as proc:
            logging.info(proc.stdout.read())

