from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.utils import iswrapper
import logging
# MAX: necessary imports for multi-threading
from threading import Thread
from queue import Queue
import time
from config import *


class IBapi(EWrapper, EClient):
	def __init__(self):
		EClient.__init__(self, self)
		# MAX: message queue for inter thread communication
		self.queue = Queue()
		self.started = False
		self.request_id = 0
		self.next_valid_order_id = None
		self.pending_ends = set()
		self.requests = {}

    # MAX: function to send the termination signal
	def send_done(self, code):
		print(f'Sending code {code}')
		self.queue.put(code)

    # MAX: function to wait for the termination signal
	def wait_done(self):
		print('Waiting for thread to finish ...')
		code = self.queue.get()
		print(f'Received code {code}')
		self.queue.task_done()
		return code

	def error(self, reqId, errorCode:int, errorString):
		print ("Server Error: %s" % errorString)

	def reply_handler(msg):
		print ("Server Response: %s, %s" % (msg.typeName, msg))

	def tickPrice(self, reqId, tickType, price, attrib):
		if tickType == 2 and reqId == 1:
			print('The current ask price is: ', price)

	#You can change this to manipulate the news however you please
	def tickNews(self, tickerId: int, timeStamp: int, providerCode: str, articleId: str, headline: str, extraData: str):
		print("TickNews. TickerId:", tickerId, "TimeStamp:", timeStamp,"ProviderCode:", providerCode, "ArticleId:", articleId,"Headline:", headline, "ExtraData:", extraData)
		#self.send_done(-1)

	def contractDetails(self, reqId: int, contractDetails):
		super().contractDetails(reqId, contractDetails)
		print(f"contractDetails:{contractDetails}")
    
	def contractDetailsEnd(self, reqId: int):
		super().contractDetailsEnd(reqId)
		print("ContractDetailsEnd. ReqId:", reqId)

	def next_request_id(self, contract: Contract) -> int:
		self.request_id += 1
		self.requests[self.request_id] = contract
		return self.request_id
	
	@iswrapper
	def nextValidId(self, order_id: int):
		super().nextValidId(order_id)

		self.next_valid_order_id = order_id
		logging.info(f"nextValidId: {order_id}")
		# we can start now
		self.start()
		
	def start(self):
		print("start")
		if self.started:
			return

		self.started = True
		contract = Contract()
		contract.secType = "NEWS"
		contract.exchange = "BRFG"
		contract.symbol = "BRFG:BRFG_ALL" 
		print(f"contract:{contract}")
		#app.reqContractDetails(10004, contract)
		cid = self.next_request_id(contract)
		self.pending_ends.add(cid)
		app.reqMktData(cid, contract, "mdoff,292", False, False, [])
	    
app = IBapi()
app.connect('127.0.0.1', port, 4)

# MAX: Start the application as a separate thread
Thread(target=app.run).start()

# MAX: Wait for the application to terminate
code = app.wait_done()
app.disconnect()
