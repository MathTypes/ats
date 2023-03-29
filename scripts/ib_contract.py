from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook
#
#
import pandas as pd


def add_contract_details(ib_client, ib_contract, df):
    list_of_contract_details = ib_client.reqContractDetails(
        contract=ib_contract)
    if list_of_contract_details:
        print(
            f"Found {len(list_of_contract_details)} contract{'s' if len(list_of_contract_details) > 1 else ''} for {ib_contract.symbol}: "
        )
        for contract_details in list_of_contract_details:
            print(f'contract_details:{contract_details}')
            data = {}
            for k, v in contract_details.contract.__dict__.items():
                data[k] = v
            for k, v in contract_details.__dict__.items():
                if k != "contract":
                    data[k] = v
            df = pd.DataFrame([data]) if df is None else df.append(
                pd.DataFrame([data]))
    else:
        print(f"No details found for contract {ib_contract.symbol}.")
    return df


ib = IB()
ib.connect(host="127.0.0.1", port=4001, clientId=1)
btc_fut_cont_contract = ContFuture("BRR", "CMECRYPTO")
ib.qualifyContracts(btc_fut_cont_contract)
fx_contract_usdjpy = Forex('USDJPY')
ib.qualifyContracts(fx_contract_usdjpy)
es_fut_cont_contract = ContFuture("ES", "SMART")
ib.qualifyContracts(es_fut_cont_contract)
nq_fut_cont_contract = ContFuture("NQ", "SMART")
ib.qualifyContracts(nq_fut_cont_contract)

df_contract_details = None
for c in [btc_fut_cont_contract, fx_contract_usdjpy, es_fut_cont_contract, nq_fut_cont_contract]:
    df_contract_details = add_contract_details(ib, c, df_contract_details)
print(df_contract_details)
