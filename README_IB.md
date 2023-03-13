We need to access IB for market data. To do that, we first need to run IBGateway and IBC.
IBGateway is provided by IB and IBC is an open source UX controller providing headless API.
Our client application will interact with IBC API.

## Docker

```bash
make build-ib TWS_USERID=<userid> TWS_PASSWORD=<password>
```

### Run IB Gateway


```bash
make run-notebook
```

### Query IB


```bash
TODO
```

### Run Tests

To run the test suite locally, execute the following command.

```bash
TODO
```

