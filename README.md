AkoloAts is an automated trading system. It will consist of following components:
- **Security Master**: This will manage universe of financial instrument, market calendar, etc.
- **Market Data Feed**: This handes market data ingestion for both realtime, historic backfills and data clean processing.
- **Event Feed**: This handes events ingestion for both realtime, historic backfills and data clean processing.
- **Offline training**: This generates training examples from historic marketdata and event and trains models to generate trading signal online.

- **Online prediction**": This runs in real time to generate trading signals.

## Getting Started


## Installation

AkoloAts requires Python >= 3.7 for all functionality to work as expected.

```bash
pip install -r requirements.txt
```

## Docker

To run the commands below, ensure Docker is installed. Visit https://docs.docker.com/install/ for more information.

### Run Jupyter Notebooks

To run a jupyter notebook in your browser, execute the following command and visit the `http://127.0.0.1:8888/?token=...` link printed to the command line.

```bash
make run-notebook
```

### Build Documentation

To build the HTML documentation, execute the following command.

```bash
make run-docs
```

### Run Test Suite

To run the test suite, execute the following command.

```bash
make run-tests
```


