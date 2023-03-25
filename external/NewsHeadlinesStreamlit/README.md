
# NewsHeadlinesStreamlit
News headlines visualization using Python 3, NumPy, pandas, Streamlit and pydeck

A collection of news headlines as JSON data files from a number of publicly availabe news sources (these are from the time period between the end of June 2020 and the end of July 2020). A simple Streamlit app to analyze and visualize the data.

The columns:
```
date_time, source, heading
```
are the primary data (date_time is in EST). Data in the rest of the columns were derived from the primary data (through a rudimentary analysis of the data). The data strings went through some cleanup. It can still contain spurious data. 

This is provided with the hope that it will be a useful learning tool.

## Getting Started

The following instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

You will need:
```
Python 3, NumPy, pandas, Streamlit and pydeck
``` 
### Installing

Setup a Python 3 virtual environment and install the prerequesites.

## Running

Run the app by:
```
streamlit run app.py
```

## Built With

* [Streamlit](https://www.streamlit.io/) - ML app framework 
* [pandas](https://pandas.pydata.org/) - Data handling 
* [NumPy](https://numpy.org/) - Mathematical functions 
* [pydeck](https://pypi.org/project/pydeck/) - Data visualization 

## License

This project is licensed under the GPLv3 License
