import logging

import requests
import pandas as pd

from ats.util import logging_utils

class MacroDataBuilder:
    def __init__(
        self,
        api_key: str = "5e7a5f80ba03257b5913e97bec786466",
        base_url: str = "https://api.stlouisfed.org/fred/series/observations?series_id=",
    ):
        self.api_key = api_key
        self.base_url = base_url

    def most_recent_fetch(self, api_key: str, base_url: str):
        # FRED Series IDs for each economic indicator
        series_ids = {
            "GDP": "GDP",
            "Unemployment Rate": "UNRATE",
            "Inflation Rate": "T10YIE",  # 10-Year Breakeven Inflation Rate
            "Interest Rates": "GS10",  # 10-Year Treasury Constant Maturity Rate
            "Housing Market Indicator": "HOUST",  # Housing Starts: Total: New Privately Owned Housing Units Started
        }

        rows = []

        for name, series_id in series_ids.items():
            url = f"{base_url}{series_id}&api_key={api_key}&file_type=json"
            response = requests.get(url).json()

            data = response["observations"]
            df = pd.DataFrame(data)

            df["date"] = pd.to_datetime(df["date"])

            most_recent_data = df.loc[df["date"].idxmax()]
            print(name)

            rows.append(
                {
                    "date": most_recent_data["date"],
                    "indicator": name,
                    "indicator_value": most_recent_data["value"],
                }
            )

        return pd.DataFrame(rows)

    def full_fetch(self, api_key: str, base_url: str):
        # FRED Series IDs for each economic indicator
        series_ids = {
            "GDP": "GDP",
            "Unemployment Rate": "UNRATE",
            "Inflation Rate": "T10YIE",  # 10-Year Breakeven Inflation Rate
            "Interest Rates": "GS10",  # 10-Year Treasury Constant Maturity Rate
            "Housing Market Indicator": "HOUST",  # Housing Starts: Total: New Privately Owned Housing Units Started
        }

        rows = []

        for name, series_id in series_ids.items():
            url = f"{base_url}{series_id}&api_key={api_key}&file_type=json"
            response = requests.get(url).json()

            data = response["observations"]
            df = pd.DataFrame(data)

            df["date"] = pd.to_datetime(df["date"])

            rows.append(
                {"date": df["date"], "indicator": name, "indicator_value": df["value"]}
            )

        return pd.DataFrame(rows)


def add_macro_indicators(api_key, base_url):
    macro_information = MacroDataBuilder(api_key, base_url)
    macro_data = macro_information.full_fetch(api_key, base_url)

    return macro_data


if __name__ == "__main__":
    logging_utils.init_logging()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    indicators_df = add_macro_indicators(api_key="5e7a5f80ba03257b5913e97bec786466",
                                         base_url="https://api.stlouisfed.org/fred/series/observations?series_id=")
    logging.info(f"indicators_df:{indicators_df}")
    indicators_df = indicators_df.apply(lambda x: x.explode()).reset_index(drop=True)
    logging.info(f"indicators_df_explode:{indicators_df}")
    indicators_df["date"] = pd.to_datetime(indicators_df["date"])
    indicators_df_pivot = indicators_df.pivot_table(
        index="date", columns="indicator", values="indicator_value", aggfunc="first"
    )
    logging.info(f"pivot:{indicators_df}")
    indicators_df_pivot.reset_index(inplace=True)
    #    data_df = pd.merge(data_df, indicators_df_pivot, on='date', how='left')
