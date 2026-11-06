import numpy as np
import pandas as pd
import datetime as dt

class DataCleaner:
    def __init__(self):
        sheet = pd.read_excel("Data20191113.xlsx", header=None)

        self.spy_close = sheet.iat[2, 6]
        self.df = sheet.iloc[6:].copy()
        self.df.columns = sheet.iloc[5].tolist()

        # Clean column names
        self.df.columns = (
            self.df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Convert dates
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df["exdate"] = pd.to_datetime(self.df["exdate"], errors="coerce")

        self.df["time_to_maturity"] = (
            self.df["exdate"] - self.df["date"]
        ).dt.days / 365 # er i tvivl her


    def remove_0_bid_prices(self):
        self.df = self.df[self.df["best_bid"] != 0]

    def remove_itm_prices(self):
        # removing in-the-money Call options
        print(self.df.shape[0])
        self.df = self.df[~((self.df["cp_flag"] == "C") & (self.df["strike"] < self.spy_close))]
        print(self.df.shape[0])
        # removing in-the-money Put options
        self.df = self.df[~((self.df["cp_flag"] == "P") & (self.df["strike"] > self.spy_close))]
        print(self.df.shape[0])
        
    def remove_arbitrage_prices(self):
        pass

    def clean_data_for_calibration(self):
        self.remove_0_bid_prices()
        self.remove_itm_prices()
        #self.remove_arbitrage_prices()
        
        return self.df, self.spy_close

    def clean_data_for_put_call_parity(self, n: int = 5):
        df_sorted = self.df.assign(
            strike_diff=abs(self.df["strike"] - self.spy_close)
        ).sort_values(by=["exdate", "strike_diff"])

        # Select n unique nearest strikes per exdate
        nearest_strikes = (
            df_sorted.groupby("exdate")["strike"]
            .unique()
            .apply(lambda strikes: strikes[:n*2])
            .explode()
            .reset_index()
        )

        # Merge back to get both Call and Put records for those strikes
        pc_df = pd.merge(self.df, nearest_strikes, on=["exdate", "strike"])

        return pc_df, self.spy_close



if __name__ == "__main__":
    cleaner = DataCleaner()
    #cleaned_data, spy_close = cleaner.clean_data_for_put_call_parity()
    #print(cleaned_data)
    #print(spy_close)

    test, spy_close_test = cleaner.clean_data_for_calibration()
    print(test)
    print(spy_close_test)


