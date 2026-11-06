import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self):
        sheet = pd.read_excel("Data20191113.xlsx", header=None)
        self.spy_close = sheet.iat[2, 6]
        self.df = sheet.iloc[6:].copy()
        self.df.columns = sheet.iloc[5].tolist()


    def remove_0_bid_prices(self):
        self.df = self.df[self.df["best_bid"] != 0]

    def remove_itm_prices(self):
        # removing in-the-money Call options
        self.df = self.df[~((self.df["cp_flag"] == "C") & (self.df["Strike"] < self.spy_close))]

        # removing in-the-money Put options
        self.df = self.df[~((self.df["cp_flag"] == "P") & (self.df["Strike"] > self.spy_close))]

    def remove_arbitrage_prices(self):
        pass

    def clean_data_for_calibration(self) -> pd.DataFrame:
        self.df = self.remove_0_bid_prices()
        self.df = self.remove_itm_prices()
        self.df = self.remove_arbitrage_prices()
        
        return self.df

    def clean_data_for_put_call_parity(self, n: int = 5) -> pd.DataFrame:
        # n is the n-th nearest strike prices nearest to the spot price
        # For all Maturity dates, we will select the n nearest strike prices to the spot price
        self.grouped_df = self.df.groupby("exdate")
        PC_df = self.grouped_df.apply(lambda x: x.sort_values(by="Strike", key=lambda y: abs(y - self.spy_close)))
        # selecting the n nearest strike prices
        PC_df = PC_df.groupby("exdate").head(n)
        # ungrouping the dataframe
        PC_df = PC_df.reset_index(drop=True)
        return PC_df



if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data_for_put_call_parity()
    print(cleaned_data)


