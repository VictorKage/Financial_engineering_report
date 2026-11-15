import numpy as np
import pandas as pd

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

        print(f"Intial length of self.df is: {self.df.shape[0]}")
        print(self.df.head())
        print(self.df.columns.tolist())

    def remove_0_bid_prices(self):
        self.df = self.df[self.df["best_bid"] != 0]
        print(f"Length of self.df after removing 0 prices bid options: {self.df.shape[0]}")

    def remove_itm_prices(self):
        # removing in-the-money Call options
        self.df = self.df[~((self.df["cp_flag"] == "C") & (self.df["strike"] < self.spy_close))]
        # removing in-the-money Put options
        self.df = self.df[~((self.df["cp_flag"] == "P") & (self.df["strike"] > self.spy_close))]
        print(f"Length of self.df after removing itm options: {self.df.shape[0]}")

    def find_vertical_spread_arbitrage(
        self,
        price_col="mid",
        cp_flag="C",
        tolerance=1e-8
    ):
        """
        Calls:
            Q = (C(K1) - C(K2)) / (K2 - K1)

        Puts:
            Q = (P(K2) - P(K1)) / (K2 - K1)

        No-arbitrage condition:
            0 <= Q <= 1
        """

        flagged = set()

        group_cols = ["date", "exdate", "style"]

        df = self.df[self.df["cp_flag"] == cp_flag]

        for _, group in df.groupby(group_cols, dropna=False):

            group = group.sort_values("strike").dropna(
                subset=["strike", price_col]
            )

            strikes = group["strike"].to_numpy()
            prices = group[price_col].to_numpy()
            indices = group.index.to_numpy()

            for i in range(len(group) - 1):

                k1, k2 = strikes[i:i+2]
                p1, p2 = prices[i:i+2]

                if k2 <= k1:
                    continue

                if cp_flag == "C":
                    Q = (p1 - p2) / (k2 - k1)

                elif cp_flag == "P":
                    Q = (p2 - p1) / (k2 - k1)

                else:
                    raise ValueError("cp_flag must be C or P")

                if Q < -tolerance or Q > 1 + tolerance:
                    flagged.update(indices[i:i+2])

        return flagged

    def find_butterfly_spread_arbitrage(
        self,
        price_col="mid",
        cp_flag="C",
        tolerance=1e-8
    ):
        """
        Butterfly condition for both calls and puts:

        P1 - [(K3-K1)/(K3-K2)]*P2
           + [(K2-K1)/(K3-K2)]*P3 >= 0
        """

        flagged = set()

        group_cols = ["date", "exdate", "style"]

        df = self.df[self.df["cp_flag"] == cp_flag]

        for _, group in df.groupby(group_cols, dropna=False):

            group = group.sort_values("strike").dropna(
                subset=["strike", price_col]
            )

            strikes = group["strike"].to_numpy()
            prices = group[price_col].to_numpy()
            indices = group.index.to_numpy()

            for i in range(len(group) - 2):

                k1, k2, k3 = strikes[i:i+3]
                p1, p2, p3 = prices[i:i+3]

                if not (k1 < k2 < k3):
                    continue

                butterfly_price = (
                    p1
                    - ((k3-k1)/(k3-k2))*p2
                    + ((k2-k1)/(k3-k2))*p3
                )

                if butterfly_price < -tolerance:
                    flagged.update(indices[i:i+3])

        return flagged

    def find_calendar_spread_arbitrage(
        self,
        price_col="mid",
        cp_flag="C",
        tolerance=1e-8
    ):
        """
        Calendar condition for both calls and puts:

            Price(K,T2) - Price(K,T1) >= 0

        where T2 > T1.

        Assumes zero interest rates and dividends.
        """

        flagged = set()

        group_cols = ["date", "strike", "style"]

        df = self.df[self.df["cp_flag"] == cp_flag]

        for _, group in df.groupby(group_cols, dropna=False):

            group = group.sort_values("exdate").dropna(
                subset=["exdate", price_col]
            )

            prices = group[price_col].to_numpy()
            indices = group.index.to_numpy()

            for i in range(len(group) - 1):

                price_difference = prices[i+1] - prices[i]

                if price_difference < -tolerance:
                    flagged.update(indices[i:i+2])

        return flagged

    def remove_static_arbitrage(
        self,
        price_col="mid",
        tolerance=0
    ):
        """
        Remove vertical, butterfly and calendar spread
        arbitrage from calls and puts.

        Returns:
            cleaned_df, report
        """

        self.df = self.df.copy().reset_index(drop=True)

        all_flags = set()
        report = {}

        # Process calls and puts independently
        for option_type in ["C", "P"]:

            vertical_flags = self.find_vertical_spread_arbitrage(
                price_col=price_col,
                cp_flag=option_type,
                tolerance=tolerance
            )

            butterfly_flags = self.find_butterfly_spread_arbitrage(
                price_col=price_col,
                cp_flag=option_type,
                tolerance=tolerance
            )

            calendar_flags = self.find_calendar_spread_arbitrage(
                price_col=price_col,
                cp_flag=option_type,
                tolerance=tolerance
            )

            option_flags = (
                vertical_flags
                | butterfly_flags
                | calendar_flags
            )

            all_flags.update(option_flags)

            report[option_type] = {
                "vertical_spread": len(vertical_flags),
                "butterfly_spread": len(butterfly_flags),
                "calendar_spread": len(calendar_flags),
                "total_unique_removed": len(option_flags)
            }

        cleaned_df = self.df.loc[
            ~self.df.index.isin(all_flags)
        ].copy()

        report["total_unique_observations_removed"] = len(all_flags)

        return cleaned_df, report

    def remove_arbitrage_prices(self):

        cleaned_df, report = self.remove_static_arbitrage()

        print("\nStatic Arbitrage Removal Report")
        print("--------------------------------")

        for option_type in ["C", "P"]:
            print(f"\n{option_type} options:")
            for test, count in report[option_type].items():
                print(f"{test}: {count}")

        print(
            "\nTotal unique observations removed:",
            report["total_unique_observations_removed"]
        )

        return cleaned_df

    def clean_data_for_calibration(self):
        self.remove_0_bid_prices()
        self.remove_itm_prices()
        self.remove_arbitrage_prices()
        
        return self.df, self.spy_close

    def clean_data_for_put_call_parity(self, n: int = 5):
        self.df_sorted = self.df.assign(
            strike_diff=abs(self.df["strike"] - self.spy_close)
        ).sort_values(by=["exdate", "strike_diff"])

        # Select n unique nearest strikes per exdate
        nearest_strikes = (
            self.df_sorted.groupby("exdate")["strike"]
            .unique()
            .apply(lambda strikes: strikes[:n*2])
            .explode()
            .reset_index()
        )

        # Merge back to get both Call and Put records for those strikes
        pc_self.df = pd.merge(self.df, nearest_strikes, on=["exdate", "strike"])

        return pc_self.df, self.spy_close



if __name__ == "__main__":
    cleaner = DataCleaner()
    test, spy_close_test = cleaner.clean_data_for_calibration()



